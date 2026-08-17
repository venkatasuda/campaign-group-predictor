"""Unit tests for structured logging and experiment tracking.

Both modules are observability rather than behaviour, which makes them easy to leave
untested and easy to get quietly wrong. Two failure modes justify the tests:

* A log formatter that raises turns a logging call into an outage. It runs inside the
  request path, so an unserialisable value in an ``extra=`` dictionary must degrade, not
  propagate.
* A tracking layer that raises turns a *successful* training run into a failed one. The
  layer that observes a process must never be able to break it.

Neither requires a tracking server. The MLflow calls are exercised against a recording
stub, which tests the code paths in this repository rather than MLflow's.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src import tracking
from src.logging_config import CloudLoggingFormatter, _use_json_logs, configure_logging


def _record(level: int = logging.INFO, message: str = "hello", **extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="src.test",
        level=level,
        pathname=__file__,
        lineno=42,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestCloudLoggingFormatter:
    def test_emits_a_single_line_of_valid_json(self) -> None:
        """Cloud Logging parses one JSON object per line; anything else is opaque text."""
        output = CloudLoggingFormatter().format(_record())

        assert "\n" not in output
        payload = json.loads(output)
        assert payload["message"] == "hello"
        assert payload["logger"] == "src.test"

    @pytest.mark.parametrize(
        ("level", "severity"),
        [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ],
    )
    def test_maps_python_levels_to_cloud_logging_severities(self, level, severity) -> None:
        """Without a recognised `severity`, every line is DEFAULT and an alert on errors
        cannot be expressed at all - which is the main reason for structured logs here."""
        payload = json.loads(CloudLoggingFormatter().format(_record(level=level)))
        assert payload["severity"] == severity

    def test_promotes_extra_fields_to_queryable_keys(self) -> None:
        """`request_id` must be a field, not text inside the message.

        `jsonPayload.request_id="abc"` retrieving every line from one request is the whole
        point of the correlation ID; interpolating it into the message would require a
        substring search instead.
        """
        payload = json.loads(
            CloudLoggingFormatter().format(_record(request_id="abc123", latency_ms=12.5))
        )
        assert payload["request_id"] == "abc123"
        assert payload["latency_ms"] == 12.5

    def test_does_not_leak_internal_logrecord_attributes(self) -> None:
        payload = json.loads(CloudLoggingFormatter().format(_record()))
        for reserved in ("msg", "args", "levelno", "pathname", "created"):
            assert reserved not in payload

    def test_includes_source_location(self) -> None:
        payload = json.loads(CloudLoggingFormatter().format(_record()))
        assert "42" in payload["source"]

    def test_survives_a_value_that_cannot_be_serialised(self) -> None:
        """A log call must never be the thing that breaks a request.

        `default=str` means an exotic value degrades to its repr rather than raising
        inside the request path.
        """

        class Unserialisable:
            def __repr__(self) -> str:
                return "<opaque>"

        payload = json.loads(CloudLoggingFormatter().format(_record(thing=Unserialisable())))
        assert payload["thing"] == "<opaque>"

    def test_records_exception_text_when_present(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = _record(level=logging.ERROR)
            record.exc_info = sys.exc_info()

        payload = json.loads(CloudLoggingFormatter().format(record))
        assert "ValueError" in payload["exception"]


class TestLogFormatSelection:
    def test_json_on_cloud_run(self, monkeypatch) -> None:
        """`K_SERVICE` is injected by the Cloud Run runtime, so the default is correct in
        both environments without anyone having to set anything."""
        monkeypatch.delenv("JSON_LOGS", raising=False)
        monkeypatch.setenv("K_SERVICE", "campaign-api")
        assert _use_json_logs() is True

    def test_text_locally(self, monkeypatch) -> None:
        monkeypatch.delenv("JSON_LOGS", raising=False)
        monkeypatch.delenv("K_SERVICE", raising=False)
        assert _use_json_logs() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
    def test_explicit_override_wins(self, monkeypatch, value) -> None:
        monkeypatch.delenv("K_SERVICE", raising=False)
        monkeypatch.setenv("JSON_LOGS", value)
        assert _use_json_logs() is True

    def test_reconfiguring_replaces_the_formatter(self, monkeypatch) -> None:
        """uvicorn installs its own handler. Leaving it in place would mean application
        lines are structured while server lines are not - the worst of both."""
        monkeypatch.setenv("JSON_LOGS", "true")
        configure_logging("INFO")
        configure_logging("DEBUG")

        handlers = logging.getLogger().handlers
        assert handlers
        assert all(isinstance(h.formatter, CloudLoggingFormatter) for h in handlers)


class _StubMlflow:
    """Records calls instead of contacting a tracking server."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[str] = []

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def log_metrics(self, values: dict[str, float]) -> None:
        self.metrics.update(values)

    def log_params(self, values: dict[str, str]) -> None:
        self.params.update(values)

    def set_tags(self, values: dict[str, str]) -> None:
        self.tags.update(values)


class TestTrackingDegradesGracefully:
    """Every entry point must be a no-op when MLflow is absent.

    A training run that fails because the layer observing it is not installed would make
    tracking a liability rather than an aid.
    """

    def test_no_call_raises_without_mlflow(self) -> None:
        tracking.log_params(None, {"a": 1})
        tracking.log_metrics(None, {"b": 2.0})
        tracking.log_artifact(None, "does/not/exist.json")
        tracking.log_model(None, object())
        tracking.end_training_run(None)

        with tracking.model_run(None, "any-model") as child:
            assert child is None

    def test_start_returns_none_when_mlflow_is_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(tracking, "_mlflow", lambda: None)
        assert tracking.start_training_run("run") is None
        assert tracking.is_available() is False


class TestArtifactLogging:
    """``log_artifact`` attaches a file that exists and stays silent about one that does not.

    The missing-file case is not defensive padding. ``train()`` logs ``metrics.json`` right
    after writing it, and a run configured with ``--fast`` or interrupted before that write
    would otherwise raise from the tracking layer - which is precisely the failure mode the
    module promises not to have.
    """

    def test_an_existing_file_is_attached(self, tmp_path) -> None:
        stub = _StubMlflow()
        path = tmp_path / "metrics.json"
        path.write_text("{}", encoding="utf-8")

        tracking.log_artifact(stub, path)

        assert stub.artifacts == [str(path)]

    def test_a_missing_file_is_skipped_silently(self, tmp_path) -> None:
        stub = _StubMlflow()

        tracking.log_artifact(stub, tmp_path / "never-written.json")

        assert stub.artifacts == []


class TestModelLogging:
    """``log_model`` uses the cloudpickle flavour, and never propagates a failure.

    Patching ``mlflow.sklearn.log_model`` rather than starting a run: the function under test
    takes the handle only to decide whether tracking is active, and reaches the real
    ``mlflow.sklearn`` module by import. So the behaviour is fully exercisable without a
    tracking backend, a run, or a byte written to disk.
    """

    def test_serialises_with_cloudpickle(self, monkeypatch) -> None:
        mlflow_sklearn = pytest.importorskip("mlflow.sklearn")
        recorded: dict[str, Any] = {}

        def fake_log_model(pipeline: Any, *, name: str, serialization_format: str) -> None:
            recorded.update(pipeline=pipeline, name=name, serialization_format=serialization_format)

        monkeypatch.setattr(mlflow_sklearn, "log_model", fake_log_model)
        pipeline = object()

        tracking.log_model(_StubMlflow(), pipeline, name="champion")

        assert recorded["pipeline"] is pipeline
        assert recorded["name"] == "champion"
        # Not the default serialiser: it refuses PairwiseFeatureBuilder, because
        # deserialising an arbitrary class is arbitrary code execution.
        assert recorded["serialization_format"] == mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE

    def test_a_serialisation_failure_never_reaches_the_caller(self, monkeypatch, caplog) -> None:
        """The guarantee that matters: a training run that has already written a valid
        artifact to disk must not fail because the tracking layer could not serialise it."""
        mlflow_sklearn = pytest.importorskip("mlflow.sklearn")

        def explode(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("unserialisable estimator")

        monkeypatch.setattr(mlflow_sklearn, "log_model", explode)

        with caplog.at_level(logging.WARNING):
            tracking.log_model(_StubMlflow(), object())

        assert "Training is unaffected" in caplog.text


class TestTrackingUriPrecedence:
    """An explicit argument wins, then MLFLOW_TRACKING_URI, then the local default.

    The middle rule is the one worth testing, because it was broken and broke quietly.
    ``src/tracking.py`` documents MLFLOW_TRACKING_URI as the whole migration path to a shared
    server, but the code called ``set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)`` -
    overwriting the variable MLflow would otherwise have honoured. A team pointing at a
    shared server would have seen a successful training run, no error, and an empty
    dashboard, because the run landed in a SQLite file on the training machine.
    """

    class _UriRecorder:
        def __init__(self) -> None:
            self.uri: str | None = None
            self.info = type("Info", (), {"run_id": "0" * 32})()

        def set_tracking_uri(self, uri: str) -> None:
            self.uri = uri

        def set_experiment(self, name: str) -> None:  # noqa: D102 - stub
            pass

        def start_run(self, run_name: str) -> Any:
            return type("Run", (), {"info": self.info})()

        def set_tags(self, values: dict[str, str]) -> None:  # noqa: D102 - stub
            pass

        def get_tracking_uri(self) -> str | None:
            return self.uri

    @pytest.fixture
    def recorder(self, monkeypatch) -> Any:
        stub = self._UriRecorder()
        monkeypatch.setattr(tracking, "_mlflow", lambda: stub)
        return stub

    def test_environment_variable_is_honoured(self, recorder, monkeypatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
        tracking.start_training_run("run")
        assert recorder.uri == "http://mlflow.internal:5000"

    def test_explicit_argument_beats_the_environment(self, recorder, monkeypatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
        tracking.start_training_run("run", tracking_uri="sqlite:///explicit.db")
        assert recorder.uri == "sqlite:///explicit.db"

    def test_falls_back_to_the_local_default(self, recorder, monkeypatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        tracking.start_training_run("run")
        assert recorder.uri == tracking.DEFAULT_TRACKING_URI


class TestMetricFlattening:
    def test_nested_metrics_become_dotted_keys(self) -> None:
        """metrics.json nests per-class scores three deep. Flattening rather than dropping
        keeps `per_class.0.recall` plottable across runs - which is the class this model
        struggles with, so it is the one worth tracking."""
        stub = _StubMlflow()
        tracking.log_metrics(
            stub, {"accuracy": 0.57, "per_class": {"0": {"recall": 0.12, "f1": 0.18}}}
        )

        assert stub.metrics["accuracy"] == pytest.approx(0.57)
        assert stub.metrics["per_class.0.recall"] == pytest.approx(0.12)

    def test_booleans_are_recorded_as_numbers(self) -> None:
        stub = _StubMlflow()
        tracking.log_metrics(stub, {"significantly_positive": True})
        assert stub.metrics["significantly_positive"] == 1.0

    def test_non_numeric_and_nan_values_are_skipped(self) -> None:
        """MLflow metrics are scalar. A string or NaN would raise on submission, so they
        are dropped here rather than allowed to fail a run at its final step."""
        stub = _StubMlflow()
        tracking.log_metrics(stub, {"model": "mlp", "cv_std": float("nan"), "accuracy": 0.57})

        assert "model" not in stub.metrics
        assert "cv_std" not in stub.metrics
        assert stub.metrics["accuracy"] == pytest.approx(0.57)

    def test_params_are_stringified_and_none_dropped(self) -> None:
        stub = _StubMlflow()
        tracking.log_params(stub, {"n_iter": 40, "search": None, "tune": True})

        assert stub.params == {"n_iter": "40", "tune": "True"}
