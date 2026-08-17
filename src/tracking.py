"""Experiment tracking.

Every training run logs its parameters, metrics and artifacts to MLflow, so comparing runs
is a query rather than a diff of JSON files.

Why this exists
---------------
The first version of this project wrote `artifacts/metrics.json` per run. That is adequate
for exactly one run. The moment the same training job was invoked three times with
different search strategies, output directories started accumulating - `artifacts/`,
`artifacts_tuned/`, `artifacts_verify/` - which is a naming convention standing in for run
history. That substitution is the signal that experiment tracking has stopped being a
nice-to-have.

`metrics.json` is still written. It travels with the model artifact, is diffable in version
control, and does not require MLflow to be installed to read. MLflow adds what a file
cannot: history across runs, comparison, and lineage from a served model back to the run
that produced it.

Design notes
------------
**Local file store by default.** ``mlruns/`` needs no server, so a contributor gets
tracking by cloning rather than by provisioning. Point ``MLFLOW_TRACKING_URI`` at a shared
server (or a Vertex AI / Databricks endpoint) and the same code logs there instead - that
is the migration, and it is an environment variable.

**Nested runs.** One parent run per training invocation, one child per candidate model. A
flat structure would make "which models were compared in the run that produced the deployed
artifact?" unanswerable, which is the question that matters during an incident.

**Optional dependency.** Every entry point here degrades to a no-op when MLflow is not
installed, so training never fails because of the tracking layer. A tool that can break the
thing it observes is a liability.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_EXPERIMENT = "campaign-group-predictor"

#: Default tracking backend: a local SQLite database.
#:
#: Two reasons this is not the plain ``./mlruns`` file store.
#:
#: First, ``mlflow ui`` defaults to ``sqlite:///mlflow.db``. Writing runs to the file store
#: while the UI reads from SQLite produces an empty dashboard and no error - a silent
#: mismatch that costs an afternoon to diagnose. Matching the tool's own default removes
#: the trap.
#:
#: Second, the MLflow **Model Registry** requires a database-backed store; the file store
#: cannot support it. Starting on SQLite means model versioning and stage promotion are
#: available without a migration later.
#:
#: Overridden with ``--tracking-uri`` or ``MLFLOW_TRACKING_URI`` to point at a shared
#: server. That is the whole migration path.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _mlflow() -> Any | None:
    """Return the mlflow module, or None when it is not installed."""
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def is_available() -> bool:
    """Whether experiment tracking will actually record anything."""
    return _mlflow() is not None


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Flatten nested metric dictionaries into `parent.child` keys.

    MLflow metrics are scalar. `metrics.json` is deeply nested (per-class precision inside
    per-class inside a model), and flattening rather than dropping keeps the whole structure
    queryable - `per_class.0.recall` becomes a metric that can be plotted across runs, which
    is the class this model struggles with.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    # `bool` is a subclass of `int`, so this branch covers True/False as 1.0/0.0 without a
    # separate case. `value == value` is False only for NaN, which MLflow rejects - a
    # string or a NaN reaching log_metrics would fail the run at its final step, after all
    # the expensive work.
    elif isinstance(value, (int, float)) and value == value:
        out[prefix] = float(value)


def start_training_run(
    run_name: str,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
) -> Any | None:
    """Open a parent run for one invocation of the training script.

    Returns the mlflow module when tracking is active, otherwise ``None`` - so callers pass
    the result around and never import mlflow themselves.

    Paired with :func:`end_training_run` rather than exposed as a context manager. A
    context manager would require the whole of ``train()`` to be re-indented inside a
    ``with`` block, and reformatting two hundred lines of working code to accommodate an
    observability layer is the wrong trade. MLflow's fluent API is built around a global
    active run precisely for this.
    """
    mlflow = _mlflow()
    if mlflow is None:
        logger.info(
            "MLflow is not installed; continuing without experiment tracking. "
            'Install it with: pip install -e ".[tracking]"'
        )
        return None

    # Precedence: explicit argument, then MLFLOW_TRACKING_URI, then the local default.
    #
    # The middle term was missing. This module's docstring says "point MLFLOW_TRACKING_URI at
    # a shared server and the same code logs there instead - that is the migration", but
    # `tracking_uri or DEFAULT_TRACKING_URI` overwrote the environment variable with the
    # hardcoded local SQLite path on every call. MLflow reads that variable itself; calling
    # set_tracking_uri unconditionally is what silenced it.
    #
    # So the documented migration path did not work, and it failed in the quiet direction: a
    # team pointing at a shared server would get a successful run logged to a file on the
    # training machine, with no error and an empty shared dashboard.
    resolved_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment)

    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    logger.info(
        "MLflow run %s in experiment '%s' (%s)",
        run.info.run_id[:8],
        experiment,
        mlflow.get_tracking_uri(),
    )
    return mlflow


def end_training_run(mlflow: Any | None, status: str = "FINISHED") -> None:
    """Close the parent run.

    Call from a ``finally`` block. A run left open is recorded as RUNNING forever, which
    makes "did this training job succeed?" unanswerable from the tracking UI - the one
    question it exists to answer.
    """
    if mlflow is None:
        return
    mlflow.end_run(status=status)


@contextmanager
def model_run(mlflow: Any | None, model_name: str) -> Iterator[Any | None]:
    """Open a child run for one candidate model inside a training invocation."""
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.set_tag("model_name", model_name)
        yield mlflow


def log_params(mlflow: Any | None, params: dict[str, Any]) -> None:
    """Record the configuration a run was executed with."""
    if mlflow is None:
        return
    mlflow.log_params({key: str(value) for key, value in params.items() if value is not None})


def log_metrics(mlflow: Any | None, metrics: dict[str, Any]) -> None:
    """Record every scalar in a nested metrics dictionary."""
    if mlflow is None:
        return
    flat: dict[str, float] = {}
    _flatten("", metrics, flat)
    if flat:
        mlflow.log_metrics(flat)


def log_artifact(mlflow: Any | None, path: str | Path) -> None:
    """Attach a file to the run. Silent when the file does not exist."""
    if mlflow is None:
        return
    artifact = Path(path)
    if artifact.exists():
        mlflow.log_artifact(str(artifact))


def log_model(mlflow: Any | None, pipeline: Any, name: str = "model") -> None:
    """Register the fitted pipeline with the run.

    Serialisation format
    --------------------
    MLflow's default sklearn serialiser refuses to write types it does not recognise -
    here ``src.features.PairwiseFeatureBuilder`` and ``numpy.dtype`` - because
    deserialising an arbitrary class is arbitrary code execution. That refusal is correct
    behaviour, not a bug: a model registry is a place other people load things from.

    ``cloudpickle`` is used instead, which serialises the custom transformer by value. The
    trade is explicit: the artifact can now only be loaded somewhere the class definition
    is importable, and loading it executes code. That is already true of the ``model.pkl``
    the service loads, so this adds no new exposure - and the alternative, adding the
    project's own classes to a trusted-types allowlist, grants the same permission while
    making it look like a configuration detail.

    Wrapped in a guard regardless. A training run that has already produced a valid
    artifact on disk must not fail because the tracking layer could not serialise it.
    """
    if mlflow is None:
        return
    try:
        # Imported under an alias, not as `import mlflow.sklearn`.
        #
        # That form binds the top-level name `mlflow`, which shadows this function's own
        # `mlflow` parameter. It happens to work - the module and the passed handle are the
        # same object - so the bug is invisible at runtime and only appears as a mypy
        # redefinition error. It would stop working the moment a caller passed anything
        # other than the module itself, which is exactly what the test suite's recording
        # stub does.
        from mlflow import sklearn as mlflow_sklearn

        mlflow_sklearn.log_model(
            pipeline,
            name=name,
            serialization_format=mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    except Exception as error:  # noqa: BLE001 - tracking must never break training
        logger.warning(
            "Could not log the model to MLflow (%s). Training is unaffected: the artifact "
            "on disk and metrics.json are both written.",
            error,
        )


__all__ = [
    "DEFAULT_EXPERIMENT",
    "DEFAULT_TRACKING_URI",
    "end_training_run",
    "is_available",
    "log_artifact",
    "log_metrics",
    "log_model",
    "log_params",
    "model_run",
    "start_training_run",
]
