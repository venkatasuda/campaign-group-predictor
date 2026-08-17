"""Shared pytest fixtures.

A tiny synthetic dataset stands in for ``customerGroups.csv`` so the suite runs fast,
deterministically, and without shipping proprietary data in the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import tracking
from src.api.main import create_app
from src.config import Settings
from src.constants import (
    BASE_FEATURES,
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
    TARGET_COLUMN,
)
from src.predictors import MajorityClassPredictor
from src.registry import ModelRegistry
from src.training.pipeline import build_pipeline, candidate_models

RANDOM_STATE = 7


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator."""
    return np.random.default_rng(RANDOM_STATE)


@pytest.fixture
def raw_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic dataset with the same column contract as ``customerGroups.csv``."""
    n_rows = 300
    data: dict[str, Any] = {column: rng.normal(size=n_rows) for column in BASE_FEATURES}

    # Make the target learnable: group 1 wins when its first features dominate.
    signal = data["g1_1"] + data["g1_2"] - data["g2_1"] - data["g2_2"]
    target = np.where(signal > 0.6, 1, np.where(signal < -0.6, 2, 0))
    data[TARGET_COLUMN] = target

    # Post-campaign columns: perfectly correlated with the target (i.e. leakage).
    for column in LEAKAGE_FEATURES:
        data[column] = target.astype(float) + rng.normal(scale=0.01, size=n_rows)

    return pd.DataFrame(data)


@pytest.fixture
def feature_frame(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix restricted to the 67 legitimate pre-campaign columns."""
    return raw_dataset[BASE_FEATURES].copy()


@pytest.fixture
def valid_payload() -> dict[str, dict[str, float]]:
    """A well-formed ``/predict`` request body."""
    return {
        "group_1": dict.fromkeys(GROUP_1_FEATURES, 1.0),
        "group_2": dict.fromkeys(GROUP_2_FEATURES, 0.5),
        "comparison": dict.fromkeys(COMPARISON_FEATURES, 0.25),
    }


@pytest.fixture
def trained_artifact(raw_dataset: pd.DataFrame, tmp_path: Path) -> Path:
    """Train a small pipeline and persist it as a joblib artifact."""
    import joblib

    features = raw_dataset[BASE_FEATURES]
    target = raw_dataset[TARGET_COLUMN]

    pipeline = build_pipeline(candidate_models(RANDOM_STATE)["logistic_regression"])
    pipeline.fit(features, target)

    path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": "logistic_regression",
            "model_version": "test",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"f1_macro": 0.9},
        },
        path,
    )
    return path


@pytest.fixture(autouse=True)
def clean_registry() -> Any:
    """Reset the registry singleton around every test so state never leaks."""
    ModelRegistry.reset_instance()
    yield
    ModelRegistry.reset_instance()


@pytest.fixture(autouse=True)
def no_mlflow_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the suite with experiment tracking switched off.

    Without this, every test that calls ``train()`` logs into the developer's REAL tracking
    store - ``sqlite:///mlflow.db`` and ``./mlruns/`` in the repository root - because that is
    ``DEFAULT_TRACKING_URI`` and the tests never override it.

    Three separate problems, in increasing order of seriousness:

    1. **The store grows without bound.** Each training test writes ~100 metric rows, a run
       directory, and a pickled model. Repeated over a week of development this reached
       297 MB of ``mlruns/`` plus a 21 MB ``mlflow.db``, and contributed to filling the disk -
       at which point pytest failed with ``sqlite3.OperationalError: database or disk is
       full`` and pandas with ``[Errno 28]``, in two tests that had nothing to do with either.

    2. **Test runs pollute real experiment history.** Runs trained on the 300-row synthetic
       fixture sit in the same experiment as runs trained on the actual dataset, with
       plausible-looking metrics. Anyone reading that dashboard to compare champions is
       reading a mix of two different things.

    3. **Tests are not hermetic.** A suite whose result depends on free space in a directory
       outside the test environment is not reproducible, and it fails confusingly.

    Disabling the layer rather than redirecting it, after two redirections failed:

    - A **tmp SQLite URI** relocates run metadata but not artifacts. ``train.py`` calls
      ``log_artifact`` and ``log_model``, and with a database-backed store MLflow still
      resolves the artifact root to ``./mlruns`` relative to the working directory - so the
      pickled models keep landing in the repository.
    - A **tmp file-store URI** relocates both, but MLflow now refuses it outright:
      ``MlflowException: The filesystem tracking backend ... is in maintenance mode``. Opting
      back in via ``MLFLOW_ALLOW_FILE_STORE`` would pin the suite to a deprecated backend.

    ``_mlflow()`` returning ``None`` is the module's own documented degradation path - the
    same one that runs when MLflow is not installed, and the one
    ``TestTrackingDegradesGracefully`` already covers. Every ``tracking.*`` entry point
    no-ops, so ``train()`` is exercised end to end with nothing written anywhere.

    This does not reduce coverage: the MLflow logging branch in ``train()`` was never
    asserted on, only executed for its side effects, and README already records it as one of
    the untested paths in that module. Tracking behaviour itself is covered by the stub-based
    tests in ``test_logging_and_tracking.py``, which do not touch a real backend.
    """
    monkeypatch.setattr(tracking, "_mlflow", lambda: None)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a non-existent artifact, with fallback enabled."""
    return Settings(
        model_path=str(tmp_path / "missing.pkl"),
        model_type="sklearn_pipeline",
        allow_baseline_fallback=True,
        log_level="WARNING",
        # Deterministic decisions in tests: no exploration overrides.
        exploration_rate=0.0,
    )


@pytest.fixture
def client(settings: Settings) -> Any:
    """TestClient backed by the baseline predictor (no artifact required)."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_with_model(settings: Settings) -> Any:
    """TestClient whose predictor is overridden with a deterministic stub."""
    from src.api.dependencies import get_predictor

    app = create_app(settings)
    app.dependency_overrides[get_predictor] = lambda: MajorityClassPredictor(
        majority_class=2,
        class_distribution={"no_group_profitable": 0.1, "group_1": 0.3, "group_2": 0.6},
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def settings_with_decisions(settings: Settings) -> Settings:
    """Settings with the cost-sensitive decision layer explicitly switched on.

    The layer defaults to **off** because it converts probabilities into a recommended
    action using cost values nobody has supplied - the dataset contains no monetary figures.
    Tests that exercise the layer therefore have to ask for it, and that is the point of a
    separate fixture rather than flipping the base one: the default fixture keeps testing
    the behaviour that actually ships, and any test needing the layer states so at its own
    call site.
    """
    return settings.model_copy(update={"enable_decision_layer": True})


@pytest.fixture
def client_with_decisions(settings_with_decisions: Settings) -> Any:
    """TestClient with a deterministic predictor AND the decision layer enabled."""
    from src.api.dependencies import get_predictor

    app = create_app(settings_with_decisions)
    app.dependency_overrides[get_predictor] = lambda: MajorityClassPredictor(
        majority_class=2,
        class_distribution={"no_group_profitable": 0.1, "group_1": 0.3, "group_2": 0.6},
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
