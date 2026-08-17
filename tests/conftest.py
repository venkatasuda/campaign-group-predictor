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
