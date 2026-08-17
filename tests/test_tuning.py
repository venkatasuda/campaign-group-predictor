"""Unit tests for the hyperparameter search module."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.constants import TARGET_COLUMN
from src.training.pipeline import build_pipeline, candidate_models
from src.training.tuning import TuningResult, search_spaces, tune_pipeline


class TestSearchSpaces:
    def test_covers_every_tunable_model_in_the_zoo(self) -> None:
        spaces = search_spaces()
        zoo = candidate_models(fast=True)
        # Every installed model should either have a space or be intentionally absent.
        assert set(zoo).issubset(set(spaces))

    def test_every_parameter_targets_the_model_step(self) -> None:
        for space in search_spaces().values():
            assert all(key.startswith("model__") for key in space)

    def test_spaces_are_non_empty(self) -> None:
        assert all(len(space) > 0 for space in search_spaces().values())

    def test_includes_a_neural_baseline(self) -> None:
        assert "mlp" in search_spaces()


class TestTunePipeline:
    @pytest.fixture
    def small_data(self, raw_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        from src.constants import BASE_FEATURES

        subset = raw_dataset.head(120)
        return subset[BASE_FEATURES], subset[TARGET_COLUMN]

    def test_returns_a_fitted_pipeline(self, small_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = small_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])

        best, result = tune_pipeline(
            pipeline,
            "logistic_regression",
            features,
            target,
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
            n_iter=2,
            n_jobs=1,
        )

        assert isinstance(best, Pipeline)
        assert len(best.predict(features.head(3))) == 3
        assert result.tuned is True

    def test_reports_the_best_score_and_params(
        self, small_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, target = small_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])

        _, result = tune_pipeline(
            pipeline,
            "logistic_regression",
            features,
            target,
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
            n_iter=2,
            n_jobs=1,
        )

        assert 0.0 <= result.best_score <= 1.0
        assert result.n_candidates == 2
        assert any(key.startswith("model__") for key in result.best_params)

    def test_unknown_model_falls_back_to_defaults(
        self, small_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, target = small_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])

        best, result = tune_pipeline(pipeline, "not_in_the_zoo", features, target, n_jobs=1)

        assert result.tuned is False
        assert result.best_params == {}
        assert len(best.predict(features.head(2))) == 2

    def test_result_is_json_serialisable(self) -> None:
        result = TuningResult(
            model_name="mlp",
            tuned=True,
            best_score=0.71,
            best_params={"model__hidden_layer_sizes": (128, 64)},
            n_candidates=25,
        )
        payload = json.dumps(result.to_dict())
        assert "hidden_layer_sizes" in payload
        assert json.loads(payload)["best_params"]["model__hidden_layer_sizes"] == [128, 64]
