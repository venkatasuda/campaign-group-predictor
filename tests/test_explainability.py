"""Unit tests for the explainability module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import BASE_FEATURES, TARGET_COLUMN
from src.explainability import (
    _mean_absolute_shap,
    explain,
    permutation_importance_report,
    transform_without_model,
    transformed_feature_names,
)
from src.training.pipeline import build_pipeline, candidate_models


@pytest.fixture
def fitted_pipeline(raw_dataset: pd.DataFrame):
    """A small fitted pipeline for explainability tests."""
    features = raw_dataset[BASE_FEATURES].head(150)
    target = raw_dataset[TARGET_COLUMN].head(150)
    pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])
    pipeline.fit(features, target)
    return pipeline, features, target


class TestTransformHelpers:
    def test_feature_names_include_engineered_columns(self, fitted_pipeline) -> None:
        pipeline, features, _ = fitted_pipeline
        names = transformed_feature_names(pipeline, features)
        assert "diff_1" in names
        assert "ratio_1" in names
        assert len(names) == len(BASE_FEATURES) + 40

    def test_transform_without_model_matches_name_count(self, fitted_pipeline) -> None:
        pipeline, features, _ = fitted_pipeline
        transformed = transform_without_model(pipeline, features.head(5))
        assert transformed.shape == (5, len(transformed_feature_names(pipeline, features)))

    def test_transform_output_is_finite(self, fitted_pipeline) -> None:
        pipeline, features, _ = fitted_pipeline
        assert np.isfinite(transform_without_model(pipeline, features.head(20))).all()


class TestPermutationImportance:
    def test_covers_every_raw_feature(self, fitted_pipeline) -> None:
        pipeline, features, target = fitted_pipeline
        report = permutation_importance_report(pipeline, features, target, n_repeats=2)
        assert len(report) == len(BASE_FEATURES)

    def test_is_sorted_most_important_first(self, fitted_pipeline) -> None:
        pipeline, features, target = fitted_pipeline
        report = permutation_importance_report(pipeline, features, target, n_repeats=2)
        values = report["importance_mean"].to_numpy()
        assert (np.diff(values) <= 1e-9).all()

    def test_ranks_are_consecutive(self, fitted_pipeline) -> None:
        pipeline, features, target = fitted_pipeline
        report = permutation_importance_report(pipeline, features, target, n_repeats=2)
        assert report["rank"].tolist() == list(range(1, len(report) + 1))

    def test_top_n_truncates(self, fitted_pipeline) -> None:
        pipeline, features, target = fitted_pipeline
        report = permutation_importance_report(pipeline, features, target, n_repeats=2, top_n=5)
        assert len(report) == 5

    def test_signal_features_outrank_noise(self, fitted_pipeline) -> None:
        # The synthetic target depends on g1_1, g1_2, g2_1, g2_2 only.
        pipeline, features, target = fitted_pipeline
        report = permutation_importance_report(pipeline, features, target, n_repeats=5)
        signal_ranks = [report.loc[name, "rank"] for name in ("g1_1", "g1_2", "g2_1", "g2_2")]
        assert min(signal_ranks) <= 15


class TestMeanAbsoluteShap:
    def test_handles_two_dimensional_values(self) -> None:
        values = np.array([[1.0, -2.0], [3.0, -4.0]])
        np.testing.assert_allclose(_mean_absolute_shap(values), [2.0, 3.0])

    def test_handles_three_dimensional_multiclass_values(self) -> None:
        values = np.ones((10, 4, 3))
        assert _mean_absolute_shap(values).shape == (4,)

    def test_rejects_unexpected_shape(self) -> None:
        with pytest.raises(ValueError, match="Unexpected SHAP"):
            _mean_absolute_shap(np.ones((2, 2, 2, 2)))


class TestExplain:
    def test_always_returns_permutation_importance(self, fitted_pipeline) -> None:
        pipeline, features, target = fitted_pipeline
        block = explain(pipeline, features, target, top_n=5)
        assert len(block["permutation_importance_top"]) == 5
        assert "feature" in block["permutation_importance_top"][0]

    def test_shap_key_is_always_present(self, fitted_pipeline) -> None:
        pipeline, features, target = fitted_pipeline
        assert "shap_importance_top" in explain(pipeline, features, target, top_n=3)

    def test_survives_a_broken_shap_installation(self, fitted_pipeline, monkeypatch) -> None:
        import src.explainability as module

        def _boom(*args, **kwargs):
            raise RuntimeError("shap exploded")

        monkeypatch.setattr(module, "shap_importance_report", _boom)
        block = explain(fitted_pipeline[0], fitted_pipeline[1], fitted_pipeline[2], top_n=3)
        assert block["shap_importance_top"] is None
        assert block["permutation_importance_top"]


class TestShapOptional:
    def test_summary_plot_writes_a_file_when_shap_is_available(
        self, fitted_pipeline, tmp_path: Path
    ) -> None:
        pytest.importorskip("shap")
        from src.explainability import shap_summary_plot

        pipeline, features, _ = fitted_pipeline
        path = shap_summary_plot(pipeline, features, tmp_path / "shap.png", max_samples=40)
        assert path.exists() and path.stat().st_size > 0

    def test_importance_report_ranks_features_when_shap_is_available(self, fitted_pipeline) -> None:
        pytest.importorskip("shap")
        from src.explainability import shap_importance_report

        pipeline, features, _ = fitted_pipeline
        report = shap_importance_report(pipeline, features, max_samples=40, top_n=10)
        assert len(report) == 10
        assert report["mean_abs_shap"].is_monotonic_decreasing
