"""Unit tests for probability calibration and its quality metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    CalibrationReport,
    calibrate_pipeline,
    calibration_report,
    compare_calibration_methods,
)
from src.constants import BASE_FEATURES, TARGET_COLUMN
from src.training.pipeline import build_pipeline, candidate_models


def _one_hot_probabilities(truth: np.ndarray, confidence: float) -> np.ndarray:
    """Probabilities that put ``confidence`` on the true class."""
    remainder = (1.0 - confidence) / 2.0
    proba = np.full((truth.size, 3), remainder)
    proba[np.arange(truth.size), truth] = confidence
    return proba


class TestCalibrationReport:
    def test_perfectly_confident_and_correct_is_well_calibrated(self) -> None:
        truth = np.array([0, 1, 2] * 20)
        report = calibration_report(truth, _one_hot_probabilities(truth, 0.98))
        assert report.expected_calibration_error < 0.05
        assert "well calibrated" in report.summary()

    def test_overconfident_and_wrong_is_flagged(self) -> None:
        truth = np.array([0] * 60)
        # Claims 98% on class 1 while the truth is always class 0.
        proba = np.tile([0.01, 0.98, 0.01], (60, 1))
        report = calibration_report(truth, proba)
        assert report.expected_calibration_error > 0.9
        assert "POORLY CALIBRATED" in report.summary()

    def test_brier_is_zero_for_perfect_probabilities(self) -> None:
        truth = np.array([0, 1, 2, 1])
        proba = _one_hot_probabilities(truth, 1.0)
        assert calibration_report(truth, proba).brier_multiclass == pytest.approx(0.0, abs=1e-9)

    def test_reports_per_class_brier(self) -> None:
        truth = np.array([0, 1, 2] * 10)
        report = calibration_report(truth, _one_hot_probabilities(truth, 0.9))
        assert set(report.per_class_brier) == {"no_group_profitable", "group_1", "group_2"}

    def test_reliability_curve_bins_are_ordered(self) -> None:
        rng = np.random.default_rng(0)
        proba = rng.dirichlet(np.ones(3), size=300)
        truth = np.array([rng.choice(3, p=row) for row in proba])
        curve = calibration_report(truth, proba, n_bins=10).reliability_curve
        assert all(
            curve[index]["bin_lower"] <= curve[index + 1]["bin_lower"]
            for index in range(len(curve) - 1)
        )

    def test_max_calibration_error_is_at_least_ece(self) -> None:
        rng = np.random.default_rng(1)
        proba = rng.dirichlet(np.ones(3), size=200)
        truth = np.array([rng.choice(3, p=row) for row in proba])
        report = calibration_report(truth, proba)
        assert report.max_calibration_error >= report.expected_calibration_error - 1e-9

    def test_rejects_one_dimensional_probabilities(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            calibration_report(np.array([0, 1]), np.array([0.5, 0.5]))

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            calibration_report(np.array([0, 1, 2]), np.full((2, 3), 1 / 3))

    def test_rejects_empty_sample(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            calibration_report(np.array([]), np.zeros((0, 3)))

    def test_report_is_serialisable(self) -> None:
        truth = np.array([0, 1, 2] * 5)
        payload = json.loads(
            json.dumps(calibration_report(truth, _one_hot_probabilities(truth, 0.8)).to_dict())
        )
        assert "expected_calibration_error" in payload

    def test_dataclass_summary_mentions_sample_size(self) -> None:
        report = CalibrationReport(
            n_samples=42,
            log_loss=0.5,
            brier_multiclass=0.2,
            expected_calibration_error=0.02,
            max_calibration_error=0.05,
            per_class_brier={},
        )
        assert "42 samples" in report.summary()


class TestCalibratePipeline:
    @pytest.fixture
    def split_data(self, raw_dataset: pd.DataFrame):
        """Train / calibration / test splits from the synthetic dataset."""
        features = raw_dataset[BASE_FEATURES]
        target = raw_dataset[TARGET_COLUMN]
        return (
            (features.iloc[:150], target.iloc[:150]),
            (features.iloc[150:220], target.iloc[150:220]),
            (features.iloc[220:], target.iloc[220:]),
        )

    def test_returns_an_estimator_with_predict_proba(self, split_data) -> None:
        (x_fit, y_fit), (x_cal, y_cal), (x_eval, _) = split_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])
        pipeline.fit(x_fit, y_fit)

        calibrated = calibrate_pipeline(pipeline, x_cal, y_cal)
        proba = calibrated.predict_proba(x_eval)

        assert proba.shape == (len(x_eval), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_calibrated_predictions_are_valid_classes(self, split_data) -> None:
        (x_fit, y_fit), (x_cal, y_cal), (x_eval, _) = split_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])
        pipeline.fit(x_fit, y_fit)

        predictions = calibrate_pipeline(pipeline, x_cal, y_cal).predict(x_eval)
        assert set(np.unique(predictions)).issubset({0, 1, 2})

    def test_sigmoid_method_also_works(self, split_data) -> None:
        (x_fit, y_fit), (x_cal, y_cal), (x_eval, _) = split_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])
        pipeline.fit(x_fit, y_fit)

        calibrated = calibrate_pipeline(pipeline, x_cal, y_cal, method="sigmoid")
        assert calibrated.predict_proba(x_eval).shape[1] == 3

    def test_compare_methods_recommends_one(self, split_data) -> None:
        (x_fit, y_fit), (x_cal, y_cal), (x_eval, y_eval) = split_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])
        pipeline.fit(x_fit, y_fit)

        results = compare_calibration_methods(pipeline, x_cal, y_cal, x_eval, y_eval)

        assert results["recommended"] in {"uncalibrated", "isotonic", "sigmoid"}
        assert results["uncalibrated"]["n_samples"] == len(x_eval)

    def test_compare_methods_output_is_serialisable(self, split_data) -> None:
        (x_fit, y_fit), (x_cal, y_cal), (x_eval, y_eval) = split_data
        pipeline = build_pipeline(candidate_models(fast=True)["logistic_regression"])
        pipeline.fit(x_fit, y_fit)

        results = compare_calibration_methods(pipeline, x_cal, y_cal, x_eval, y_eval)
        assert "recommended" in json.loads(json.dumps(results))
