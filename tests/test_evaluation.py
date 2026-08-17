"""Unit tests for the evaluation and business-impact logic (ML questions 1 and 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.evaluation import (
    bootstrap_lift_interval,
    campaign_outcome_distribution,
    classification_metrics,
    estimate_business_lift,
)


class TestCampaignOutcomeDistribution:
    def test_percentages_sum_to_one_hundred(self) -> None:
        target = pd.Series([0] * 20 + [1] * 50 + [2] * 30)
        distribution = campaign_outcome_distribution(target)
        total = sum(item["percentage"] for item in distribution["by_class"].values())
        assert total == pytest.approx(100.0)

    def test_reports_the_expected_percentages(self) -> None:
        target = pd.Series([0] * 20 + [1] * 50 + [2] * 30)
        by_class = campaign_outcome_distribution(target)["by_class"]
        assert by_class["group_1"]["percentage"] == pytest.approx(50.0)
        assert by_class["group_2"]["percentage"] == pytest.approx(30.0)
        assert by_class["no_group_profitable"]["percentage"] == pytest.approx(20.0)

    def test_absent_classes_are_reported_as_zero(self) -> None:
        by_class = campaign_outcome_distribution(pd.Series([1, 1, 1]))["by_class"]
        assert by_class["no_group_profitable"]["count"] == 0
        assert by_class["group_2"]["percentage"] == pytest.approx(0.0)

    def test_counts_every_campaign(self) -> None:
        assert campaign_outcome_distribution(pd.Series([0, 1, 2]))["total_campaigns"] == 3


class TestClassificationMetrics:
    def test_perfect_predictions_score_one(self) -> None:
        y = np.array([0, 1, 2, 1, 2])
        metrics = classification_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_confusion_matrix_is_three_by_three(self) -> None:
        metrics = classification_metrics(np.array([0, 1, 2]), np.array([1, 1, 2]))
        assert np.array(metrics["confusion_matrix"]).shape == (3, 3)

    def test_reports_balanced_accuracy(self) -> None:
        metrics = classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]))
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


class TestEstimateBusinessLift:
    def test_perfect_model_beats_every_naive_strategy(self) -> None:
        y = np.array([0] * 10 + [1] * 45 + [2] * 45)
        lift = estimate_business_lift(y, y)
        assert lift.model_success_rate == 1.0
        assert lift.absolute_lift_pp > 0

    def test_random_baseline_is_half_the_profitable_share(self) -> None:
        y = np.array([0] * 20 + [1] * 40 + [2] * 40)
        lift = estimate_business_lift(y, y)
        assert lift.random_choice_success_rate == pytest.approx(0.4)

    def test_always_group_one_baseline_matches_its_share(self) -> None:
        y = np.array([1] * 60 + [2] * 40)
        lift = estimate_business_lift(y, np.ones_like(y))
        assert lift.always_group_1_success_rate == pytest.approx(0.6)
        assert lift.best_naive_strategy == "always_group_1"

    def test_always_decline_baseline_is_reported(self) -> None:
        y = np.array([0] * 50 + [1] * 25 + [2] * 25)
        lift = estimate_business_lift(y, y)
        assert lift.always_decline_success_rate == pytest.approx(0.5)

    def test_always_decline_can_be_the_strongest_baseline(self) -> None:
        # Most campaigns were unprofitable: doing nothing beats every targeting rule.
        y = np.array([0] * 70 + [1] * 15 + [2] * 15)
        lift = estimate_business_lift(y, y)
        assert lift.best_naive_strategy == "always_decline"
        assert lift.best_naive_success_rate == pytest.approx(0.7)

    def test_lift_is_measured_against_the_decline_baseline_when_it_wins(self) -> None:
        y_true = np.array([0] * 70 + [1] * 15 + [2] * 15)
        y_pred = np.zeros_like(y_true)  # always declines - matches the baseline exactly
        lift = estimate_business_lift(y_true, y_pred)
        assert lift.absolute_lift_pp == pytest.approx(0.0, abs=1e-6)

    def test_targeting_baselines_still_win_when_campaigns_pay_off(self) -> None:
        y = np.array([0] * 5 + [1] * 60 + [2] * 35)
        assert estimate_business_lift(y, y).best_naive_strategy == "always_group_1"

    def test_counts_avoided_wasted_campaigns(self) -> None:
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 2])
        lift = estimate_business_lift(y_true, y_pred)
        assert lift.wasted_campaigns_avoided == 1
        assert lift.wasted_campaign_avoidance_rate == pytest.approx(0.5)

    def test_avoidance_rate_is_zero_when_no_class_zero_exists(self) -> None:
        y = np.array([1, 2, 1, 2])
        assert estimate_business_lift(y, y).wasted_campaign_avoidance_rate == 0.0

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            estimate_business_lift(np.array([]), np.array([]))

    def test_result_is_serialisable(self) -> None:
        y = np.array([0, 1, 2])
        assert "model_success_rate" in estimate_business_lift(y, y).to_dict()


class TestColumnVectorPredictions:
    """CatBoost returns (n, 1) for multiclass; every metric must be shape-independent."""

    @staticmethod
    def _both_shapes(y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return y_pred, y_pred.reshape(-1, 1)

    def test_lift_is_identical_for_flat_and_column_predictions(self) -> None:
        y_true = np.array([0] * 30 + [1] * 50 + [2] * 20)
        flat, column = self._both_shapes(y_true.copy())
        assert estimate_business_lift(y_true, flat).to_dict() == (
            estimate_business_lift(y_true, column).to_dict()
        )

    def test_column_predictions_do_not_produce_negative_lift(self) -> None:
        # The exact regression this guards: broadcasting turned a good model's lift
        # negative, because (n,) == (n, 1) yields an (n, n) matrix.
        y_true = np.array([0] * 25 + [1] * 46 + [2] * 29)
        lift = estimate_business_lift(y_true, y_true.reshape(-1, 1))
        assert lift.model_success_rate == 1.0
        assert lift.absolute_lift_pp > 0

    def test_classification_metrics_accept_column_predictions(self) -> None:
        y_true = np.array([0, 1, 2, 1, 2, 0])
        assert classification_metrics(y_true, y_true.reshape(-1, 1))["accuracy"] == 1.0

    def test_bootstrap_accepts_column_predictions(self) -> None:
        y_true = np.array([0] * 20 + [1] * 40 + [2] * 40)
        result = bootstrap_lift_interval(y_true, y_true.reshape(-1, 1), n_resamples=100)
        assert result["point_estimate_pp"] > 0

    def test_length_mismatch_is_caught(self) -> None:
        with pytest.raises(ValueError, match="entries but"):
            estimate_business_lift(np.array([0, 1, 2]), np.array([0, 1]))


class TestBootstrapLiftInterval:
    def test_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=400)
        y_pred = np.where(rng.random(400) < 0.75, y_true, rng.integers(0, 3, size=400))

        result = bootstrap_lift_interval(y_true, y_pred, n_resamples=300)
        assert result["ci_lower_pp"] <= result["point_estimate_pp"] <= result["ci_upper_pp"]

    def test_a_strong_model_is_significantly_positive(self) -> None:
        y_true = np.array([0] * 100 + [1] * 150 + [2] * 150)
        result = bootstrap_lift_interval(y_true, y_true, n_resamples=300)
        assert result["significantly_positive"] is True
        assert result["ci_lower_pp"] > 0

    def test_a_useless_model_is_not_significantly_positive(self) -> None:
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 3, size=300)
        y_pred = np.ones_like(y_true)  # always predicts group 1
        result = bootstrap_lift_interval(y_true, y_pred, n_resamples=300)
        assert result["significantly_positive"] is False

    def test_interval_is_ordered(self) -> None:
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 3, size=200)
        result = bootstrap_lift_interval(y_true, y_true, n_resamples=200)
        assert result["ci_lower_pp"] <= result["ci_upper_pp"]

    def test_more_data_gives_a_narrower_interval(self) -> None:
        rng = np.random.default_rng(5)
        small_true = rng.integers(0, 3, size=100)
        large_true = rng.integers(0, 3, size=1500)

        small = bootstrap_lift_interval(small_true, small_true, n_resamples=300)
        large = bootstrap_lift_interval(large_true, large_true, n_resamples=300)
        assert large["standard_error_pp"] < small["standard_error_pp"]

    def test_success_rate_interval_is_reported(self) -> None:
        y_true = np.array([0, 1, 2] * 40)
        result = bootstrap_lift_interval(y_true, y_true, n_resamples=200)
        lower, upper = result["model_success_rate_ci"]
        assert 0.0 <= lower <= upper <= 1.0

    def test_confidence_level_is_recorded(self) -> None:
        y_true = np.array([0, 1, 2] * 20)
        result = bootstrap_lift_interval(y_true, y_true, n_resamples=100, confidence_level=0.9)
        assert result["confidence_level"] == 0.9
        assert result["n_resamples"] == 100

    def test_is_reproducible(self) -> None:
        y_true = np.array([0, 1, 2] * 30)
        first = bootstrap_lift_interval(y_true, y_true, n_resamples=100, random_state=7)
        second = bootstrap_lift_interval(y_true, y_true, n_resamples=100, random_state=7)
        assert first == second

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_lift_interval(np.array([]), np.array([]))

    def test_invalid_confidence_level_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_lift_interval(np.array([0, 1]), np.array([0, 1]), confidence_level=1.5)
