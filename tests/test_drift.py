"""Unit tests for drift detection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.constants import BASE_FEATURES, TARGET_COLUMN
from src.drift import (
    PSI_MAJOR,
    PSI_MODERATE,
    DriftReference,
    build_reference,
    detect_drift,
    drift_summary,
    population_stability_index,
)


class TestBuildReference:
    def test_covers_every_feature(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        assert len(reference.features) == len(BASE_FEATURES)

    def test_records_the_row_count(self, feature_frame: pd.DataFrame) -> None:
        assert build_reference(feature_frame).n_rows == len(feature_frame)

    def test_expected_fractions_sum_to_one(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        for feature in reference.features.values():
            assert sum(feature.expected_fractions) == pytest.approx(1.0, abs=1e-9)

    def test_outer_bins_are_open(self, feature_frame: pd.DataFrame) -> None:
        # Unseen extremes must fall into the end bins rather than being dropped.
        feature = build_reference(feature_frame).features["g1_1"]
        assert feature.bin_edges[0] == float("-inf")
        assert feature.bin_edges[-1] == float("inf")

    def test_captures_the_target_distribution(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        reference = build_reference(feature_frame, raw_dataset[TARGET_COLUMN])
        # Shares are rounded to 6 decimal places for readable JSON, so the sum can drift
        # by up to n_classes x 5e-7. The tolerance accounts for that rounding rather than
        # demanding exactness the stored representation deliberately does not have.
        assert sum(reference.target_distribution.values()) == pytest.approx(1.0, abs=1e-4)

    def test_handles_a_constant_feature(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_1"] = 3.0
        reference = build_reference(frame)
        assert "c_1" in reference.features

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_reference(pd.DataFrame(columns=BASE_FEATURES))

    def test_round_trips_through_json(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame.head(50))
        restored = DriftReference.from_dict(json.loads(json.dumps(reference.to_dict())))
        assert restored.n_rows == reference.n_rows
        assert set(restored.features) == set(reference.features)


class TestPopulationStabilityIndex:
    def test_identical_data_scores_near_zero(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        psi = population_stability_index(reference.features["g1_1"], feature_frame["g1_1"])
        assert psi < 1e-6

    def test_shifted_data_scores_high(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        shifted = feature_frame["g1_1"] + 5.0  # far outside the training range
        assert population_stability_index(reference.features["g1_1"], shifted) > PSI_MAJOR

    def test_psi_is_non_negative(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        rng = np.random.default_rng(0)
        for scale in (0.5, 1.0, 2.0):
            values = pd.Series(rng.normal(scale=scale, size=200))
            assert population_stability_index(reference.features["g1_2"], values) >= -1e-9

    def test_larger_shift_gives_larger_psi(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        small = population_stability_index(reference.features["g1_1"], feature_frame["g1_1"] + 0.3)
        large = population_stability_index(reference.features["g1_1"], feature_frame["g1_1"] + 3.0)
        assert large > small

    def test_empty_input_returns_zero(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        assert population_stability_index(reference.features["g1_1"], pd.Series([])) == 0.0

    def test_empty_bins_do_not_produce_infinity(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        # All mass in one place - most reference bins are empty in the actual data.
        constant = pd.Series(np.full(100, 10.0))
        assert np.isfinite(population_stability_index(reference.features["g1_1"], constant))


class TestDetectDrift:
    def test_same_distribution_reports_no_drift(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        report = detect_drift(reference, feature_frame)
        assert (report["severity"] == "none").all()

    def test_shifted_features_are_flagged(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        drifted = feature_frame.copy()
        drifted["g1_1"] = drifted["g1_1"] + 6.0

        report = detect_drift(reference, drifted)
        assert report.loc["g1_1", "severity"] == "major"
        assert report.index[0] == "g1_1"  # sorted worst-first

    def test_reports_current_and_reference_means(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        report = detect_drift(reference, feature_frame)
        assert "reference_mean" in report.columns
        assert "current_mean" in report.columns

    def test_missing_columns_are_skipped(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        report = detect_drift(reference, feature_frame.drop(columns=["c_1", "c_2"]))
        assert "c_1" not in report.index
        assert len(report) == len(BASE_FEATURES) - 2

    def test_unrelated_frame_returns_empty_report(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        report = detect_drift(reference, pd.DataFrame({"unrelated": [1.0, 2.0]}))
        assert report.empty

    def test_tracks_a_change_in_missing_rate(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        drifted = feature_frame.copy()
        drifted.loc[drifted.index[:100], "g2_5"] = None

        report = detect_drift(reference, drifted)
        assert (
            report.loc["g2_5", "current_missing_rate"]
            > report.loc["g2_5", "reference_missing_rate"]
        )


class TestDriftSummary:
    def test_no_drift_recommends_no_action(self, feature_frame: pd.DataFrame) -> None:
        summary = drift_summary(build_reference(feature_frame), feature_frame)
        assert summary["action"] == "none"
        assert summary["n_major"] == 0

    def test_major_drift_recommends_retraining(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        drifted = feature_frame.copy()
        for column in ("g1_1", "g1_2", "g2_1"):
            drifted[column] = drifted[column] + 6.0

        summary = drift_summary(reference, drifted)
        assert summary["action"] == "retrain"
        assert summary["n_major"] >= 3

    def test_lists_the_worst_features(self, feature_frame: pd.DataFrame) -> None:
        reference = build_reference(feature_frame)
        drifted = feature_frame.copy()
        drifted["c_9"] = drifted["c_9"] + 8.0
        assert "c_9" in drift_summary(reference, drifted)["worst_features"]

    def test_summary_is_serialisable(self, feature_frame: pd.DataFrame) -> None:
        summary = drift_summary(build_reference(feature_frame), feature_frame)
        assert "max_psi" in json.loads(json.dumps(summary))

    def test_empty_report_is_handled(self, feature_frame: pd.DataFrame) -> None:
        summary = drift_summary(build_reference(feature_frame), pd.DataFrame({"x": [1.0]}))
        assert summary["action"] == "none"
        assert summary["n_features_checked"] == 0

    def test_thresholds_are_ordered(self) -> None:
        assert 0 < PSI_MODERATE < PSI_MAJOR
