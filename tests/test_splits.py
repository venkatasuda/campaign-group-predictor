"""Unit tests for splitting strategies and dataset diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import TARGET_COLUMN
from src.training.splits import (
    comparison_redundancy_report,
    detect_time_ordering,
    make_split,
)


class TestDetectTimeOrdering:
    def test_finds_a_planted_monotonic_column(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_1"] = np.arange(len(frame), dtype=float)

        evidence = detect_time_ordering(frame)
        flagged = [item.column for item in evidence if item.looks_like_time]
        assert "c_1" in flagged

    def test_planted_column_ranks_first(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_2"] = np.arange(len(frame), dtype=float)
        assert detect_time_ordering(frame)[0].column == "c_2"

    def test_random_data_yields_no_candidate(self, feature_frame: pd.DataFrame) -> None:
        evidence = detect_time_ordering(feature_frame)
        assert not any(item.looks_like_time for item in evidence)

    def test_reverse_sorted_column_is_also_detected(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_3"] = np.arange(len(frame), 0, -1, dtype=float)
        flagged = [item.column for item in detect_time_ordering(frame) if item.looks_like_time]
        assert "c_3" in flagged

    def test_constant_columns_are_skipped(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_4"] = 1.0
        assert "c_4" not in [item.column for item in detect_time_ordering(frame)]

    def test_evidence_is_serialisable(self, feature_frame: pd.DataFrame) -> None:
        payload = detect_time_ordering(feature_frame)[0].to_dict()
        assert set(payload) >= {"column", "monotonic_fraction", "looks_like_time"}


class TestMakeSplit:
    def test_stratified_preserves_class_proportions(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        target = raw_dataset[TARGET_COLUMN]
        _, _, y_train, y_test = make_split(feature_frame, target, strategy="stratified")

        train_share = y_train.value_counts(normalize=True).sort_index()
        test_share = y_test.value_counts(normalize=True).sort_index()
        np.testing.assert_allclose(train_share.to_numpy(), test_share.to_numpy(), atol=0.06)

    def test_stratified_respects_test_size(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        _, x_test, _, _ = make_split(feature_frame, raw_dataset[TARGET_COLUMN], test_size=0.25)
        assert len(x_test) == pytest.approx(0.25 * len(feature_frame), abs=2)

    def test_temporal_split_keeps_row_order(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        x_train, x_test, _, _ = make_split(
            feature_frame, raw_dataset[TARGET_COLUMN], strategy="temporal"
        )
        assert x_train.index.max() < x_test.index.min()

    def test_temporal_split_has_no_overlap(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        x_train, x_test, _, _ = make_split(
            feature_frame, raw_dataset[TARGET_COLUMN], strategy="temporal"
        )
        assert not set(x_train.index).intersection(x_test.index)

    def test_temporal_split_orders_by_requested_column(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        frame = feature_frame.copy()
        frame["c_1"] = np.arange(len(frame), 0, -1, dtype=float)

        x_train, x_test, _, _ = make_split(
            frame, raw_dataset[TARGET_COLUMN], strategy="temporal", order_by="c_1"
        )
        assert x_train["c_1"].max() <= x_test["c_1"].min()

    def test_unknown_order_by_column_raises(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        with pytest.raises(KeyError, match="order_by"):
            make_split(
                feature_frame,
                raw_dataset[TARGET_COLUMN],
                strategy="temporal",
                order_by="nope",
            )

    def test_degenerate_test_size_raises(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            make_split(
                feature_frame,
                raw_dataset[TARGET_COLUMN],
                strategy="temporal",
                test_size=0.0,
            )


class TestComparisonRedundancyReport:
    def test_flags_a_planted_deterministic_feature(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_1"] = 2.0 * frame["g1_4"] - 3.0 * frame["g2_4"]

        report = comparison_redundancy_report(frame)
        assert report.loc["c_1", "is_redundant"]
        assert report.loc["c_1", "r2"] == pytest.approx(1.0, abs=1e-6)

    def test_independent_feature_is_not_flagged(self, feature_frame: pd.DataFrame) -> None:
        report = comparison_redundancy_report(feature_frame)
        assert not report["is_redundant"].any()

    def test_covers_every_comparison_feature(self, feature_frame: pd.DataFrame) -> None:
        assert len(comparison_redundancy_report(feature_frame)) == 27

    def test_sorted_by_r2_descending(self, feature_frame: pd.DataFrame) -> None:
        values = comparison_redundancy_report(feature_frame)["r2"].to_numpy()
        assert (np.diff(values) <= 1e-9).all()

    def test_frame_without_group_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="no g1_/g2_"):
            comparison_redundancy_report(pd.DataFrame({"c_1": [1.0, 2.0, 3.0]}))
