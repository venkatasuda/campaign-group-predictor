"""Unit tests for feature engineering and payload adaptation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    BASE_FEATURES,
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
)
from src.exceptions import InvalidFeaturePayloadError
from src.features import FeatureTransformer, PairwiseFeatureBuilder, drop_leakage_columns


class TestDropLeakageColumns:
    def test_removes_all_post_campaign_columns(self, raw_dataset: pd.DataFrame) -> None:
        cleaned = drop_leakage_columns(raw_dataset)
        assert not set(LEAKAGE_FEATURES).intersection(cleaned.columns)

    def test_keeps_every_legitimate_column(self, raw_dataset: pd.DataFrame) -> None:
        cleaned = drop_leakage_columns(raw_dataset)
        assert set(BASE_FEATURES).issubset(cleaned.columns)

    def test_is_a_noop_when_columns_are_absent(self, feature_frame: pd.DataFrame) -> None:
        assert list(drop_leakage_columns(feature_frame).columns) == list(feature_frame.columns)


class TestPairwiseFeatureBuilder:
    def test_adds_difference_and_ratio_per_pair(self, feature_frame: pd.DataFrame) -> None:
        builder = PairwiseFeatureBuilder()
        out = builder.fit_transform(feature_frame)
        assert out.shape[1] == feature_frame.shape[1] + 40  # 20 diffs + 20 ratios

    def test_difference_is_arithmetically_correct(self) -> None:
        frame = pd.DataFrame({"g1_1": [5.0], "g2_1": [2.0]})
        out = PairwiseFeatureBuilder(add_ratios=False).fit_transform(frame)
        assert out["diff_1"].iloc[0] == pytest.approx(3.0)

    def test_ratio_is_arithmetically_correct(self) -> None:
        frame = pd.DataFrame({"g1_1": [6.0], "g2_1": [3.0]})
        out = PairwiseFeatureBuilder(add_differences=False).fit_transform(frame)
        assert out["ratio_1"].iloc[0] == pytest.approx(2.0, rel=1e-4)

    def test_division_by_zero_does_not_raise(self) -> None:
        frame = pd.DataFrame({"g1_1": [1.0], "g2_1": [0.0]})
        out = PairwiseFeatureBuilder().fit_transform(frame)
        assert np.isfinite(out["ratio_1"].iloc[0]) or np.isnan(out["ratio_1"].iloc[0])

    def test_can_disable_both_blocks(self, feature_frame: pd.DataFrame) -> None:
        builder = PairwiseFeatureBuilder(add_differences=False, add_ratios=False)
        out = builder.fit_transform(feature_frame)
        assert out.shape[1] == feature_frame.shape[1]

    def test_column_order_is_stable_between_fit_and_transform(
        self, feature_frame: pd.DataFrame
    ) -> None:
        builder = PairwiseFeatureBuilder().fit(feature_frame)
        first = builder.transform(feature_frame.head(5))
        second = builder.transform(feature_frame.tail(5))
        assert list(first.columns) == list(second.columns)

    def test_get_feature_names_out_matches_transform(self, feature_frame: pd.DataFrame) -> None:
        builder = PairwiseFeatureBuilder().fit(feature_frame)
        assert list(builder.get_feature_names_out()) == list(
            builder.transform(feature_frame).columns
        )

    def test_rejects_non_dataframe_input(self) -> None:
        with pytest.raises(InvalidFeaturePayloadError):
            PairwiseFeatureBuilder().fit(np.zeros((3, 3)))


class TestFeatureTransformer:
    def test_builds_single_row_in_canonical_order(
        self, valid_payload: dict[str, dict[str, float]]
    ) -> None:
        frame = FeatureTransformer().from_payload(**valid_payload)
        assert frame.shape == (1, len(BASE_FEATURES))
        assert list(frame.columns) == BASE_FEATURES

    def test_builds_batch(self, valid_payload: dict[str, dict[str, float]]) -> None:
        payloads = [
            (valid_payload["group_1"], valid_payload["group_2"], valid_payload["comparison"])
        ] * 3
        assert FeatureTransformer().from_payloads(payloads).shape[0] == 3

    def test_empty_batch_is_rejected(self) -> None:
        with pytest.raises(InvalidFeaturePayloadError):
            FeatureTransformer().from_payloads([])

    def test_missing_key_is_rejected(self, valid_payload: dict[str, dict[str, float]]) -> None:
        del valid_payload["group_1"]["g1_5"]
        with pytest.raises(InvalidFeaturePayloadError, match="missing required key"):
            FeatureTransformer().from_payload(**valid_payload)

    def test_unexpected_key_is_rejected(self, valid_payload: dict[str, dict[str, float]]) -> None:
        valid_payload["comparison"]["c_99"] = 1.0
        with pytest.raises(InvalidFeaturePayloadError, match="unexpected key"):
            FeatureTransformer().from_payload(**valid_payload)

    def test_leakage_key_is_rejected(self, valid_payload: dict[str, dict[str, float]]) -> None:
        valid_payload["group_1"]["g1_21"] = 1.0
        with pytest.raises(InvalidFeaturePayloadError, match="post-campaign"):
            FeatureTransformer().from_payload(**valid_payload)

    def test_none_becomes_nan_for_the_imputer(
        self, valid_payload: dict[str, dict[str, float]]
    ) -> None:
        valid_payload["group_2"]["g2_3"] = None
        frame = FeatureTransformer().from_payload(**valid_payload)
        assert np.isnan(frame["g2_3"].iloc[0])

    def test_non_numeric_value_is_rejected(
        self, valid_payload: dict[str, dict[str, float]]
    ) -> None:
        valid_payload["group_1"]["g1_1"] = "not-a-number"
        with pytest.raises(InvalidFeaturePayloadError, match="not numeric"):
            FeatureTransformer().from_payload(**valid_payload)

    def test_from_frame_accepts_a_valid_upload(self, feature_frame: pd.DataFrame) -> None:
        adapted = FeatureTransformer().from_frame(feature_frame)
        assert list(adapted.columns) == BASE_FEATURES

    def test_from_frame_rejects_incomplete_upload(self, feature_frame: pd.DataFrame) -> None:
        with pytest.raises(InvalidFeaturePayloadError, match="missing"):
            FeatureTransformer().from_frame(feature_frame.drop(columns=["c_1"]))

    def test_feature_blocks_have_the_documented_sizes(self) -> None:
        assert len(GROUP_1_FEATURES) == 20
        assert len(GROUP_2_FEATURES) == 20
        assert len(COMPARISON_FEATURES) == 27
        assert len(BASE_FEATURES) == 67
