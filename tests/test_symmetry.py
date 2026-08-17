"""Unit tests for the group-swap symmetry module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import BASE_FEATURES, TARGET_COLUMN
from src.symmetry import (
    SWAPPED_TARGET_MAP,
    augment_with_swapped,
    augment_with_swapped_grouped,
    diagnose_comparison_symmetry,
    groups_are_exchangeable,
    measure_invariance,
    suggested_negate_columns,
    swap_groups,
    swap_is_valid,
    swap_targets,
    validate_group_exchangeability,
    validate_swap,
)


class TestSwapTargets:
    def test_maps_classes_correctly(self) -> None:
        swapped = swap_targets(pd.Series([0, 1, 2]))
        assert list(swapped) == [0, 2, 1]

    def test_is_an_involution(self) -> None:
        original = pd.Series([0, 1, 2, 1, 2, 0])
        assert list(swap_targets(swap_targets(original))) == list(original)

    def test_preserves_index_and_name(self) -> None:
        original = pd.Series([1, 2], index=[10, 20], name=TARGET_COLUMN)
        swapped = swap_targets(original)
        assert list(swapped.index) == [10, 20]
        assert swapped.name == TARGET_COLUMN

    def test_rejects_unknown_class(self) -> None:
        with pytest.raises(ValueError, match="unmappable"):
            swap_targets(pd.Series([0, 1, 7]))

    def test_map_is_complete(self) -> None:
        assert SWAPPED_TARGET_MAP == {0: 0, 1: 2, 2: 1}


class TestSwapGroups:
    def test_exchanges_the_two_group_blocks(self, feature_frame: pd.DataFrame) -> None:
        swapped = swap_groups(feature_frame)
        pd.testing.assert_series_equal(swapped["g1_1"], feature_frame["g2_1"], check_names=False)
        pd.testing.assert_series_equal(swapped["g2_1"], feature_frame["g1_1"], check_names=False)

    def test_preserves_column_order(self, feature_frame: pd.DataFrame) -> None:
        assert list(swap_groups(feature_frame).columns) == list(feature_frame.columns)

    def test_is_an_involution(self, feature_frame: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(swap_groups(swap_groups(feature_frame)), feature_frame)

    def test_leaves_comparison_features_untouched_by_default(
        self, feature_frame: pd.DataFrame
    ) -> None:
        pd.testing.assert_series_equal(swap_groups(feature_frame)["c_1"], feature_frame["c_1"])

    def test_negates_requested_comparison_features(self, feature_frame: pd.DataFrame) -> None:
        swapped = swap_groups(feature_frame, negate_columns=["c_1", "c_2"])
        np.testing.assert_allclose(swapped["c_1"], -feature_frame["c_1"])
        np.testing.assert_allclose(swapped["c_2"], -feature_frame["c_2"])
        pd.testing.assert_series_equal(swapped["c_3"], feature_frame["c_3"])

    def test_negation_is_also_an_involution(self, feature_frame: pd.DataFrame) -> None:
        twice = swap_groups(swap_groups(feature_frame, ["c_1"]), ["c_1"])
        pd.testing.assert_frame_equal(twice, feature_frame)

    def test_unknown_negate_column_raises(self, feature_frame: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="not present"):
            swap_groups(feature_frame, negate_columns=["does_not_exist"])


class TestAugmentWithSwapped:
    def test_doubles_the_row_count(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        features, target = augment_with_swapped(feature_frame, raw_dataset[TARGET_COLUMN])
        assert len(features) == 2 * len(feature_frame)
        assert len(target) == 2 * len(feature_frame)

    def test_class_1_and_2_counts_are_balanced_after_augmentation(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        _, target = augment_with_swapped(feature_frame, raw_dataset[TARGET_COLUMN])
        counts = target.value_counts()
        assert counts.get(1, 0) == counts.get(2, 0)

    def test_class_0_count_doubles_exactly(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        original = raw_dataset[TARGET_COLUMN]
        _, target = augment_with_swapped(feature_frame, original)
        assert (target == 0).sum() == 2 * (original == 0).sum()

    def test_mirrored_half_matches_the_swap(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        features, _ = augment_with_swapped(feature_frame, raw_dataset[TARGET_COLUMN])
        half = len(feature_frame)
        expected = swap_groups(feature_frame).reset_index(drop=True)
        pd.testing.assert_frame_equal(features.iloc[half:].reset_index(drop=True), expected)

    def test_columns_are_unchanged(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        features, _ = augment_with_swapped(feature_frame, raw_dataset[TARGET_COLUMN])
        assert list(features.columns) == BASE_FEATURES

    def test_length_mismatch_raises(self, feature_frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="same length"):
            augment_with_swapped(feature_frame, pd.Series([0, 1, 2]))


class TestAugmentationGroups:
    """A campaign and its mirror must never straddle a cross-validation fold."""

    def test_groups_pair_each_campaign_with_its_mirror(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        augmented = augment_with_swapped_grouped(feature_frame, raw_dataset[TARGET_COLUMN])
        half = len(feature_frame)
        np.testing.assert_array_equal(augmented.groups[:half], augmented.groups[half:])

    def test_every_group_appears_exactly_twice(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        augmented = augment_with_swapped_grouped(feature_frame, raw_dataset[TARGET_COLUMN])
        _, counts = np.unique(augmented.groups, return_counts=True)
        assert set(counts) == {2}

    def test_group_count_equals_original_row_count(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        augmented = augment_with_swapped_grouped(feature_frame, raw_dataset[TARGET_COLUMN])
        assert len(np.unique(augmented.groups)) == len(feature_frame)

    def test_grouped_cv_never_splits_a_campaign(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        from sklearn.model_selection import StratifiedGroupKFold

        augmented = augment_with_swapped_grouped(feature_frame, raw_dataset[TARGET_COLUMN])
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)

        for train_idx, test_idx in splitter.split(
            augmented.features, augmented.target, groups=augmented.groups
        ):
            overlap = set(augmented.groups[train_idx]) & set(augmented.groups[test_idx])
            assert not overlap, "a campaign and its mirror ended up in different folds"

    def test_plain_stratified_cv_would_have_leaked(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        # Documents the bug this fix prevents: without groups, mirrors do straddle folds.
        from sklearn.model_selection import StratifiedKFold

        augmented = augment_with_swapped_grouped(feature_frame, raw_dataset[TARGET_COLUMN])
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

        leaked = False
        for train_idx, test_idx in splitter.split(augmented.features, augmented.target):
            if set(augmented.groups[train_idx]) & set(augmented.groups[test_idx]):
                leaked = True
                break
        assert leaked, "expected plain StratifiedKFold to leak mirrored rows across folds"

    def test_wrapper_matches_the_grouped_version(
        self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame
    ) -> None:
        target = raw_dataset[TARGET_COLUMN]
        features, labels = augment_with_swapped(feature_frame, target)
        augmented = augment_with_swapped_grouped(feature_frame, target)
        pd.testing.assert_frame_equal(features, augmented.features)
        assert list(labels) == list(augmented.target)

    def test_dataset_length(self, feature_frame: pd.DataFrame, raw_dataset: pd.DataFrame) -> None:
        augmented = augment_with_swapped_grouped(feature_frame, raw_dataset[TARGET_COLUMN])
        assert len(augmented) == 2 * len(feature_frame)


class TestGroupExchangeability:
    """Symmetry augmentation is only valid when the two slots are interchangeable."""

    def test_identically_distributed_blocks_are_exchangeable(
        self, feature_frame: pd.DataFrame
    ) -> None:
        report = validate_group_exchangeability(feature_frame)
        assert report["exchangeable"].all()
        assert groups_are_exchangeable(feature_frame) is True

    def test_a_shifted_block_is_detected(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["g1_5"] = frame["g1_5"] + 5.0  # group 1 systematically larger

        report = validate_group_exchangeability(frame)
        assert not report.loc["var_5", "exchangeable"]
        assert groups_are_exchangeable(frame) is False

    def test_a_variance_difference_is_detected(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["g2_7"] = frame["g2_7"] * 6.0  # same centre, very different spread

        report = validate_group_exchangeability(frame)
        assert not report.loc["var_7", "exchangeable"]
        assert report.loc["var_7", "std_ratio"] < 0.5

    def test_report_covers_every_paired_variable(self, feature_frame: pd.DataFrame) -> None:
        assert len(validate_group_exchangeability(feature_frame)) == 20

    def test_report_is_sorted_most_divergent_first(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["g1_11"] = frame["g1_11"] + 8.0
        assert validate_group_exchangeability(frame).index[0] == "var_11"

    def test_report_records_the_mean_gap(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["g1_3"] = frame["g1_3"] + 4.0
        assert validate_group_exchangeability(frame).loc["var_3", "mean_gap"] > 3.0


class TestValidateSwap:
    def test_symmetric_data_passes(self, feature_frame: pd.DataFrame) -> None:
        report = validate_swap(feature_frame)
        assert report["distribution_preserved"].all()
        assert swap_is_valid(feature_frame) is True

    def test_negating_a_signed_difference_is_fine(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_1"] = frame["g1_3"] - frame["g2_3"]  # symmetric about zero
        report = validate_swap(frame, negate_columns=["c_1"])
        assert report.loc["c_1", "distribution_preserved"]
        assert not report.loc["c_1", "strictly_positive"]

    def test_negating_a_ratio_is_caught(self, feature_frame: pd.DataFrame) -> None:
        # A ratio must be INVERTED when the groups swap, not negated. Sign flipping
        # produces strictly negative values that could never occur.
        frame = feature_frame.copy()
        frame["c_2"] = np.abs(frame["g1_4"]) / (np.abs(frame["g2_4"]) + 1e-6)

        report = validate_swap(frame, negate_columns=["c_2"])
        assert report.loc["c_2", "strictly_positive"]
        assert not report.loc["c_2", "distribution_preserved"]
        assert swap_is_valid(frame, negate_columns=["c_2"]) is False

    def test_leaving_a_ratio_alone_passes(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_2"] = np.abs(frame["g1_4"]) / (np.abs(frame["g2_4"]) + 1e-6)
        assert validate_swap(frame, negate_columns=[]).loc["c_2", "distribution_preserved"]

    def test_report_covers_every_comparison_column(self, feature_frame: pd.DataFrame) -> None:
        assert len(validate_swap(feature_frame)) == 27

    def test_report_is_sorted_worst_first(self, feature_frame: pd.DataFrame) -> None:
        values = validate_swap(feature_frame)["p_value"].to_numpy()
        assert (np.diff(values) >= -1e-9).all()

    def test_report_records_which_columns_were_negated(self, feature_frame: pd.DataFrame) -> None:
        report = validate_swap(feature_frame, negate_columns=["c_5"])
        assert report.loc["c_5", "negated"]
        assert not report.loc["c_6", "negated"]


class TestMeasureInvariance:
    def test_perfectly_symmetric_model_has_zero_violations(
        self, feature_frame: pd.DataFrame
    ) -> None:
        def predict(frame: pd.DataFrame) -> np.ndarray:
            # Decides purely on the sign of g1_1 - g2_1, which is exactly antisymmetric.
            difference = frame["g1_1"] - frame["g2_1"]
            return np.where(difference > 0, 1, 2)

        report = measure_invariance(predict, feature_frame)
        assert report.violation_rate == 0.0
        assert report.n_violations == 0

    def test_constant_model_is_maximally_biased(self, feature_frame: pd.DataFrame) -> None:
        def predict(frame: pd.DataFrame) -> np.ndarray:
            return np.ones(len(frame), dtype=int)

        report = measure_invariance(predict, feature_frame)
        assert report.violation_rate == 1.0
        assert report.position_bias == 1.0

    def test_class_zero_model_is_invariant(self, feature_frame: pd.DataFrame) -> None:
        def predict(frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame), dtype=int)

        report = measure_invariance(predict, feature_frame)
        assert report.violation_rate == 0.0
        assert report.class_0_flip_rate == 1.0

    def test_summary_flags_position_bias(self, feature_frame: pd.DataFrame) -> None:
        report = measure_invariance(lambda f: np.ones(len(f), dtype=int), feature_frame)
        assert "POSITION-BIASED" in report.summary()

    def test_report_is_serialisable(self, feature_frame: pd.DataFrame) -> None:
        report = measure_invariance(lambda f: np.zeros(len(f), dtype=int), feature_frame)
        assert set(report.to_dict()) >= {"violation_rate", "position_bias", "n_rows"}

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            measure_invariance(lambda f: np.array([]), pd.DataFrame(columns=BASE_FEATURES))

    def test_real_pipeline_is_measurable(
        self, trained_artifact, feature_frame: pd.DataFrame
    ) -> None:
        import joblib

        pipeline = joblib.load(trained_artifact)["pipeline"]
        report = measure_invariance(pipeline.predict, feature_frame.head(50))
        assert 0.0 <= report.violation_rate <= 1.0


class TestDiagnoseComparisonSymmetry:
    def test_detects_a_planted_difference_feature(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_5"] = frame["g1_3"] - frame["g2_3"]  # exactly direction dependent

        report = diagnose_comparison_symmetry(frame)
        assert report.loc["c_5", "likely_direction_dependent"]
        assert report.loc["c_5", "best_match"] == "diff_3"
        assert abs(report.loc["c_5", "correlation"]) == pytest.approx(1.0, abs=1e-6)

    def test_independent_feature_is_not_flagged(self, feature_frame: pd.DataFrame) -> None:
        report = diagnose_comparison_symmetry(feature_frame)
        # Synthetic c_ columns are independent noise, so none should be flagged.
        assert not report["likely_direction_dependent"].any()

    def test_report_covers_every_comparison_feature(self, feature_frame: pd.DataFrame) -> None:
        assert len(diagnose_comparison_symmetry(feature_frame)) == 27

    def test_report_is_sorted_by_evidence(self, feature_frame: pd.DataFrame) -> None:
        values = diagnose_comparison_symmetry(feature_frame)["abs_correlation"].to_numpy()
        assert (np.diff(values) <= 1e-9).all()

    def test_suggested_columns_match_the_flagged_rows(self, feature_frame: pd.DataFrame) -> None:
        frame = feature_frame.copy()
        frame["c_7"] = frame["g1_2"] - frame["g2_2"]
        assert suggested_negate_columns(frame) == ["c_7"]
