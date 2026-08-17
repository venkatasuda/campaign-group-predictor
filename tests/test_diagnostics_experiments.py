"""Smoke tests for the cross-validated diagnostic experiments.

These five functions each run a full cross-validation internally, so they are the most
expensive things in the codebase and the easiest to leave untested. That is exactly why
they are worth covering: they produce the numbers that drive the report's conclusions -
how many features are needed, whether more data would help, whether ensembling is worth
it, and what the honest generalisation estimate is.

The tests here check **contract, not accuracy**. A nine-feature synthetic frame will not
reproduce the findings from the real dataset, and pinning a score would be a change
detector rather than a test. What must hold regardless of the data is that each function
returns the columns the notebook indexes into, in a usable shape, without raising - because
a `KeyError` here surfaces forty minutes into a notebook run, after the expensive part.

Everything is deliberately tiny: two folds, two candidates, a linear model. The point is to
exercise the code path, not to measure anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.training.diagnostics import (
    ensemble_experiment,
    feature_block_ablation,
    feature_count_sweep,
    learning_curve_report,
)
from src.training.pipeline import build_pipeline
from src.training.tuning import nested_cv_score

N_ROWS = 120
PAIRS = 3


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, pd.Series]:
    """A small three-class frame using the real column naming convention.

    The `g1_`/`g2_`/`c_` prefixes matter: `feature_block_ablation` selects blocks by name,
    and `build_pipeline` derives pairwise differences from matching `g1_i`/`g2_i` pairs. A
    frame with arbitrary column names would silently exercise a different code path.
    """
    rng = np.random.default_rng(0)
    columns: dict[str, np.ndarray] = {}

    for index in range(1, PAIRS + 1):
        columns[f"g1_{index}"] = rng.normal(size=N_ROWS)
        columns[f"g2_{index}"] = rng.normal(size=N_ROWS)
        columns[f"c_{index}"] = rng.normal(size=N_ROWS)

    features = pd.DataFrame(columns)
    # A learnable but weak signal, so the models neither fail to converge nor score 1.0.
    logits = features["c_1"] + 0.5 * (features["g1_1"] - features["g2_1"])
    target = pd.Series(pd.qcut(logits, q=3, labels=[0, 1, 2]).astype(int), name="target")
    return features, target


@pytest.fixture(scope="module")
def cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=2, shuffle=True, random_state=0)


def _fast_pipeline():
    return build_pipeline(LogisticRegression(max_iter=200, random_state=0))


class TestFeatureCountSweep:
    def test_returns_one_row_per_requested_count(self, synthetic, cv) -> None:
        features, target = synthetic
        ranked = ["c_1", "g1_1", "g2_1", "c_2"]

        report = feature_count_sweep(
            _fast_pipeline, features, target, ranked_features=ranked, cv=cv, counts=(1, 2, 4)
        )

        assert list(report.index) == [1, 2, 4]
        assert {"cv_mean", "cv_std", "pct_of_best"} <= set(report.columns)

    def test_pct_of_best_peaks_at_one_hundred(self, synthetic, cv) -> None:
        """The normalisation must be against the best row, so exactly one reaches 100."""
        features, target = synthetic
        report = feature_count_sweep(
            _fast_pipeline, features, target, ranked_features=["c_1", "g1_1"], cv=cv, counts=(1, 2)
        )
        assert report["pct_of_best"].max() == pytest.approx(100.0)

    def test_ignores_requested_features_that_do_not_exist(self, synthetic, cv) -> None:
        """A stale ranking must not crash the sweep forty minutes into a notebook run."""
        features, target = synthetic
        report = feature_count_sweep(
            _fast_pipeline,
            features,
            target,
            ranked_features=["c_1", "does_not_exist", "g1_1"],
            cv=cv,
            counts=(1, 3),
        )
        assert not report.empty
        assert report.index.max() <= len(features.columns)


class TestFeatureBlockAblation:
    def test_reports_each_block_that_has_columns_present(self, synthetic, cv) -> None:
        features, target = synthetic
        report = feature_block_ablation(_fast_pipeline, features, target, cv)

        assert "all features" in report.index
        assert "comparison block only (c_)" in report.index
        assert {"n_features", "cv_mean", "cv_std", "delta_vs_all"} <= set(report.columns)

    def test_all_features_is_the_zero_point(self, synthetic, cv) -> None:
        """`delta_vs_all` is measured against the full set, so that row must be 0."""
        features, target = synthetic
        report = feature_block_ablation(_fast_pipeline, features, target, cv)
        assert report.loc["all features", "delta_vs_all"] == pytest.approx(0.0, abs=1e-9)

    def test_extra_blocks_are_evaluated(self, synthetic, cv) -> None:
        features, target = synthetic
        report = feature_block_ablation(
            _fast_pipeline, features, target, cv, extra_blocks={"c_1 alone": ["c_1"]}
        )
        assert report.loc["c_1 alone", "n_features"] == 1


class TestLearningCurve:
    def test_scores_rise_through_the_requested_fractions(self, synthetic, cv) -> None:
        features, target = synthetic
        curve = learning_curve_report(_fast_pipeline(), features, target, cv, fractions=(0.5, 1.0))

        assert len(curve) == 2
        assert {"n_train", "cv_mean", "cv_std", "gain_vs_previous"} <= set(curve.columns)
        assert curve["n_train"].is_monotonic_increasing

    def test_first_row_has_no_previous_gain(self, synthetic, cv) -> None:
        """The first point has nothing to compare against; it must not invent a gain."""
        features, target = synthetic
        curve = learning_curve_report(_fast_pipeline(), features, target, cv, fractions=(0.5, 1.0))
        first = curve["gain_vs_previous"].iloc[0]
        assert pd.isna(first) or first == pytest.approx(0.0)


class TestEnsembleExperiment:
    def test_compares_combiners_against_their_own_bases(self, synthetic, cv) -> None:
        features, target = synthetic
        bases = {
            "logistic": build_pipeline(LogisticRegression(max_iter=200, random_state=0)),
            "tree": build_pipeline(DecisionTreeClassifier(max_depth=3, random_state=0)),
        }

        report = ensemble_experiment(bases, features, target, cv=cv, stacking_cv=2)

        approaches = set(report.index)
        assert "soft voting" in approaches
        assert any("stacking" in name for name in approaches)
        assert any(name.startswith("base:") for name in approaches)
        assert {"cv_mean", "cv_std", "delta_vs_best_base", "verdict"} <= set(report.columns)

    def test_verdict_is_one_of_the_three_stated_outcomes(self, synthetic, cv) -> None:
        """A gain inside the base model's own fold-to-fold spread is not a gain."""
        features, target = synthetic
        bases = {"logistic": build_pipeline(LogisticRegression(max_iter=200, random_state=0))}

        report = ensemble_experiment(bases, features, target, cv=cv, stacking_cv=2)

        assert set(report["verdict"]) <= {
            "improves on best base",
            "within noise",
            "worse than best base",
        }


class TestNestedCrossValidation:
    def test_returns_one_score_per_outer_fold(self, synthetic) -> None:
        features, target = synthetic

        result = nested_cv_score(
            build_pipeline(LogisticRegression(max_iter=200, random_state=0)),
            model_name="logistic_regression",
            features=features,
            target=target,
            outer_splits=2,
            inner_splits=2,
            n_iter=2,
            strategy="random",
            n_jobs=1,
        )

        assert len(result["outer_scores"]) == 2
        assert result["nested_mean"] == pytest.approx(
            float(np.mean(result["outer_scores"])), abs=1e-3
        )
        assert result["strategy"] == "random"

    def test_refuses_a_model_with_no_search_space(self, synthetic) -> None:
        """Silently skipping the search would report a nested score for an untuned model."""
        features, target = synthetic

        with pytest.raises(ValueError, match="no search space"):
            nested_cv_score(
                build_pipeline(LogisticRegression(max_iter=200, random_state=0)),
                model_name="not_a_real_model",
                features=features,
                target=target,
                outer_splits=2,
                inner_splits=2,
                n_iter=2,
            )
