"""Unit tests for the selective-prediction and abstention diagnostics.

These functions decide *what the system does with a prediction*, not what the prediction
is - a confidence gate that automates part of the workload, and a decline rule that
recommends running no campaign at all. Both fit a threshold, so both are capable of the
same contamination as fitting a model: choose the threshold by looking at the test set and
the reported figure is optimistic.

The tests below therefore cover two things the type system cannot:

1. **Selection honesty** - a chosen threshold must actually beat the baseline it is
   measured against, and the function must be able to answer "no rule helps".
2. **The notebook's contract** - the analysis reads exactly two keys from the returned
   dictionary and applies the result to the test set once. Both keys must exist in every
   branch, or the analysis raises `KeyError` halfway through a long run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.diagnostics import (
    abstention_rule_comparison,
    apply_decline_rule,
    operating_points,
    select_confidence_threshold,
    select_decline_threshold,
)


def _stack(blocks: list[tuple[int, int, list[float]]]) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(y_true, probabilities)`` from ``(true_class, n_rows, [p0, p1, p2])``."""
    truth = np.concatenate([np.full(count, label) for label, count, _ in blocks])
    proba = np.vstack([np.tile(row, (count, 1)) for _, count, row in blocks])
    return truth, proba


# ---------------------------------------------------------------------------------------
# The regression test for a bug that reached findings.json.
# ---------------------------------------------------------------------------------------


class TestDeclineThresholdMustBeatArgmax:
    """A decline rule that is worse than doing nothing must not be selected.

    The failure this guards against is subtle and did occur. ``select_decline_threshold``
    ranked candidate thresholds against *each other* and returned the best one - without
    ever checking it beat plain ``argmax``. Because the rule routes every ``P(0) < tau``
    row to the better of classes 1 and 2, it *removes* the class-0 predictions argmax makes
    when ``P(0)`` is the maximum but still below ``tau``. The "best" candidate can
    therefore have lower class-0 recall than the baseline it is being compared to, and the
    search will still return it.
    """

    # Every candidate must sit above P(0) on the class-0 rows. With three classes a value
    # can only be the argmax if it exceeds 1/3, so a grid starting below 0.34 would let
    # some candidates match argmax's recall instead of falling short of it - and the test
    # would then pass for the wrong reason.
    THRESHOLDS = (0.40, 0.45, 0.50)
    BUDGET = 0.05

    @staticmethod
    def _argmax_already_wins() -> tuple[np.ndarray, np.ndarray]:
        """P(0) is the argmax on the class-0 rows, but below every candidate threshold.

        No candidate can fire on those rows, so every one has class-0 recall of exactly
        zero while argmax has perfect recall. The class-0 population is deliberately tiny,
        so missing it costs only two points of accuracy - which is what makes the
        candidates look affordable and springs the trap.
        """
        return _stack(
            [
                (0, 2, [0.36, 0.34, 0.30]),  # argmax -> 0 (correct); below every threshold
                (1, 49, [0.10, 0.60, 0.30]),
                (2, 49, [0.10, 0.30, 0.60]),
            ]
        )

    def test_returns_none_when_no_threshold_improves_class_0_recall(self) -> None:
        truth, proba = self._argmax_already_wins()

        result = select_decline_threshold(
            truth,
            proba,
            max_accuracy_loss=self.BUDGET,
            decline_thresholds=self.THRESHOLDS,
        )

        assert result["threshold"] is None, (
            "a threshold was selected whose class-0 recall is no better than argmax; "
            "the search is ranking candidates against each other instead of the baseline"
        )
        assert result["argmax_class_0_recall"] == pytest.approx(1.0)

    def test_the_rejected_candidates_really_were_affordable(self) -> None:
        """Proves the previous test exercises the recall condition, not the accuracy one.

        Without this, a change that tightened the accuracy budget would make the test above
        pass for the wrong reason and stop guarding anything.
        """
        truth, proba = self._argmax_already_wins()
        table = abstention_rule_comparison(truth, proba, decline_thresholds=self.THRESHOLDS)

        floor = float(table.loc["argmax", "accuracy"]) - self.BUDGET
        candidates = table.drop(index="argmax")

        assert (candidates["accuracy"] >= floor).all(), "the accuracy floor already excluded them"
        assert (candidates["class_0_recall"] < table.loc["argmax", "class_0_recall"]).all()

    def test_selects_a_threshold_when_one_genuinely_helps(self) -> None:
        """The mirror image: argmax hides a class-0 belief that a threshold recovers."""
        truth, proba = _stack(
            [
                (0, 20, [0.45, 0.50, 0.05]),  # argmax -> 1, so class 0 is never predicted
                (1, 40, [0.10, 0.60, 0.30]),
                (2, 40, [0.10, 0.30, 0.60]),
            ]
        )

        result = select_decline_threshold(truth, proba, max_accuracy_loss=0.01)

        assert result["threshold"] is not None
        assert result["argmax_class_0_recall"] == pytest.approx(0.0)
        assert result["validation_class_0_recall"] > result["argmax_class_0_recall"]


class TestSelectionResultContract:
    """The notebook reads two keys and applies the result once. Both branches must supply
    them, or a multi-minute analysis dies on a ``KeyError`` at the display step."""

    NO_RULE_HELPS = [
        (0, 2, [0.36, 0.34, 0.30]),
        (1, 49, [0.10, 0.60, 0.30]),
        (2, 49, [0.10, 0.30, 0.60]),
    ]
    A_RULE_HELPS = [
        (0, 20, [0.45, 0.50, 0.05]),
        (1, 40, [0.10, 0.60, 0.30]),
        (2, 40, [0.10, 0.30, 0.60]),
    ]

    @pytest.mark.parametrize(
        "blocks",
        [
            pytest.param(NO_RULE_HELPS, id="no-threshold-helps"),
            pytest.param(A_RULE_HELPS, id="a-threshold-helps"),
        ],
    )
    def test_threshold_and_rule_are_always_present(self, blocks) -> None:
        truth, proba = _stack(blocks)
        result = select_decline_threshold(truth, proba)

        assert "threshold" in result
        assert isinstance(result["rule"], str) and result["rule"]

    def test_a_none_threshold_is_applicable_without_a_special_case(self) -> None:
        """``apply_decline_rule`` must accept the null result, so the caller never branches.

        If the caller had to write ``if threshold is not None``, the null case would
        eventually be handled differently - or forgotten - in one of the places it is used.
        """
        _, proba = _stack([(0, 10, [0.60, 0.20, 0.20]), (1, 10, [0.10, 0.70, 0.20])])

        assert np.array_equal(
            apply_decline_rule(proba, None),
            np.asarray([0, 1, 2])[proba.argmax(axis=1)],
        )


class TestApplyDeclineRule:
    def test_declines_exactly_when_the_threshold_is_cleared(self) -> None:
        proba = np.array(
            [
                [0.50, 0.30, 0.20],  # >= 0.40 -> decline
                [0.40, 0.35, 0.25],  # == 0.40 -> decline (boundary is inclusive)
                [0.39, 0.36, 0.25],  # <  0.40 -> better group, which is 1
                [0.10, 0.30, 0.60],  # <  0.40 -> better group, which is 2
            ]
        )
        assert np.array_equal(apply_decline_rule(proba, 0.40), np.array([0, 0, 1, 2]))

    def test_never_returns_class_0_below_the_threshold(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.random((200, 3))
        proba = raw / raw.sum(axis=1, keepdims=True)

        predictions = apply_decline_rule(proba, 0.45)

        assert (proba[predictions == 0, 0] >= 0.45).all()


class TestAbstentionRuleComparison:
    def test_argmax_is_the_first_row_so_every_rule_has_a_baseline(self) -> None:
        truth, proba = _stack([(0, 10, [0.50, 0.30, 0.20]), (1, 10, [0.10, 0.70, 0.20])])
        table = abstention_rule_comparison(truth, proba)

        assert table.index[0] == "argmax"
        assert {"n_declined", "class_0_recall", "precision_of_decline", "accuracy"} <= set(
            table.columns
        )

    def test_fewer_campaigns_are_declined_as_the_threshold_rises(self) -> None:
        """A lower threshold declines a superset of what a higher one declines."""
        rng = np.random.default_rng(1)
        raw = rng.random((300, 3))
        proba = raw / raw.sum(axis=1, keepdims=True)
        truth = rng.integers(0, 3, size=300)

        table = abstention_rule_comparison(truth, proba).drop(index="argmax")
        counts = table["n_declined"].to_numpy()

        assert (np.diff(counts) <= 0).all(), "n_declined should fall as the threshold rises"


class TestConfidenceGate:
    def test_respects_the_minimum_coverage_constraint(self) -> None:
        rng = np.random.default_rng(2)
        raw = rng.random((400, 3)) ** 3  # skewed, so confidences span a wide range
        proba = raw / raw.sum(axis=1, keepdims=True)
        truth = proba.argmax(axis=1)  # a strong model, so accuracy rises with confidence

        gate = select_confidence_threshold(truth, proba, min_coverage=0.30)

        assert gate["coverage"] >= 0.30
        assert "coverage" in gate["rule"]

    def test_coverage_falls_as_the_confidence_bar_rises(self) -> None:
        rng = np.random.default_rng(3)
        raw = rng.random((400, 3))
        proba = raw / raw.sum(axis=1, keepdims=True)
        truth = rng.integers(0, 3, size=400)

        table = operating_points(truth, proba, thresholds=(0.0, 0.4, 0.5, 0.6, 0.7))

        assert (np.diff(table["coverage"].to_numpy()) <= 0).all()
        assert table["coverage"].iloc[0] == pytest.approx(1.0)

    def test_full_coverage_accuracy_is_reported_at_every_operating_point(self) -> None:
        """The headline number must travel with the operating point.

        Quoting "76% accurate" without "on 31% of campaigns" is the single easiest way to
        oversell a selective-prediction result, so `accuracy_overall` is carried on every
        row rather than left for the reader to look up.
        """
        rng = np.random.default_rng(4)
        raw = rng.random((200, 3))
        proba = raw / raw.sum(axis=1, keepdims=True)
        truth = rng.integers(0, 3, size=200)

        table = operating_points(truth, proba, thresholds=(0.0, 0.5, 0.6))

        assert table["accuracy_overall"].nunique() == 1
