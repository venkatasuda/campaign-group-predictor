"""Decision-support diagnostics.

Three questions a stakeholder asks once the headline accuracy is on the table. Each is
answered with evidence rather than opinion, and each changes what you would do next.

``learning_curve_report``
    *"Would collecting more campaigns fix this?"* Train on increasing fractions of the
    data and watch the validation score. A curve still rising at 100% means more data
    helps; a flat curve means the ceiling is the features, and gathering more rows is
    wasted budget.

``operating_points``
    *"When can we trust the model, and when should a person decide?"* Sweep a confidence
    threshold and report coverage against accuracy. This converts a single accuracy number
    into a deployable policy: automate above the threshold, escalate below it.

``feature_block_ablation``
    *"Which of these 67 anonymised columns actually matter?"* Retrain on subsets of the
    feature blocks. Answers whether the comparison block earns its place, and how much of
    the signal a handful of features carries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, cross_val_score

from src.constants import COMPARISON_FEATURES, GROUP_1_FEATURES, GROUP_2_FEATURES
from src.logging_config import get_logger

logger = get_logger(__name__)


def learning_curve_report(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    cv: BaseCrossValidator,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    scoring: str = "accuracy",
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure how validation score responds to training-set size.

    Interpretation
    --------------
    * **Still rising at 100%** - the model is data-limited. Collecting more campaigns is
      the highest-value investment.
    * **Flat over the last two points** - the model is *feature*-limited. More rows will
      not help; better features or a better-defined target will.

    This distinction decides where the next budget goes, and it is answerable in minutes.

    Returns
    -------
    pandas.DataFrame
        One row per fraction with ``n_train``, ``cv_mean``, ``cv_std`` and ``gain_vs_previous``.
    """
    from sklearn.model_selection import train_test_split

    rows: list[dict[str, Any]] = []
    previous: float | None = None

    # Minimum rows for the cross-validator to be able to form its folds with every class
    # represented. Below this the score is meaningless rather than merely noisy.
    min_rows = cv.get_n_splits() * int(target.nunique()) * 2

    for fraction in fractions:
        if fraction >= 1.0:
            subset_x, subset_y = features, target
        else:
            # Stratified subsample so class balance is identical at every size - the curve
            # must reflect sample size alone, not a drifting class mix.
            #
            # train_test_split with stratify= is used rather than a manual groupby sample:
            # it is the idiomatic, obviously-correct construction, and it guarantees the
            # returned subset is genuinely stratified rather than merely assembled from
            # per-class samples that a later truncation could unbalance.
            subset_x, _, subset_y, _ = train_test_split(
                features,
                target,
                train_size=fraction,
                stratify=target,
                random_state=random_state,
            )

        if len(subset_x) < min_rows or subset_y.nunique() < 2:
            logger.info(
                "Skipping fraction %.2f: %d rows is too few for %d-fold CV over %d classes.",
                fraction,
                len(subset_x),
                cv.get_n_splits(),
                target.nunique(),
            )
            continue

        scores = cross_val_score(clone(pipeline), subset_x, subset_y, cv=cv, scoring=scoring)
        mean = float(scores.mean())

        rows.append(
            {
                "fraction": round(float(fraction), 3),
                "n_train": int(len(subset_x)),
                "cv_mean": round(mean, 4),
                "cv_std": round(float(scores.std()), 4),
                "gain_vs_previous": None if previous is None else round(mean - previous, 4),
            }
        )
        previous = mean

    report = pd.DataFrame(rows).set_index("fraction")
    if len(report) >= 2:
        final_gain = report["gain_vs_previous"].iloc[-1]
        logger.info(
            "Learning curve: last doubling of data changed the score by %+.4f - %s.",
            final_gain,
            (
                "more data would still help"
                if final_gain and final_gain > 0.01
                else "the model is feature-limited, not data-limited"
            ),
        )
    return report


def operating_points(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
    thresholds: Sequence[float] = (0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Trade coverage against accuracy by requiring a minimum confidence.

    A single accuracy figure assumes the model must answer every case. In practice it does
    not have to: campaigns it is unsure about can be routed to a person. This sweep shows
    what that buys.

    Returns
    -------
    pandas.DataFrame
        Per threshold: ``coverage`` (share of campaigns the model would decide),
        ``accuracy_covered`` (accuracy on those), ``n_escalated``, and
        ``accuracy_overall``.

        ``accuracy_overall`` is the model's accuracy at **full coverage** and is therefore
        identical on every row. It is repeated deliberately rather than reported once: it
        is the number ``accuracy_covered`` must always be quoted against. "76% accurate"
        is a different claim from "76% accurate on the 31% of campaigns it chose to
        answer", and carrying the comparator on the same row makes the second one the
        easier sentence to write.

        It makes **no** assumption about how escalated campaigns are resolved - it is not
        a blended human-plus-model figure. Estimating that would require knowing how
        accurate the human reviewer is, which this dataset cannot tell us.

    Reading it
    ----------
    If accuracy on the covered subset climbs steeply as coverage falls, the predicted
    probabilities carry a usable ordering of reliability, and a confidence gate is worth
    deploying. If accuracy stays flat while coverage drops, they do not, and gating only
    loses volume.

    Note this describes the *ordering* of the probabilities, not their calibration - a
    model can rank its own reliability well while being systematically overconfident about
    the absolute numbers. See :mod:`src.calibration` for that question.
    """
    truth = np.asarray(y_true).ravel().astype(int)
    proba = np.asarray(probabilities, dtype=float)
    class_order = np.asarray(classes).astype(int)

    confidence = proba.max(axis=1)
    predicted = class_order[proba.argmax(axis=1)]
    correct = predicted == truth

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        covered = confidence >= threshold
        n_covered = int(covered.sum())

        rows.append(
            {
                "min_confidence": round(float(threshold), 3),
                "coverage": round(n_covered / len(truth), 4),
                "n_decided": n_covered,
                "n_escalated": int(len(truth) - n_covered),
                "accuracy_covered": (
                    round(float(correct[covered].mean()), 4) if n_covered else float("nan")
                ),
                "accuracy_overall": round(float(correct.mean()), 4),
            }
        )

    return pd.DataFrame(rows).set_index("min_confidence")


def feature_block_ablation(
    make_pipeline: Callable[[], Any],
    features: pd.DataFrame,
    target: pd.Series,
    cv: BaseCrossValidator,
    extra_blocks: dict[str, list[str]] | None = None,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Retrain on subsets of the feature blocks to see which earn their place.

    Parameters
    ----------
    make_pipeline:
        Zero-argument factory returning a *fresh, unfitted* pipeline. A factory rather than
        an instance so each ablation starts from identical, untouched state.
    extra_blocks:
        Additional named column subsets to evaluate - for example the top-k features from
        permutation importance, to test how concentrated the signal is.

    Returns
    -------
    pandas.DataFrame
        Per block: ``n_features``, ``cv_mean``, ``cv_std``, and ``delta_vs_all`` - the
        change against using all 67 features. A block whose removal costs nothing is not
        contributing.
    """
    blocks: dict[str, list[str]] = {
        "all features": list(features.columns),
        "group blocks only (g1_ + g2_)": GROUP_1_FEATURES + GROUP_2_FEATURES,
        "comparison block only (c_)": COMPARISON_FEATURES,
        "group 1 only": GROUP_1_FEATURES,
        "group 2 only": GROUP_2_FEATURES,
    }
    if extra_blocks:
        blocks.update(extra_blocks)

    rows: list[dict[str, Any]] = []
    baseline: float | None = None

    for name, columns in blocks.items():
        available = [column for column in columns if column in features.columns]
        if not available:
            continue

        scores = cross_val_score(
            make_pipeline(), features[available], target, cv=cv, scoring=scoring
        )
        mean = float(scores.mean())
        if name == "all features":
            baseline = mean

        rows.append(
            {
                "block": name,
                "n_features": len(available),
                "cv_mean": round(mean, 4),
                "cv_std": round(float(scores.std()), 4),
            }
        )

    report = pd.DataFrame(rows).set_index("block")
    if baseline is not None:
        report["delta_vs_all"] = (report["cv_mean"] - baseline).round(4)

    return report.sort_values("cv_mean", ascending=False)


def ensemble_experiment(
    base_pipelines: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    cv: BaseCrossValidator,
    scoring: str = "accuracy",
    stacking_cv: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Test whether combining models beats the best single model.

    Two combiners are evaluated against the best individual base learner:

    ``soft voting``
        Averages predicted probabilities. Cheap, no meta-learner, no extra fitting.
    ``stacking``
        Trains a logistic-regression meta-learner on out-of-fold base predictions.
        Strictly more expressive than voting, and strictly more expensive.

    Why this is worth running even when the answer is likely "no"
    ------------------------------------------------------------
    Ensembling only helps when base learners make *different* mistakes. Published 2026
    benchmarks on tabular data report sub-percent gains at roughly 250x the compute of the
    best base model, with several ensembles performing *worse* than their best member -
    because near-identical base predictions leave nothing for a combiner to exploit. A
    field of models clustering within a couple of accuracy points is exactly that
    situation.

    A measured negative result is a finding: it says the ceiling is the data, not the
    combination strategy, and it closes the "did you try ensembling?" question with
    evidence rather than assertion.

    Parameters
    ----------
    base_pipelines:
        Named, *unfitted* pipelines. Keep this to three or four - stacking refits every
        base learner ``stacking_cv`` times inside every outer fold.

    Returns
    -------
    pandas.DataFrame
        Per approach: ``cv_mean``, ``cv_std``, ``delta_vs_best_base``, and ``verdict``.
    """
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression

    rows: list[dict[str, Any]] = []

    for name, pipeline in base_pipelines.items():
        scores = cross_val_score(clone(pipeline), features, target, cv=cv, scoring=scoring)
        rows.append(
            {
                "approach": f"base: {name}",
                "cv_mean": round(float(scores.mean()), 4),
                "cv_std": round(float(scores.std()), 4),
            }
        )

    best_base = max(rows, key=lambda row: row["cv_mean"])
    estimators = [(name, clone(pipeline)) for name, pipeline in base_pipelines.items()]

    voting = VotingClassifier(estimators=estimators, voting="soft")
    scores = cross_val_score(voting, features, target, cv=cv, scoring=scoring)
    rows.append(
        {
            "approach": "soft voting",
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }
    )

    stacking = StackingClassifier(
        estimators=[(name, clone(pipeline)) for name, pipeline in base_pipelines.items()],
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        cv=stacking_cv,
        stack_method="predict_proba",
    )
    scores = cross_val_score(stacking, features, target, cv=cv, scoring=scoring)
    rows.append(
        {
            "approach": "stacking (logistic meta-learner)",
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }
    )

    report = pd.DataFrame(rows).set_index("approach")
    report["delta_vs_best_base"] = (report["cv_mean"] - best_base["cv_mean"]).round(4)

    # A gain smaller than the base model's own fold-to-fold spread is not a real gain.
    threshold = best_base["cv_std"]
    report["verdict"] = np.where(
        report["delta_vs_best_base"] > threshold,
        "improves on best base",
        np.where(report["delta_vs_best_base"] < -threshold, "worse than best base", "within noise"),
    )

    logger.info(
        "Ensemble experiment: best base '%s' at %.4f; best combiner delta %+.4f "
        "against a fold-to-fold spread of %.4f.",
        best_base["approach"],
        best_base["cv_mean"],
        report["delta_vs_best_base"].max(),
        threshold,
    )
    return report.sort_values("cv_mean", ascending=False)


def select_confidence_threshold(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
    min_coverage: float = 0.30,
    target_accuracy: float | None = None,
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.34, 0.96, 0.02), 2)),
) -> dict[str, Any]:
    """Choose an automation threshold on **validation** data, to be frozen before testing.

    Selecting the operating point by inspecting test-set results contaminates the test
    set exactly as feature selection on test would: the threshold becomes a fitted
    parameter, and the accuracy reported at that threshold is optimistically biased.

    Call this on a held-out split that is *not* the final test set, freeze the returned
    threshold, then evaluate it once on test.

    Parameters
    ----------
    min_coverage:
        Smallest share of campaigns the automated path must still handle. Without a floor
        the search degenerates to "automate the three easiest cases at 100% accuracy".
    target_accuracy:
        If given, choose the *largest coverage* that reaches this accuracy. Otherwise
        choose the threshold with the best accuracy subject to ``min_coverage``.

    Returns
    -------
    dict
        ``threshold``, ``coverage``, ``accuracy`` on the selection split, plus the rule
        used - so the choice is auditable rather than eyeballed.
    """
    curve = operating_points(y_true, probabilities, classes=classes, thresholds=thresholds)
    eligible = curve[curve["coverage"] >= min_coverage].dropna(subset=["accuracy_covered"])

    if eligible.empty:  # pragma: no cover - only if min_coverage is unattainable
        return {
            "threshold": 0.0,
            "coverage": 1.0,
            "accuracy": float(curve["accuracy_overall"].iloc[0]),
            "rule": "no threshold met the coverage floor; automation disabled",
        }

    if target_accuracy is not None:
        reaching = eligible[eligible["accuracy_covered"] >= target_accuracy]
        chosen = (
            reaching.index.min() if not reaching.empty else eligible["accuracy_covered"].idxmax()
        )
        rule = (
            f"largest coverage reaching {target_accuracy:.0%} accuracy"
            if not reaching.empty
            else f"target accuracy unreachable; best accuracy at coverage >= {min_coverage:.0%}"
        )
    else:
        chosen = eligible["accuracy_covered"].idxmax()
        rule = f"best accuracy subject to coverage >= {min_coverage:.0%}"

    row = curve.loc[chosen]
    logger.info(
        "Selected automation threshold %.2f on the validation split (%s): "
        "coverage %.1f%%, accuracy %.4f.",
        chosen,
        rule,
        row["coverage"] * 100,
        row["accuracy_covered"],
    )
    return {
        "threshold": float(chosen),
        "coverage": float(row["coverage"]),
        "accuracy": float(row["accuracy_covered"]),
        "rule": rule,
    }


def abstention_rule_comparison(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
    decline_thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
) -> pd.DataFrame:
    """Compare ``argmax`` against an explicit "decline if P(class 0) is high" rule.

    Why this matters
    ----------------
    A model can hold perfectly reasonable beliefs about the unprofitable case and still
    never *predict* it: with three classes, ``P(class 0) = 0.40`` loses to ``P(class 1) =
    0.45`` under ``argmax``, so the decline recommendation never surfaces. Concluding
    "this model cannot identify unprofitable campaigns" from its argmax behaviour confuses
    the **decision rule** with the **model**.

    This function separates them. If a modest threshold on ``P(class 0)`` recovers most of
    the unprofitable campaigns without much cost elsewhere, the model was never the
    problem - the decision rule was.

    Returns
    -------
    pandas.DataFrame
        ``argmax`` as the first row, then one row per threshold, with ``n_declined``,
        ``class_0_recall``, ``accuracy`` and ``precision_of_decline``.
    """
    truth = np.asarray(y_true).ravel().astype(int)
    proba = np.asarray(probabilities, dtype=float)
    class_order = np.asarray(classes).astype(int)
    zero_column = int(np.where(class_order == 0)[0][0])

    argmax_pred = class_order[proba.argmax(axis=1)]
    n_class_0 = int((truth == 0).sum())

    def _row(label: str, predictions: np.ndarray) -> dict[str, Any]:
        declined = predictions == 0
        return {
            "rule": label,
            "n_declined": int(declined.sum()),
            "class_0_recall": round(
                float(((truth == 0) & declined).sum() / n_class_0) if n_class_0 else 0.0, 4
            ),
            "precision_of_decline": round(
                float(((truth == 0) & declined).sum() / declined.sum()) if declined.any() else 0.0,
                4,
            ),
            "accuracy": round(float((predictions == truth).mean()), 4),
        }

    rows = [_row("argmax", argmax_pred)]

    # Columns for the two "target a group" classes, whatever order `classes` supplies.
    group_columns = [index for index, value in enumerate(class_order) if value != 0]
    group_labels = class_order[group_columns]
    group_proba = proba[:, group_columns]

    # Between the two groups, always take the likelier one. This is independent of the
    # decline threshold, so it is computed once.
    better_group = group_labels[group_proba.argmax(axis=1)]

    for threshold in decline_thresholds:
        # Decline whenever P(class 0) clears the threshold; otherwise target the better
        # group. A single extra rule on top of argmax - deliberately the minimal change,
        # so any improvement is attributable to the rule and not to a different model.
        predictions = np.where(proba[:, zero_column] >= threshold, 0, better_group)
        rows.append(_row(f"decline if P(0) >= {threshold:.2f}", predictions))

    return pd.DataFrame(rows).set_index("rule")


def select_decline_threshold(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
    max_accuracy_loss: float = 0.01,
    decline_thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
) -> dict[str, Any]:
    """Choose the "decline if P(class 0) >= tau" threshold on **validation** data.

    The companion to :func:`abstention_rule_comparison`. That function *shows* the
    trade-off; this one *commits* to a point on it using a stated rule, so the threshold
    can be frozen before the test set is touched.

    Scanning thresholds on the test set and keeping the flattering one is the same
    contamination as selecting features there - the threshold becomes a fitted parameter
    and the reported figures are optimistic.

    Selection rule
    --------------
    Maximise class-0 recall subject to two constraints: overall accuracy falls no more
    than ``max_accuracy_loss`` below the ``argmax`` baseline, **and** class-0 recall
    strictly exceeds the ``argmax`` baseline. This encodes the business position
    explicitly: identifying unprofitable campaigns is worth a small, bounded concession
    in targeting accuracy - and *how* small is a stated parameter rather than an
    aesthetic judgement made while looking at results.

    The second constraint matters more than it looks. A decline rule routes every
    ``P(0) < tau`` case to the better of classes 1 and 2, which *removes* the class-0
    predictions argmax would have made when ``P(0)`` was the maximum but still below
    ``tau``. A badly chosen threshold therefore scores **worse** on class-0 recall than
    plain argmax, and a search that only ranks thresholds against each other will return
    it. ``threshold=None`` is the correct output when no rule earns its place.

    Returns
    -------
    dict
        ``threshold`` (``None`` when plain ``argmax`` wins), the validation metrics at
        that choice, and the rule applied - so the decision is auditable.
    """
    table = abstention_rule_comparison(
        y_true, probabilities, classes=classes, decline_thresholds=decline_thresholds
    )
    baseline_accuracy = float(table.loc["argmax", "accuracy"])
    floor = baseline_accuracy - max_accuracy_loss

    baseline_recall = float(table.loc["argmax", "class_0_recall"])
    candidates = table.drop(index="argmax")

    # Two conditions, not one. A threshold must (a) stay within the accuracy budget and
    # (b) actually beat argmax on the objective being maximised. Without (b) the search
    # happily returns a threshold whose class-0 recall is *below* the baseline it is
    # measured against - it is the best of the candidates, but the candidates are all
    # worse than doing nothing. "No rule helps" is a legitimate answer and must be
    # representable, otherwise the function manufactures a decision it has not earned.
    affordable = candidates[
        (candidates["accuracy"] >= floor) & (candidates["class_0_recall"] > baseline_recall)
    ]

    if affordable.empty:
        logger.info(
            "No decline threshold both stays within %.1f pp of argmax accuracy (%.4f) and "
            "improves on its class-0 recall (%.4f); keeping plain argmax.",
            max_accuracy_loss * 100,
            baseline_accuracy,
            baseline_recall,
        )
        return {
            "threshold": None,
            "rule": (
                f"argmax retained; no threshold beat its class-0 recall while staying "
                f"within {max_accuracy_loss:.1%} of its accuracy"
            ),
            "argmax_accuracy": round(baseline_accuracy, 4),
            "argmax_class_0_recall": round(baseline_recall, 4),
        }

    best = affordable["class_0_recall"].idxmax()
    row = affordable.loc[best]
    threshold = float(str(best).split(">=")[-1].strip())

    logger.info(
        "Selected decline threshold P(0) >= %.2f on validation: class-0 recall %.3f "
        "(argmax %.3f) at accuracy %.4f (argmax %.4f).",
        threshold,
        row["class_0_recall"],
        table.loc["argmax", "class_0_recall"],
        row["accuracy"],
        baseline_accuracy,
    )
    return {
        "threshold": threshold,
        "rule": f"max class-0 recall subject to accuracy >= argmax - {max_accuracy_loss:.1%}",
        "validation_accuracy": round(float(row["accuracy"]), 4),
        "validation_class_0_recall": round(float(row["class_0_recall"]), 4),
        "validation_decline_precision": round(float(row["precision_of_decline"]), 4),
        "argmax_accuracy": round(baseline_accuracy, 4),
        "argmax_class_0_recall": round(float(table.loc["argmax", "class_0_recall"]), 4),
    }


def apply_decline_rule(
    probabilities: np.ndarray,
    threshold: float | None,
    classes: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """Apply a frozen decline threshold to fresh probabilities.

    Kept separate from selection so the *chosen* rule can be applied to the test set
    exactly once, with no opportunity to re-tune while looking at the outcome.
    """
    proba = np.asarray(probabilities, dtype=float)
    class_order = np.asarray(classes).astype(int)

    if threshold is None:
        return class_order[proba.argmax(axis=1)]

    zero_column = int(np.where(class_order == 0)[0][0])
    group_columns = [index for index, value in enumerate(class_order) if value != 0]
    group_labels = class_order[group_columns]
    better_group = group_labels[proba[:, group_columns].argmax(axis=1)]

    return np.where(proba[:, zero_column] >= threshold, 0, better_group)


def feature_count_sweep(
    make_pipeline: Callable[[], Any],
    features: pd.DataFrame,
    target: pd.Series,
    ranked_features: Sequence[str],
    cv: BaseCrossValidator,
    counts: Sequence[int] = (1, 3, 5, 10, 20, 40, 67),
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Retrain on the top-k ranked features to locate where the signal saturates.

    With weak, diffuse signal, dropping uninformative columns can *reduce variance* enough
    to offset the information lost. This sweep finds the point at which adding features
    stops paying - and, more usefully for the report, shows how few features are needed to
    reach most of the achievable performance.

    Parameters
    ----------
    ranked_features:
        Features in descending importance order, e.g. the index of
        :func:`src.explainability.permutation_importance_report`.
    """
    rows: list[dict[str, Any]] = []
    ranked = [name for name in ranked_features if name in features.columns]

    for k in counts:
        selected = ranked[: min(k, len(ranked))]
        if not selected:
            continue
        scores = cross_val_score(
            make_pipeline(), features[selected], target, cv=cv, scoring=scoring
        )
        rows.append(
            {
                "n_features": len(selected),
                "cv_mean": round(float(scores.mean()), 4),
                "cv_std": round(float(scores.std()), 4),
            }
        )

    report = pd.DataFrame(rows).set_index("n_features")
    if not report.empty:
        best = report["cv_mean"].max()
        report["pct_of_best"] = (report["cv_mean"] / best * 100).round(1)
    return report


__all__ = [
    "abstention_rule_comparison",
    "apply_decline_rule",
    "ensemble_experiment",
    "feature_block_ablation",
    "feature_count_sweep",
    "learning_curve_report",
    "operating_points",
    "select_confidence_threshold",
    "select_decline_threshold",
]
