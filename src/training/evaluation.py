"""Model evaluation and business-impact estimation.

This module answers ML questions 1 and 3 of the challenge:

* :func:`campaign_outcome_distribution` - what share of campaigns favoured group 1,
  group 2, or neither.
* :func:`estimate_business_lift` - how much the model improves the campaign success
  rate over the naive strategies currently available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.constants import CLASS_DESCRIPTIONS, CLASS_LABELS


def as_label_vector(values: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce predictions or targets to a flat integer array.

    Not defensive boilerplate - a genuine correctness guard. CatBoost's ``predict``
    returns a column vector of shape ``(n, 1)`` for multiclass problems, while scikit-learn
    estimators return ``(n,)``. Comparing a ``(n,)`` target with a ``(n, 1)`` prediction
    broadcasts into an ``(n, n)`` boolean matrix, so ``(y_true == y_pred).mean()`` silently
    returns the probability that two random labels agree rather than the accuracy.

    That produced a *negative* business lift for an otherwise reasonable model. Flattening
    at every entry point makes the metric independent of which library produced the
    predictions.
    """
    return np.asarray(values).ravel().astype(int)


def campaign_outcome_distribution(target: pd.Series) -> dict[str, Any]:
    """Answer ML question 1: the historical split of campaign outcomes.

    Returns counts and percentages for every class, including classes that never occur.
    """
    counts = target.value_counts().reindex(sorted(CLASS_LABELS), fill_value=0)
    total = int(counts.sum())
    percentages = (counts / total * 100).round(2) if total else counts.astype(float)

    return {
        "total_campaigns": total,
        "by_class": {
            CLASS_LABELS[int(class_value)]: {
                "class": int(class_value),
                "description": CLASS_DESCRIPTIONS[int(class_value)],
                "count": int(counts.loc[class_value]),
                "percentage": float(percentages.loc[class_value]),
            }
            for class_value in counts.index
        },
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute the metric set used to compare candidate models.

    Macro F1 and balanced accuracy are the headline numbers because the three classes
    are unlikely to be balanced, which would make plain accuracy misleading.
    """
    y_true, y_pred = as_label_vector(y_true), as_label_vector(y_pred)
    labels = sorted(CLASS_LABELS)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": [CLASS_LABELS[label] for label in labels],
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


@dataclass(frozen=True)
class BusinessLift:
    """Campaign success-rate comparison between the model and naive strategies."""

    n_campaigns: int
    model_success_rate: float
    random_choice_success_rate: float
    always_group_1_success_rate: float
    always_group_2_success_rate: float
    always_decline_success_rate: float
    best_naive_strategy: str
    best_naive_success_rate: float
    absolute_lift_pp: float
    relative_lift_pct: float
    wasted_campaigns_avoided: int
    wasted_campaign_avoidance_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def estimate_business_lift(y_true: np.ndarray, y_pred: np.ndarray) -> BusinessLift:
    """Answer ML question 3: quantify the improvement in campaign success rate.

    Definitions
    -----------
    *Success* means the decision taken matches the historically profitable outcome:
    picking the group that actually turned out most profitable, or correctly declining
    to run a campaign where neither group was profitable (class 0).

    Naive baselines compared against:

    ``random_choice``
        Pick group 1 or group 2 with equal probability, never declining. Its expected
        success rate is half the share of campaigns where some group was profitable.
    ``always_group_1`` / ``always_group_2``
        Always target the same group.
    ``always_decline``
        Run no campaigns at all. Its success rate is P(class 0). This baseline is easy to
        forget and is frequently the **strongest** one: if most campaigns were
        unprofitable, doing nothing beats every targeting strategy. Omitting it would
        measure the lift against the wrong number and overstate the model's value.

    ``wasted_campaigns_avoided`` counts the class-0 campaigns the model correctly
    declines - direct budget saved rather than revenue gained.
    """
    y_true = as_label_vector(y_true)
    y_pred = as_label_vector(y_pred)
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true has {y_true.size} entries but y_pred has {y_pred.size}.")
    n = int(y_true.size)
    if n == 0:
        raise ValueError("Cannot estimate lift on an empty evaluation set.")

    model_rate = float((y_true == y_pred).mean())

    share_group_1 = float((y_true == 1).mean())
    share_group_2 = float((y_true == 2).mean())
    share_decline = float((y_true == 0).mean())
    share_profitable = share_group_1 + share_group_2

    random_rate = 0.5 * share_profitable
    naive_rates = {
        "random_choice": random_rate,
        "always_group_1": share_group_1,
        "always_group_2": share_group_2,
        "always_decline": share_decline,
    }
    best_naive = max(naive_rates, key=lambda key: naive_rates[key])
    best_naive_rate = naive_rates[best_naive]

    n_class_0 = int((y_true == 0).sum())
    avoided = int(((y_true == 0) & (y_pred == 0)).sum())

    absolute_lift = model_rate - best_naive_rate
    relative_lift = (absolute_lift / best_naive_rate * 100) if best_naive_rate > 0 else float("inf")

    return BusinessLift(
        n_campaigns=n,
        model_success_rate=round(model_rate, 4),
        random_choice_success_rate=round(random_rate, 4),
        always_group_1_success_rate=round(share_group_1, 4),
        always_group_2_success_rate=round(share_group_2, 4),
        always_decline_success_rate=round(share_decline, 4),
        best_naive_strategy=best_naive,
        best_naive_success_rate=round(best_naive_rate, 4),
        absolute_lift_pp=round(absolute_lift * 100, 2),
        relative_lift_pct=round(relative_lift, 2),
        wasted_campaigns_avoided=avoided,
        wasted_campaign_avoidance_rate=round(avoided / n_class_0, 4) if n_class_0 else 0.0,
    )


def bootstrap_lift_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Put a confidence interval around the headline lift number.

    A single point estimate such as "+18 percentage points" is the classic junior
    reporting mistake: on a modest test set that number carries real uncertainty, and a
    reviewer will ask how much. This resamples the evaluation set with replacement,
    recomputes the lift each time, and returns the percentile interval.

    Both the model's success rate and the naive baselines are recomputed inside every
    resample, because the baselines are themselves estimated from the same data - holding
    them fixed would understate the uncertainty.

    Returns
    -------
    dict
        ``point_estimate_pp`` plus ``ci_lower_pp`` / ``ci_upper_pp``, the standard error,
        the interval for the model success rate, and ``significantly_positive`` - True
        when the whole interval lies above zero.
    """
    truth = as_label_vector(y_true)
    predictions = as_label_vector(y_pred)
    n = int(truth.size)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    rng = np.random.default_rng(random_state)
    lifts = np.empty(n_resamples, dtype=float)
    success_rates = np.empty(n_resamples, dtype=float)

    for index in range(n_resamples):
        sample = rng.integers(0, n, size=n)
        resampled_truth = truth[sample]
        resampled_pred = predictions[sample]

        model_rate = float((resampled_truth == resampled_pred).mean())
        share_0 = float((resampled_truth == 0).mean())
        share_1 = float((resampled_truth == 1).mean())
        share_2 = float((resampled_truth == 2).mean())
        best_naive = max(0.5 * (share_1 + share_2), share_1, share_2, share_0)

        success_rates[index] = model_rate
        lifts[index] = (model_rate - best_naive) * 100.0

    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.percentile(lifts, [100 * tail, 100 * (1.0 - tail)])
    rate_lower, rate_upper = np.percentile(success_rates, [100 * tail, 100 * (1.0 - tail)])

    point = estimate_business_lift(truth, predictions).absolute_lift_pp

    return {
        "n_resamples": int(n_resamples),
        "confidence_level": confidence_level,
        "point_estimate_pp": point,
        "ci_lower_pp": round(float(lower), 2),
        "ci_upper_pp": round(float(upper), 2),
        "standard_error_pp": round(float(lifts.std(ddof=1)), 3),
        "model_success_rate_ci": [round(float(rate_lower), 4), round(float(rate_upper), 4)],
        "significantly_positive": bool(lower > 0.0),
    }
