"""Single source of truth for the dataset contract.

Everything that depends on the shape of ``customerGroups.csv`` is declared here so
that training, serving and tests can never drift apart.
"""

from __future__ import annotations

from typing import Final

#: Number of per-group variables known *before* the campaign was run.
N_GROUP_FEATURES: Final[int] = 20

#: Number of comparison variables known *before* the campaign was run.
N_COMPARISON_FEATURES: Final[int] = 27

GROUP_1_FEATURES: Final[list[str]] = [f"g1_{i}" for i in range(1, N_GROUP_FEATURES + 1)]
GROUP_2_FEATURES: Final[list[str]] = [f"g2_{i}" for i in range(1, N_GROUP_FEATURES + 1)]
COMPARISON_FEATURES: Final[list[str]] = [f"c_{i}" for i in range(1, N_COMPARISON_FEATURES + 1)]

#: The 67 columns that are legitimately available at prediction time.
BASE_FEATURES: Final[list[str]] = GROUP_1_FEATURES + GROUP_2_FEATURES + COMPARISON_FEATURES

#: Columns recorded *after* the campaign ran. Using them as model inputs is target
#: leakage: they would not exist when the model is called in production.
LEAKAGE_FEATURES: Final[list[str]] = ["g1_21", "g2_21", "c_28"]

TARGET_COLUMN: Final[str] = "target"

#: Business meaning of each target class, taken verbatim from the task description.
CLASS_LABELS: Final[dict[int, str]] = {
    0: "no_group_profitable",
    1: "group_1",
    2: "group_2",
}

CLASS_DESCRIPTIONS: Final[dict[int, str]] = {
    0: "None of the two groups were profitable",
    1: "Group 1 was the most profitable",
    2: "Group 2 was the most profitable",
}

#: The action a campaign manager should take for each predicted class.
RECOMMENDED_ACTIONS: Final[dict[int, str]] = {
    0: "Do not run this campaign - neither group is expected to be profitable.",
    1: "Target customer group 1.",
    2: "Target customer group 2.",
}

#: Observed class frequencies across the 6,620 historical campaigns.
#:
#: Used as the prior for :class:`~src.predictors.MajorityClassPredictor`, the baseline the
#: API degrades to when no trained artifact is available. The alternative - putting all
#: probability mass on the majority class - would make the baseline report
#: ``confidence: 1.0`` while being correct only 46% of the time. That is not a cosmetic
#: problem: the confidence value is consumed by the cost-sensitive decision layer and by
#: the automation gate, both of which would treat a coin-flip as a certainty.
#:
#: These are aggregate frequencies, not records, so they are safe to hold in source. They
#: describe one specific dataset; retraining on new data should overwrite them from the
#: artifact rather than relying on this fallback.
HISTORICAL_CLASS_DISTRIBUTION: Final[dict[str, float]] = {
    "no_group_profitable": 0.2518,
    "group_1": 0.4647,
    "group_2": 0.2835,
}
