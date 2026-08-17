"""Train/test splitting strategies and dataset diagnostics.

Two questions are answered here, both of which are easy to get wrong and both of which
reviewers look for:

1. **Is there a time ordering hiding in this dataset?** The campaigns were run "over many
   years". If any column encodes that ordering - or if the row order itself does - then a
   random split leaks the future into the past and inflates the score. The honest
   estimate of future performance comes from training on older campaigns and testing on
   newer ones.

2. **Are the ``c_`` comparison features redundant?** If ``c_j`` is a deterministic
   function of the ``g1_``/``g2_`` blocks it adds no information, inflates the feature
   count, and dilutes permutation-importance results. Worth knowing before modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import COMPARISON_FEATURES, N_GROUP_FEATURES
from src.logging_config import get_logger

logger = get_logger(__name__)

SplitStrategy = Literal["stratified", "temporal"]


@dataclass(frozen=True)
class TimeOrderingEvidence:
    """Evidence that a column encodes the order in which campaigns were run."""

    column: str
    spearman_with_row_order: float
    monotonic_fraction: float
    n_unique: int
    looks_like_time: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "column": self.column,
            "spearman_with_row_order": self.spearman_with_row_order,
            "monotonic_fraction": self.monotonic_fraction,
            "n_unique": self.n_unique,
            "looks_like_time": self.looks_like_time,
        }


def detect_time_ordering(
    frame: pd.DataFrame,
    monotonic_threshold: float = 0.95,
) -> list[TimeOrderingEvidence]:
    """Search the feature columns for anything that behaves like a timestamp or counter.

    Heuristics, in order of strength:

    * ``monotonic_fraction`` - the share of consecutive rows where the value does not
      decrease. A true ordering column is near 1.0 (or near 0.0 if reverse-sorted).
    * ``spearman_with_row_order`` - rank correlation with the row index. High magnitude
      means the column and the file order agree.
    * ``n_unique`` - a date-like column usually has many distinct values.

    Returns
    -------
    list[TimeOrderingEvidence]
        Sorted by evidence strength, strongest first. An empty list means no candidate
        was found, in which case a stratified random split is the right choice - and you
        can say so in the report with evidence rather than by assumption.
    """
    numeric = frame.select_dtypes(include=[np.number])
    row_order = pd.Series(np.arange(len(numeric)), index=numeric.index)
    evidence: list[TimeOrderingEvidence] = []

    for column in numeric.columns:
        values = numeric[column]
        if values.notna().sum() < 3 or values.nunique(dropna=True) < 3:
            continue

        differences = values.diff().dropna()
        if differences.empty:
            continue

        non_decreasing = float((differences >= 0).mean())
        monotonic_fraction = max(non_decreasing, 1.0 - non_decreasing)

        try:
            spearman = float(values.corr(row_order, method="spearman"))
        except Exception:  # noqa: BLE001 - degenerate columns
            spearman = 0.0
        spearman = 0.0 if np.isnan(spearman) else spearman

        looks_like_time = (
            monotonic_fraction >= monotonic_threshold or abs(spearman) >= monotonic_threshold
        )

        evidence.append(
            TimeOrderingEvidence(
                column=str(column),
                spearman_with_row_order=round(spearman, 4),
                monotonic_fraction=round(monotonic_fraction, 4),
                n_unique=int(values.nunique(dropna=True)),
                looks_like_time=looks_like_time,
            )
        )

    evidence.sort(
        key=lambda item: (
            item.looks_like_time,
            item.monotonic_fraction,
            abs(item.spearman_with_row_order),
        ),
        reverse=True,
    )

    candidates = [item.column for item in evidence if item.looks_like_time]
    if candidates:
        logger.info("Possible time-ordering column(s) detected: %s", candidates)
    else:
        logger.info("No time-ordering column detected; a stratified split is appropriate.")

    return evidence


def make_split(
    features: pd.DataFrame,
    target: pd.Series,
    strategy: SplitStrategy = "stratified",
    test_size: float = 0.2,
    random_state: int = 42,
    order_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test using the requested strategy.

    Parameters
    ----------
    strategy:
        ``"stratified"`` - random split preserving class proportions. Correct when the
        rows are exchangeable.
        ``"temporal"`` - the last ``test_size`` fraction becomes the test set, so the
        model is always evaluated on campaigns that ran *after* the ones it learned
        from. This is the honest protocol when an ordering exists.
    order_by:
        Column to sort by before a temporal split. When ``None`` the existing row order
        is assumed to be chronological, which is the common convention for exported
        campaign histories - state the assumption explicitly in the report.
    """
    if strategy == "stratified":
        return train_test_split(
            features,
            target,
            test_size=test_size,
            stratify=target,
            random_state=random_state,
        )

    if strategy != "temporal":  # pragma: no cover - guarded by Literal
        raise ValueError(f"Unknown split strategy: {strategy}")

    if order_by is not None:
        if order_by not in features.columns:
            raise KeyError(f"order_by column '{order_by}' is not in the feature frame.")
        order = features[order_by].sort_values(kind="mergesort").index
    else:
        order = features.index

    ordered_features = features.loc[order]
    ordered_target = target.loc[order]

    cut = int(len(ordered_features) * (1.0 - test_size))
    if cut <= 0 or cut >= len(ordered_features):
        raise ValueError("test_size produces an empty train or test split.")

    x_train = ordered_features.iloc[:cut]
    x_test = ordered_features.iloc[cut:]
    y_train = ordered_target.iloc[:cut]
    y_test = ordered_target.iloc[cut:]

    missing_classes = set(y_train.unique()).symmetric_difference(y_test.unique())
    if missing_classes:
        logger.warning(
            "Temporal split leaves class(es) %s on only one side - the class balance "
            "shifts over time, which is itself worth reporting.",
            sorted(missing_classes),
        )

    logger.info("Temporal split: %d train rows, %d test rows.", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def comparison_redundancy_report(
    frame: pd.DataFrame,
    r2_threshold: float = 0.999,
) -> pd.DataFrame:
    """Test whether each ``c_`` feature is a deterministic function of the group blocks.

    Fits a plain least-squares regression of each ``c_j`` on the 40 group variables plus
    their 20 pairwise differences. An R^2 at or above ``r2_threshold`` means the feature
    is (numerically) fully reconstructible and carries no independent information.

    This is a cheap check with real consequences: redundant columns dilute permutation
    importance, inflate the apparent feature count, and can mask the genuinely
    informative comparison features.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature, with ``r2``, ``is_redundant`` and ``max_abs_corr_with_diff``,
        sorted by ``r2`` descending.
    """
    group_columns = [
        column
        for index in range(1, N_GROUP_FEATURES + 1)
        for column in (f"g1_{index}", f"g2_{index}")
        if column in frame.columns
    ]
    if not group_columns:
        raise ValueError("Frame contains no g1_/g2_ columns to regress against.")

    differences = pd.DataFrame(
        {
            f"diff_{index}": frame[f"g1_{index}"] - frame[f"g2_{index}"]
            for index in range(1, N_GROUP_FEATURES + 1)
            if f"g1_{index}" in frame.columns and f"g2_{index}" in frame.columns
        },
        index=frame.index,
    )

    design = pd.concat([frame[group_columns], differences], axis=1)
    design = design.apply(pd.to_numeric, errors="coerce")

    rows: list[dict[str, Any]] = []
    for column in [name for name in COMPARISON_FEATURES if name in frame.columns]:
        response = pd.to_numeric(frame[column], errors="coerce")

        usable = design.notna().all(axis=1) & response.notna()
        if usable.sum() <= design.shape[1]:
            rows.append(
                {
                    "feature": column,
                    "r2": float("nan"),
                    "is_redundant": False,
                    "max_abs_corr_with_diff": float("nan"),
                }
            )
            continue

        matrix = np.column_stack([np.ones(usable.sum()), design.loc[usable].to_numpy(dtype=float)])
        observed = response.loc[usable].to_numpy(dtype=float)

        coefficients, *_ = np.linalg.lstsq(matrix, observed, rcond=None)
        residuals = observed - matrix @ coefficients
        total_variance = float(((observed - observed.mean()) ** 2).sum())
        r2 = 1.0 - float((residuals**2).sum()) / total_variance if total_variance > 0 else 0.0

        correlations = differences.loc[usable].corrwith(response.loc[usable]).dropna()
        max_corr = float(correlations.abs().max()) if not correlations.empty else float("nan")

        rows.append(
            {
                "feature": column,
                "r2": round(r2, 6),
                "is_redundant": bool(r2 >= r2_threshold),
                "max_abs_corr_with_diff": round(max_corr, 4),
            }
        )

    report = pd.DataFrame(rows).set_index("feature").sort_values("r2", ascending=False)
    n_redundant = int(report["is_redundant"].sum())
    logger.info(
        "Comparison redundancy check: %d of %d c_ features are reconstructible from "
        "the group blocks.",
        n_redundant,
        len(report),
    )
    return report


__all__ = [
    "SplitStrategy",
    "TimeOrderingEvidence",
    "comparison_redundancy_report",
    "detect_time_ordering",
    "make_split",
]
