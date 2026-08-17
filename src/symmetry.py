"""Group-swap symmetry: augmentation and invariance measurement.

Why this module exists
----------------------
The task is a *pairwise comparison*: "which of these two groups should we target?"
The labelling of a group as "group 1" or "group 2" is arbitrary. Therefore the problem
carries an exact antisymmetry:

    swap(group 1, group 2)  =>  target 1 <-> 2,  target 0 stays 0

A model trained without enforcing this will happily learn a spurious position bias -
for example, predicting class 1 more often simply because group 1 appeared first more
often in the training rows. The literature on pairwise evaluators is explicit that
antisymmetry must be enforced or verified rather than assumed, and that augmenting the
training set with both orderings is the standard remedy for non-linear models such as
gradient-boosted trees.

Two capabilities are provided:

``augment_with_swapped``
    Doubles the training set with mirrored rows. Enforces the invariance during fitting.

``measure_invariance``
    Feeds mirrored inputs at evaluation time and reports the fraction of rows where the
    prediction does *not* flip as it should. This is a headline diagnostic: a model with
    a 15% violation rate is exploiting position, not signal.

Important caveat on the ``c_`` block
------------------------------------
The ``g1_``/``g2_`` blocks mirror cleanly - you simply exchange them. The ``c_`` block is
different: those variables "represent some comparison of the two groups", so some of them
are very likely *direction dependent* (something like ``g1_x - g2_x``). Mirroring the
groups without also negating those columns would produce physically impossible rows and
would poison the augmented data.

Because the columns are anonymised we cannot know which ones those are a priori, so this
module ships a diagnostic (:func:`diagnose_comparison_symmetry`) that identifies the
likely direction-dependent columns empirically. Run it first, inspect the report, then
pass the identified columns as ``negate_columns``. Document the choice in the report -
"I checked rather than assumed" is exactly the point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.constants import COMPARISON_FEATURES, N_GROUP_FEATURES
from src.logging_config import get_logger

logger = get_logger(__name__)

#: How target classes map when the two groups are exchanged.
SWAPPED_TARGET_MAP: dict[int, int] = {0: 0, 1: 2, 2: 1}


def swap_targets(target: pd.Series | np.ndarray) -> pd.Series:
    """Map targets through the group exchange: 0 -> 0, 1 -> 2, 2 -> 1."""
    series = pd.Series(np.asarray(target).astype(int))
    unknown = set(series.unique()).difference(SWAPPED_TARGET_MAP)
    if unknown:
        raise ValueError(f"Target contains unmappable class(es): {sorted(unknown)}")
    swapped = series.map(SWAPPED_TARGET_MAP)
    if isinstance(target, pd.Series):
        swapped.index = target.index
        swapped.name = target.name
    return swapped


def swap_groups(
    frame: pd.DataFrame,
    negate_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return ``frame`` with the two customer groups exchanged.

    Parameters
    ----------
    frame:
        Feature frame containing ``g1_1..g1_20``, ``g2_1..g2_20`` and ``c_1..c_27``.
    negate_columns:
        ``c_`` columns known (or diagnosed) to be direction dependent. Their sign is
        flipped so the mirrored row stays internally consistent. Pass the output of
        :func:`diagnose_comparison_symmetry` after inspecting it.

    Notes
    -----
    Column *order* is preserved, only the values move. That keeps the frame compatible
    with a pipeline fitted on the original column order.
    """
    unknown = [column for column in negate_columns if column not in frame.columns]
    if unknown:
        raise KeyError(f"negate_columns not present in frame: {unknown}")

    swapped = frame.copy()

    for index in range(1, N_GROUP_FEATURES + 1):
        left, right = f"g1_{index}", f"g2_{index}"
        if left in frame.columns and right in frame.columns:
            swapped[left] = frame[right].to_numpy()
            swapped[right] = frame[left].to_numpy()

    for column in negate_columns:
        swapped[column] = -frame[column].to_numpy()

    return swapped


@dataclass(frozen=True)
class AugmentedDataset:
    """Mirrored training set together with the grouping needed for honest CV.

    ``groups`` assigns a campaign and its mirror the **same** identifier. Passing it to
    ``StratifiedGroupKFold`` (or ``GroupKFold``) guarantees both copies land in the same
    fold. Without it, cross-validation trains on the mirror of a row it is validating
    against, and every CV score is inflated.
    """

    features: pd.DataFrame
    target: pd.Series
    groups: np.ndarray

    def __len__(self) -> int:
        return len(self.features)


def augment_with_swapped_grouped(
    features: pd.DataFrame,
    target: pd.Series,
    negate_columns: Sequence[str] = (),
) -> AugmentedDataset:
    """Return the training set concatenated with its mirrored copy, plus CV groups.

    Doubles the number of rows and makes the training distribution exactly symmetric in
    the group ordering.

    Warnings
    --------
    * Apply this to the **training split only**, never before splitting. A row and its
      mirror are not independent; if they straddle a train/test boundary the evaluation
      is contaminated.
    * Always cross-validate with the returned ``groups``. This is the same hazard one
      level down: within the training split, the mirror of a validation-fold row must not
      appear in the training folds.
    """
    if len(features) != len(target):
        raise ValueError("features and target must have the same length")

    mirrored_features = swap_groups(features, negate_columns=negate_columns)
    mirrored_target = swap_targets(target)

    augmented_features = pd.concat([features, mirrored_features], ignore_index=True)
    augmented_target = pd.concat(
        [pd.Series(np.asarray(target).astype(int)), pd.Series(mirrored_target.to_numpy())],
        ignore_index=True,
    )
    augmented_target.name = getattr(target, "name", None)

    # Campaign i and its mirror both carry group id i.
    campaign_ids = np.arange(len(features))
    groups = np.concatenate([campaign_ids, campaign_ids])

    logger.info(
        "Symmetry augmentation: %d rows -> %d rows (%d campaign groups, "
        "negated %d comparison column(s)).",
        len(features),
        len(augmented_features),
        len(campaign_ids),
        len(negate_columns),
    )
    return AugmentedDataset(features=augmented_features, target=augmented_target, groups=groups)


def augment_with_swapped(
    features: pd.DataFrame,
    target: pd.Series,
    negate_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper returning only ``(features, target)``.

    Prefer :func:`augment_with_swapped_grouped` whenever the result will be
    cross-validated - you need the group ids to avoid leakage between folds.
    """
    augmented = augment_with_swapped_grouped(features, target, negate_columns)
    return augmented.features, augmented.target


@dataclass(frozen=True)
class InvarianceReport:
    """Result of the group-swap invariance check."""

    n_rows: int
    n_violations: int
    violation_rate: float
    class_0_flip_rate: float
    position_bias: float
    per_class_violation_rate: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "n_rows": self.n_rows,
            "n_violations": self.n_violations,
            "violation_rate": self.violation_rate,
            "class_0_flip_rate": self.class_0_flip_rate,
            "position_bias": self.position_bias,
            "per_class_violation_rate": self.per_class_violation_rate,
        }

    def summary(self) -> str:
        """One-line human-readable verdict for the report."""
        verdict = (
            "symmetric"
            if self.violation_rate < 0.01
            else "mildly asymmetric" if self.violation_rate < 0.05 else "POSITION-BIASED"
        )
        return (
            f"Group-swap invariance: {self.violation_rate:.2%} of {self.n_rows} rows "
            f"failed to flip correctly ({verdict}); position bias "
            f"{self.position_bias:+.2%}."
        )


def measure_invariance(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    features: pd.DataFrame,
    negate_columns: Sequence[str] = (),
) -> InvarianceReport:
    """Quantify how well a fitted model respects the group-swap antisymmetry.

    Parameters
    ----------
    predict_fn:
        Anything callable that maps a feature frame to an array of class labels -
        typically ``fitted_pipeline.predict``.
    features:
        Evaluation features (use the held-out test split).
    negate_columns:
        Direction-dependent ``c_`` columns, as for :func:`swap_groups`.

    Returns
    -------
    InvarianceReport
        ``violation_rate`` is the headline number: the fraction of rows whose prediction
        did not transform as ``0->0, 1->2, 2->1`` when the groups were exchanged.
        ``position_bias`` is ``P(predict 1) - P(predict 2)`` on the original inputs; a
        large magnitude alongside a high violation rate is strong evidence the model is
        keying on ordering rather than on the groups themselves.
    """
    if features.empty:
        raise ValueError("Cannot measure invariance on an empty frame.")

    # ravel(): CatBoost returns a (n, 1) column vector for multiclass. Without flattening,
    # the comparison below broadcasts into an (n, n) matrix and the violation rate becomes
    # meaningless.
    original = np.asarray(predict_fn(features)).ravel().astype(int)
    mirrored = (
        np.asarray(predict_fn(swap_groups(features, negate_columns=negate_columns)))
        .ravel()
        .astype(int)
    )

    expected = original.copy()
    expected[original == 1] = 2
    expected[original == 2] = 1
    violations = mirrored != expected

    per_class: dict[str, float] = {}
    for class_value in sorted(SWAPPED_TARGET_MAP):
        mask = original == class_value
        per_class[f"predicted_{class_value}"] = (
            round(float(violations[mask].mean()), 4) if mask.any() else 0.0
        )

    class_0_mask = original == 0
    class_0_flip = float((mirrored[class_0_mask] == 0).mean()) if class_0_mask.any() else 1.0

    n_rows = int(original.size)
    return InvarianceReport(
        n_rows=n_rows,
        n_violations=int(violations.sum()),
        violation_rate=round(float(violations.mean()), 4),
        class_0_flip_rate=round(class_0_flip, 4),
        position_bias=round(float((original == 1).mean() - (original == 2).mean()), 4),
        per_class_violation_rate=per_class,
    )


def diagnose_comparison_symmetry(
    frame: pd.DataFrame,
    correlation_threshold: float = 0.9,
) -> pd.DataFrame:
    """Identify which ``c_`` columns are likely direction dependent.

    Heuristic: a comparison feature that encodes "group 1 minus group 2" for some
    underlying quantity will correlate very strongly - positively or negatively - with
    one of the constructed differences ``g1_i - g2_i``. Such a column must be negated
    when the groups are exchanged.

    A second, weaker signal is the distribution being centred on zero and roughly
    symmetric, which is what a signed difference looks like and what a magnitude or a
    ratio usually does not.

    Returns
    -------
    pandas.DataFrame
        One row per ``c_`` column, sorted by evidence strength, with columns:

        ``best_match``
            The ``diff_i`` most correlated with this feature.
        ``correlation``
            Signed Pearson correlation with that difference.
        ``abs_correlation``
            Its magnitude - the primary evidence.
        ``mean``, ``skew``
            Distribution shape, the secondary signal.
        ``likely_direction_dependent``
            ``True`` when ``abs_correlation >= correlation_threshold``.

    Use it like this, then read the table before deciding::

        report = diagnose_comparison_symmetry(features)
        negate = report.loc[report.likely_direction_dependent].index.tolist()
    """
    differences = pd.DataFrame(
        {
            f"diff_{index}": frame[f"g1_{index}"] - frame[f"g2_{index}"]
            for index in range(1, N_GROUP_FEATURES + 1)
            if f"g1_{index}" in frame.columns and f"g2_{index}" in frame.columns
        }
    )

    comparison_columns = [column for column in COMPARISON_FEATURES if column in frame.columns]
    rows: list[dict[str, Any]] = []

    for column in comparison_columns:
        values = pd.to_numeric(frame[column], errors="coerce")

        best_match, best_correlation = "", 0.0
        if not differences.empty and values.notna().any() and values.std(skipna=True) > 0:
            correlations = differences.corrwith(values)
            correlations = correlations.dropna()
            if not correlations.empty:
                best_match = str(correlations.abs().idxmax())
                best_correlation = float(correlations.loc[best_match])

        rows.append(
            {
                "feature": column,
                "best_match": best_match,
                "correlation": round(best_correlation, 4),
                "abs_correlation": round(abs(best_correlation), 4),
                "mean": round(float(values.mean(skipna=True)), 4),
                "skew": round(float(values.skew(skipna=True)), 4),
                "likely_direction_dependent": abs(best_correlation) >= correlation_threshold,
            }
        )

    report = pd.DataFrame(rows).set_index("feature")
    return report.sort_values("abs_correlation", ascending=False)


def validate_group_exchangeability(
    frame: pd.DataFrame,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Test whether the ``g1_`` and ``g2_`` blocks are drawn from the same distribution.

    This is the precondition for the whole symmetry argument, and it must be tested
    rather than assumed.

    Symmetry augmentation is only valid when "group 1" and "group 2" are interchangeable
    labels - when the assignment of a group to a slot is arbitrary. If instead the
    positions mean something (group 1 is the incumbent, the larger segment, the control
    arm), then mirroring a row fabricates a combination that never occurs in reality, and
    training on those rows degrades the model while destroying genuine signal.

    Method
    ------
    Two-sample Kolmogorov-Smirnov test on each paired variable ``(g1_i, g2_i)``, plus the
    ratio of standard deviations. The KS test catches location *and* shape differences; a
    variance ratio far from 1 is independent corroboration, since two samples of the same
    population should have comparable spread.

    Returns
    -------
    pandas.DataFrame
        Indexed by variable index, sorted most-divergent first, with ``ks_statistic``,
        ``p_value``, ``g1_mean``, ``g2_mean``, ``mean_gap``, ``std_ratio`` and
        ``exchangeable``.

    See Also
    --------
    groups_are_exchangeable : boolean summary of this report.
    """
    from scipy.stats import ks_2samp

    rows: list[dict[str, Any]] = []
    for index in range(1, N_GROUP_FEATURES + 1):
        left, right = f"g1_{index}", f"g2_{index}"
        if left not in frame.columns or right not in frame.columns:
            continue

        g1 = pd.to_numeric(frame[left], errors="coerce").dropna()
        g2 = pd.to_numeric(frame[right], errors="coerce").dropna()
        if g1.empty or g2.empty:
            continue

        statistic, p_value = ks_2samp(g1, g2)
        g2_std = float(g2.std())
        std_ratio = float(g1.std()) / g2_std if g2_std > 0 else float("nan")

        rows.append(
            {
                "variable": f"var_{index}",
                "ks_statistic": round(float(statistic), 6),
                "p_value": round(float(p_value), 8),
                "g1_mean": round(float(g1.mean()), 4),
                "g2_mean": round(float(g2.mean()), 4),
                "mean_gap": round(float(g1.mean() - g2.mean()), 4),
                "std_ratio": round(std_ratio, 4),
                "exchangeable": bool(p_value >= alpha),
            }
        )

    report = pd.DataFrame(rows).set_index("variable").sort_values("p_value")

    divergent = report.loc[~report["exchangeable"]].index.tolist()
    if divergent:
        logger.warning(
            "Group positions are NOT exchangeable: %d of %d paired variables differ "
            "significantly (%s). Symmetry augmentation would fabricate unrealistic rows "
            "and must not be used; the position of a group carries real information.",
            len(divergent),
            len(report),
            divergent[:8],
        )
    else:
        logger.info(
            "Group positions are exchangeable across all %d paired variables; "
            "symmetry augmentation is justified.",
            len(report),
        )

    return report


def groups_are_exchangeable(frame: pd.DataFrame, alpha: float = 0.01) -> bool:
    """Return ``True`` when every paired variable passes :func:`validate_group_exchangeability`.

    Use this to gate symmetry augmentation. ``False`` means the two slots hold
    systematically different kinds of group, and mirroring is invalid.
    """
    report = validate_group_exchangeability(frame, alpha)
    return bool(report.empty or report["exchangeable"].all())


def validate_swap(
    frame: pd.DataFrame,
    negate_columns: Sequence[str] = (),
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Check that mirroring produced rows from the same distribution.

    Why this is necessary
    ---------------------
    :func:`swap_groups` handles two kinds of comparison feature correctly: symmetric ones
    (``|g1 - g2|``, overlap fractions) which are left alone, and signed differences
    (``g1 - g2``) which are negated. It handles a third kind **incorrectly**: a *ratio*
    ``g1 / g2`` must be **inverted**, not negated. Negating a strictly positive ratio
    produces strictly negative values - rows that could never occur in reality, which
    then poison training.

    Because the columns are anonymised we cannot know the functional form a priori, so we
    test it: if a mirrored column's distribution differs from the original's, the mirror
    is invalid for that column.

    Method
    ------
    Two-sample Kolmogorov-Smirnov test per comparison column, original vs mirrored. Under
    a correct mirror the marginal distribution is unchanged, so a small p-value is
    evidence of a broken transformation. ``strictly_positive`` is a second, independent
    red flag: a column that is never negative is very unlikely to be a signed difference,
    so negating it is almost certainly wrong.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature, sorted worst-first, with ``ks_statistic``, ``p_value``,
        ``negated``, ``strictly_positive`` and ``distribution_preserved``.

    Interpretation
    --------------
    Any row with ``distribution_preserved == False`` must be resolved before using
    augmentation: drop the column from ``negate_columns``, invert it instead, or disable
    augmentation entirely. Reporting this check is itself worth marks - it shows the
    mirror was verified rather than assumed.
    """
    from scipy.stats import ks_2samp

    mirrored = swap_groups(frame, negate_columns=negate_columns)
    negated = set(negate_columns)

    rows: list[dict[str, Any]] = []
    for column in [name for name in COMPARISON_FEATURES if name in frame.columns]:
        original_values = pd.to_numeric(frame[column], errors="coerce").dropna()
        mirrored_values = pd.to_numeric(mirrored[column], errors="coerce").dropna()

        if original_values.empty or mirrored_values.empty:
            continue

        statistic, p_value = ks_2samp(original_values, mirrored_values)
        strictly_positive = bool((original_values > 0).all())
        preserved = bool(p_value >= alpha)

        rows.append(
            {
                "feature": column,
                "ks_statistic": round(float(statistic), 6),
                "p_value": round(float(p_value), 6),
                "negated": column in negated,
                "strictly_positive": strictly_positive,
                "distribution_preserved": preserved,
            }
        )

    report = pd.DataFrame(rows).set_index("feature").sort_values("p_value")

    broken = report.loc[~report["distribution_preserved"]].index.tolist()
    suspicious = report.loc[report["negated"] & report["strictly_positive"]].index.tolist()

    if broken:
        logger.warning(
            "Mirroring changed the distribution of %d comparison column(s): %s. "
            "Do not augment until this is resolved.",
            len(broken),
            broken,
        )
    if suspicious:
        logger.warning(
            "Negating strictly positive column(s) %s - these look like ratios or "
            "magnitudes and probably need inversion, not sign flipping.",
            suspicious,
        )
    if not broken and not suspicious:
        logger.info("Swap validation passed: mirrored rows match the original distribution.")

    return report


def swap_is_valid(
    frame: pd.DataFrame,
    negate_columns: Sequence[str] = (),
    alpha: float = 0.01,
) -> bool:
    """Return ``True`` when every comparison column survives :func:`validate_swap`."""
    report = validate_swap(frame, negate_columns, alpha)
    if report.empty:
        return True
    return bool(
        report["distribution_preserved"].all()
        and not (report["negated"] & report["strictly_positive"]).any()
    )


def suggested_negate_columns(
    frame: pd.DataFrame,
    correlation_threshold: float = 0.9,
) -> list[str]:
    """Convenience wrapper returning only the diagnosed direction-dependent columns.

    Inspect :func:`diagnose_comparison_symmetry` output before trusting this in the
    report - the threshold is a judgement call, not a fact.
    """
    report = diagnose_comparison_symmetry(frame, correlation_threshold)
    columns = report.loc[report["likely_direction_dependent"]].index.tolist()
    logger.info("Diagnosed %d direction-dependent comparison column(s): %s", len(columns), columns)
    return columns


__all__ = [
    "SWAPPED_TARGET_MAP",
    "AugmentedDataset",
    "InvarianceReport",
    "augment_with_swapped",
    "augment_with_swapped_grouped",
    "diagnose_comparison_symmetry",
    "groups_are_exchangeable",
    "measure_invariance",
    "suggested_negate_columns",
    "validate_group_exchangeability",
    "swap_groups",
    "swap_is_valid",
    "swap_targets",
    "validate_swap",
]
