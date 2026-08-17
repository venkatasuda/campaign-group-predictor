"""Data drift detection.

Why this module exists
----------------------
The architecture promises PSI-based drift monitoring. A monitoring claim is worthless
without a **reference**: you cannot detect that today's traffic has moved unless the
training distribution was captured at the time the model was fitted. This module captures
that reference, stores it inside the model artifact, and scores live data against it.

Population Stability Index
--------------------------
PSI compares two distributions over the same bins::

    PSI = sum over bins of  (actual% - expected%) * ln(actual% / expected%)

Conventional reading, widely used in credit risk and adopted here:

===========  ==========================================================
PSI          Interpretation
===========  ==========================================================
< 0.10       No significant shift - no action
0.10 - 0.25  Moderate shift - investigate
>= 0.25      Major shift - retraining is likely required
===========  ==========================================================

Quantile bins are taken from the *reference* data so the expected distribution is uniform
by construction, which makes PSI values comparable across features of different scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

#: Conventional PSI decision thresholds.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


@dataclass(frozen=True)
class FeatureReference:
    """Reference distribution for one feature, captured at training time."""

    name: str
    bin_edges: list[float]
    expected_fractions: list[float]
    mean: float
    std: float
    missing_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "bin_edges": self.bin_edges,
            "expected_fractions": self.expected_fractions,
            "mean": self.mean,
            "std": self.std,
            "missing_rate": self.missing_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureReference:
        """Rebuild from a serialised representation."""
        return cls(**payload)


@dataclass(frozen=True)
class DriftReference:
    """Everything needed to detect drift against the training data.

    Stored inside the model artifact so the reference travels with the model it belongs
    to. A reference from a different model version would give meaningless PSI values.
    """

    n_rows: int
    n_bins: int
    features: dict[str, FeatureReference] = field(default_factory=dict)
    target_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "n_rows": self.n_rows,
            "n_bins": self.n_bins,
            "features": {name: ref.to_dict() for name, ref in self.features.items()},
            "target_distribution": self.target_distribution,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DriftReference:
        """Rebuild from a serialised representation."""
        return cls(
            n_rows=payload["n_rows"],
            n_bins=payload["n_bins"],
            features={
                name: FeatureReference.from_dict(ref)
                for name, ref in payload.get("features", {}).items()
            },
            target_distribution=payload.get("target_distribution", {}),
        )


def build_reference(
    features: pd.DataFrame,
    target: pd.Series | None = None,
    n_bins: int = 10,
) -> DriftReference:
    """Capture the training distribution so drift can be measured later.

    Call this on the **training split**, at the moment the champion is fitted, and store
    the result in the model artifact.
    """
    if features.empty:
        raise ValueError("Cannot build a drift reference from an empty frame.")

    references: dict[str, FeatureReference] = {}

    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            continue

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(observed, quantiles))
        if edges.size < 2:
            # Constant feature: a single bin, and PSI will always be 0.
            edges = np.array([observed.iloc[0] - 0.5, observed.iloc[0] + 0.5])

        # Open the outer edges so unseen extremes fall into the end bins rather than
        # being dropped.
        edges[0], edges[-1] = -np.inf, np.inf

        counts, _ = np.histogram(observed, bins=edges)
        fractions = counts / counts.sum()

        references[str(column)] = FeatureReference(
            name=str(column),
            bin_edges=[float(edge) for edge in edges],
            expected_fractions=[float(value) for value in fractions],
            mean=round(float(observed.mean()), 6),
            std=round(float(observed.std()), 6),
            missing_rate=round(float(values.isna().mean()), 6),
        )

    target_distribution: dict[str, float] = {}
    if target is not None and len(target):
        shares = pd.Series(target).value_counts(normalize=True)
        target_distribution = {str(key): round(float(value), 6) for key, value in shares.items()}

    logger.info(
        "Captured drift reference over %d features from %d training rows.",
        len(references),
        len(features),
    )
    return DriftReference(
        n_rows=int(len(features)),
        n_bins=n_bins,
        features=references,
        target_distribution=target_distribution,
    )


def population_stability_index(
    reference: FeatureReference,
    values: pd.Series | np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """PSI of ``values`` against a captured reference distribution.

    ``epsilon`` floors empty bins so the logarithm stays finite - without it a single
    empty bin would return infinity and drown out the real signal.
    """
    observed = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if observed.empty:
        return 0.0

    edges = np.array(reference.bin_edges, dtype=float)
    counts, _ = np.histogram(observed, bins=edges)
    actual = counts / max(counts.sum(), 1)
    expected = np.array(reference.expected_fractions, dtype=float)

    actual = np.clip(actual, epsilon, None)
    expected = np.clip(expected, epsilon, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def detect_drift(
    reference: DriftReference,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Score live data against the training reference, feature by feature.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature and sorted worst-first, with ``psi``, ``severity``
        (``none`` / ``moderate`` / ``major``), the reference and current means, and the
        change in missing rate.
    """
    rows: list[dict[str, Any]] = []

    for name, feature_reference in reference.features.items():
        if name not in features.columns:
            continue

        values = pd.to_numeric(features[name], errors="coerce")
        psi = population_stability_index(feature_reference, values)
        severity = "major" if psi >= PSI_MAJOR else "moderate" if psi >= PSI_MODERATE else "none"

        rows.append(
            {
                "feature": name,
                "psi": round(psi, 6),
                "severity": severity,
                "reference_mean": feature_reference.mean,
                "current_mean": round(float(values.mean(skipna=True)), 6),
                "reference_missing_rate": feature_reference.missing_rate,
                "current_missing_rate": round(float(values.isna().mean()), 6),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "psi",
                "severity",
                "reference_mean",
                "current_mean",
                "reference_missing_rate",
                "current_missing_rate",
            ]
        )

    return pd.DataFrame(rows).set_index("feature").sort_values("psi", ascending=False)


def drift_summary(reference: DriftReference, features: pd.DataFrame) -> dict[str, Any]:
    """Condense :func:`detect_drift` into an alertable summary.

    ``action`` is the recommendation a monitoring job would act on:
    ``none``, ``investigate`` or ``retrain``.
    """
    report = detect_drift(reference, features)
    if report.empty:
        return {
            "n_features_checked": 0,
            "n_moderate": 0,
            "n_major": 0,
            "max_psi": 0.0,
            "worst_features": [],
            "action": "none",
        }

    n_major = int((report["severity"] == "major").sum())
    n_moderate = int((report["severity"] == "moderate").sum())
    action = "retrain" if n_major else "investigate" if n_moderate else "none"

    summary = {
        "n_features_checked": int(len(report)),
        "n_moderate": n_moderate,
        "n_major": n_major,
        "max_psi": float(report["psi"].max()),
        "worst_features": report.head(5).index.tolist(),
        "action": action,
    }

    if action != "none":
        logger.warning(
            "Drift detected: %d major, %d moderate (max PSI %.4f). Recommended action: %s.",
            n_major,
            n_moderate,
            summary["max_psi"],
            action,
        )
    return summary


__all__ = [
    "PSI_MAJOR",
    "PSI_MODERATE",
    "DriftReference",
    "FeatureReference",
    "build_reference",
    "detect_drift",
    "drift_summary",
    "population_stability_index",
]
