"""Probability calibration and calibration quality measurement.

Why this matters here
---------------------
The service does not just return a label - it returns probabilities, and the
cost-sensitive decision rule in :mod:`src.decision` multiplies those probabilities by
euro amounts. If the model says 61% and is right only 45% of the time, every expected-
value calculation downstream is wrong, and the recommendation can flip.

Gradient-boosted trees are usually *not* well calibrated out of the box - boosting
pushes probabilities toward the extremes. So calibration is not optional polish here,
it is a correctness requirement for the decision layer.

Method
------
Isotonic regression is the default: the empirical literature finds it the most
consistent improver of probability estimates, whereas Platt/sigmoid scaling helps some
models and actively harms others. Isotonic can overfit on small samples, so ``sigmoid``
remains available and the module reports the metrics needed to choose between them.

Calibration must be fitted on data the model has never seen. Two supported protocols:

``prefit``
    You already fitted the model on the training split; pass a *separate* calibration
    split. Cheapest and easiest to reason about.
``cv``
    Cross-validated calibration that refits the estimator internally. Uses the data more
    efficiently when you cannot afford a third split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from src.constants import CLASS_LABELS
from src.logging_config import get_logger

logger = get_logger(__name__)

CalibrationMethod = Literal["isotonic", "sigmoid"]


@dataclass(frozen=True)
class CalibrationReport:
    """Quality of a model's probability estimates."""

    n_samples: int
    log_loss: float
    brier_multiclass: float
    expected_calibration_error: float
    max_calibration_error: float
    per_class_brier: dict[str, float]
    reliability_curve: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "n_samples": self.n_samples,
            "log_loss": self.log_loss,
            "brier_multiclass": self.brier_multiclass,
            "expected_calibration_error": self.expected_calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "per_class_brier": self.per_class_brier,
            "reliability_curve": self.reliability_curve,
        }

    def summary(self) -> str:
        """One-line verdict for the report."""
        verdict = (
            "well calibrated"
            if self.expected_calibration_error < 0.05
            else "acceptable" if self.expected_calibration_error < 0.10 else "POORLY CALIBRATED"
        )
        return (
            f"ECE={self.expected_calibration_error:.4f} ({verdict}), "
            f"Brier={self.brier_multiclass:.4f}, log-loss={self.log_loss:.4f} "
            f"on {self.n_samples} samples."
        )


def calibration_report(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray | list[int] | None = None,
    n_bins: int = 10,
) -> CalibrationReport:
    """Measure how trustworthy a set of predicted probabilities is.

    Metrics
    -------
    ``log_loss``
        Proper scoring rule; punishes confident mistakes hardest.
    ``brier_multiclass``
        Mean squared error between the one-hot truth and the probability vector.
        Decomposes into calibration + refinement, and is the standard companion to ECE.
    ``expected_calibration_error``
        Top-label ECE: bin predictions by confidence, then average
        ``|accuracy(bin) - mean confidence(bin)|`` weighted by bin size. 0 is perfect.
    ``max_calibration_error``
        The worst single bin - catches a model that is fine on average but badly
        overconfident in the high-confidence region, which is exactly the region where
        an automated decision would be taken without human review.
    ``reliability_curve``
        Per-bin confidence vs. accuracy, ready to plot.
    """
    truth = np.asarray(y_true).ravel().astype(int)
    proba = np.asarray(probabilities, dtype=float)

    if proba.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {proba.shape}")

    # Some boosters emit rows that sum to 1 only within floating-point tolerance, which
    # makes log_loss warn. Renormalise so the scoring rules are computed on a genuine
    # probability distribution.
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(
        proba, row_sums, out=np.full_like(proba, 1 / proba.shape[1]), where=row_sums > 0
    )
    if len(truth) != len(proba):
        raise ValueError("y_true and probabilities must have the same number of rows")
    if len(truth) == 0:
        raise ValueError("Cannot assess calibration on an empty sample.")

    class_order = np.asarray(classes if classes is not None else sorted(CLASS_LABELS)).astype(int)

    # --- proper scoring rules -------------------------------------------------
    ll = float(log_loss(truth, proba, labels=list(class_order)))

    one_hot = np.zeros_like(proba)
    for column, class_value in enumerate(class_order):
        one_hot[:, column] = (truth == class_value).astype(float)
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    per_class_brier = {
        CLASS_LABELS[int(class_value)]: round(
            float(brier_score_loss((truth == class_value).astype(int), proba[:, column])), 6
        )
        for column, class_value in enumerate(class_order)
    }

    # --- top-label calibration ------------------------------------------------
    confidence = proba.max(axis=1)
    predicted = class_order[proba.argmax(axis=1)]
    correct = (predicted == truth).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    curve: list[dict[str, float]] = []
    ece, mce = 0.0, 0.0

    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue

        bin_confidence = float(confidence[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        gap = abs(bin_accuracy - bin_confidence)

        ece += (count / len(truth)) * gap
        mce = max(mce, gap)
        curve.append(
            {
                "bin_lower": round(float(lower), 4),
                "bin_upper": round(float(upper), 4),
                "count": count,
                "mean_confidence": round(bin_confidence, 4),
                "accuracy": round(bin_accuracy, 4),
                "gap": round(gap, 4),
            }
        )

    return CalibrationReport(
        n_samples=int(len(truth)),
        log_loss=round(ll, 6),
        brier_multiclass=round(brier, 6),
        expected_calibration_error=round(float(ece), 6),
        max_calibration_error=round(float(mce), 6),
        per_class_brier=per_class_brier,
        reliability_curve=curve,
    )


def calibrate_pipeline(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    method: CalibrationMethod = "isotonic",
    prefit: bool = True,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Wrap a fitted pipeline in a calibration layer.

    Parameters
    ----------
    pipeline:
        The estimator to calibrate. When ``prefit`` is True this must already be fitted,
        and ``features``/``target`` must be a *held-out calibration split* the model has
        never seen. Calibrating on training data produces a confidently wrong model.
    method:
        ``"isotonic"`` (default) or ``"sigmoid"``. Compare both with
        :func:`calibration_report` and keep the better one - do not assume.
    prefit:
        ``True`` uses the already-fitted estimator; ``False`` refits it inside a
        cross-validated calibration loop.

    Returns
    -------
    CalibratedClassifierCV
        A drop-in replacement exposing ``predict`` and ``predict_proba``, so it can be
        serialised into the same artifact and served by ``SklearnPipelinePredictor``
        without any API change.
    """
    if prefit:
        calibrated = CalibratedClassifierCV(
            _freeze(pipeline), method=method, cv=_prefit_cv(pipeline)
        )
    else:
        calibrated = CalibratedClassifierCV(pipeline, method=method, cv=cv)

    calibrated.fit(features, target)
    logger.info(
        "Calibrated the pipeline with %s regression on %d held-out rows.",
        method,
        len(features),
    )
    return calibrated


def _freeze(estimator: Any) -> Any:
    """Return an estimator that scikit-learn will not refit.

    scikit-learn >= 1.6 provides ``FrozenEstimator``; earlier versions express the same
    intent through ``cv="prefit"``. Supporting both keeps the code working across the
    version boundary instead of pinning the project to one release.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return FrozenEstimator(estimator)
    except ImportError:  # pragma: no cover - version dependent
        return estimator


def _prefit_cv(estimator: Any) -> Any:
    """Return the ``cv`` argument matching the freezing strategy used by :func:`_freeze`."""
    del estimator
    try:
        import sklearn.frozen  # noqa: F401

        return None  # FrozenEstimator handles it; cv is ignored for a frozen estimator
    except ImportError:  # pragma: no cover - version dependent
        return "prefit"


def compare_calibration_methods(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    eval_features: pd.DataFrame,
    eval_target: pd.Series,
) -> dict[str, Any]:
    """Fit both calibration methods and report which one to keep.

    Returns a dict with an ``uncalibrated`` baseline, an entry per method, and
    ``recommended`` - the option with the lowest expected calibration error. Include this
    table in the report: "I compared and chose" beats "I applied isotonic because a blog
    post said so".

    .. warning::
       ``eval_features``/``eval_target`` **must not be the final test set.** Choosing a
       calibration method by comparing calibration error on the test set makes that choice
       a fitted parameter and biases every figure reported afterwards - the same
       contamination as selecting features or a decision threshold there.

       They must also differ from ``features``/``target``: a calibrator scored on the data
       it was fitted to always looks well calibrated. The correct arrangement is three
       disjoint sets - fit the model, fit and select the calibrator, then report once.
       :func:`src.training.train.train` halves its calibration split to achieve this.
    """
    results: dict[str, Any] = {
        "uncalibrated": calibration_report(
            eval_target, pipeline.predict_proba(eval_features), _classes(pipeline)
        ).to_dict()
    }

    best_method, best_ece = "uncalibrated", results["uncalibrated"]["expected_calibration_error"]

    for method in ("isotonic", "sigmoid"):
        try:
            calibrated = calibrate_pipeline(pipeline, features, target, method=method)
            report = calibration_report(
                eval_target, calibrated.predict_proba(eval_features), _classes(calibrated)
            )
        except Exception as error:  # noqa: BLE001 - a failing method must not stop the run
            logger.warning("Calibration method '%s' failed: %s", method, error)
            results[method] = None
            continue

        results[method] = report.to_dict()
        if report.expected_calibration_error < best_ece:
            best_method, best_ece = method, report.expected_calibration_error

    results["recommended"] = best_method
    logger.info("Recommended calibration: %s (ECE=%.4f).", best_method, best_ece)
    return results


def _classes(estimator: Any) -> np.ndarray:
    """Best-effort extraction of the class order from any fitted estimator."""
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = sorted(CLASS_LABELS)
    return np.asarray(classes).astype(int)


__all__ = [
    "CalibrationMethod",
    "CalibrationReport",
    "calibrate_pipeline",
    "calibration_report",
    "compare_calibration_methods",
]
