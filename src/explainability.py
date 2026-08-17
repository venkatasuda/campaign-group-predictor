"""Model explainability.

The brief asks that "any third person can understand the solution", so the model must
be able to say *why* it recommends a group, not only *which* one.

Two complementary views are provided:

``permutation_importance_report``
    Model-agnostic, always available, and measures what the model actually relies on
    (drop in score when a feature is shuffled). This is the safe default.

``shap_importance_report`` / ``shap_summary_plot``
    Local attributions that also explain individual campaigns - "group 2 is recommended
    mainly because diff_7 and c_12 favour it". Requires the optional ``shap`` package.

Both operate on the *transformed* feature space, so the engineered ``diff_i`` and
``ratio_i`` columns are explained by name rather than being hidden inside the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.logging_config import get_logger

logger = get_logger(__name__)


def transformed_feature_names(pipeline: Pipeline, features: pd.DataFrame) -> list[str]:
    """Return the column names entering the final estimator.

    Falls back to positional names if a step does not implement
    ``get_feature_names_out``.
    """
    step = getattr(pipeline, "named_steps", {}).get("pairwise")
    if step is not None and hasattr(step, "get_feature_names_out"):
        names = [str(name) for name in step.get_feature_names_out()]
        if names:
            return names
    return [str(column) for column in features.columns]


def transform_without_model(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Push ``features`` through every pipeline step except the final estimator."""
    return pipeline[:-1].transform(features)


def permutation_importance_report(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Rank raw input features by the score drop caused by shuffling them.

    Computed on the *raw* 67 columns (not the engineered ones), because that is the
    level at which a stakeholder can act. Use a held-out split, never the training data.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature, with ``importance_mean``, ``importance_std`` and ``rank``,
        sorted most-important first.
    """
    # n_jobs defaults to sequential: each parallel worker receives a copy of the feature
    # frame, and with 67 features x 10 repeats that has been observed to exhaust memory
    # and kill a notebook kernel. Pass n_jobs=-1 explicitly on a machine with headroom.
    result = permutation_importance(
        pipeline,
        features,
        target,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    report = (
        pd.DataFrame(
            {
                "feature": list(features.columns),
                "importance_mean": np.round(result.importances_mean, 6),
                "importance_std": np.round(result.importances_std, 6),
            }
        )
        .set_index("feature")
        .sort_values("importance_mean", ascending=False)
    )
    report["rank"] = range(1, len(report) + 1)

    logger.info(
        "Permutation importance computed on %d features; top 5: %s",
        len(report),
        ", ".join(report.head(5).index),
    )
    return report.head(top_n) if top_n else report


def shap_importance_report(
    pipeline: Pipeline,
    features: pd.DataFrame,
    max_samples: int = 500,
    random_state: int = 42,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Global SHAP importance over the engineered feature space.

    Mean absolute SHAP value per feature, averaged across the three classes. Sampling is
    capped by ``max_samples`` because exact SHAP is expensive; 300-500 rows is plenty
    for a stable global ranking.

    Raises
    ------
    ImportError
        If ``shap`` is not installed. Catch it and fall back to
        :func:`permutation_importance_report`.
    """
    import shap  # imported lazily: optional dependency

    sample = features.sample(n=min(max_samples, len(features)), random_state=random_state)
    transformed = transform_without_model(pipeline, sample)
    names = transformed_feature_names(pipeline, sample)
    model = pipeline[-1]

    # SHAP dispatches on what it is given. Passing the *estimator* works for tree models
    # (fast, exact TreeExplainer) but fails for an MLP with "the passed model is not
    # callable". Passing the *bound predict_proba method* instead makes SHAP fall back to
    # a model-agnostic permutation explainer, which works for any estimator.
    #
    # Trees are tried first because the exact path is orders of magnitude faster; the
    # generic path is the fallback, on a reduced background set to keep it tractable.
    try:
        explainer = shap.Explainer(model, transformed)
        shap_values = explainer(transformed)
    except Exception as tree_error:  # noqa: BLE001 - any dispatch failure is handled the same
        logger.info(
            "Exact SHAP unavailable for %s (%s); using the model-agnostic explainer.",
            type(model).__name__,
            tree_error,
        )
        background = transformed[: min(100, len(transformed))]
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(background)

    magnitudes = _mean_absolute_shap(shap_values.values)

    if magnitudes.shape[0] != len(names):  # pragma: no cover - defensive
        names = [f"feature_{index}" for index in range(magnitudes.shape[0])]

    report = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.round(magnitudes, 6)})
        .set_index("feature")
        .sort_values("mean_abs_shap", ascending=False)
    )
    report["rank"] = range(1, len(report) + 1)
    return report.head(top_n) if top_n else report


def _mean_absolute_shap(values: np.ndarray) -> np.ndarray:
    """Reduce a SHAP value array to one magnitude per feature.

    Handles both the ``(n_samples, n_features)`` binary layout and the
    ``(n_samples, n_features, n_classes)`` multiclass layout emitted by modern shap.
    """
    array = np.asarray(values)
    if array.ndim == 3:
        return np.abs(array).mean(axis=(0, 2))
    if array.ndim == 2:
        return np.abs(array).mean(axis=0)
    raise ValueError(f"Unexpected SHAP value shape: {array.shape}")


def shap_summary_plot(
    pipeline: Pipeline,
    features: pd.DataFrame,
    output_path: str | Path,
    max_samples: int = 500,
    random_state: int = 42,
    top_n: int = 20,
) -> Path:
    """Write a global SHAP importance bar chart to ``output_path``.

    A bar chart of mean absolute SHAP value per feature is used rather than a beeswarm:
    the target is multiclass, so a beeswarm would need one panel per class and is harder
    to read in a report. The bar chart answers the question a stakeholder actually asks -
    "which variables drive the recommendation?" - in one image.

    Returns the path so the caller can embed it in the report.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: no display needed in CI or a container
    import matplotlib.pyplot as plt

    report = shap_importance_report(
        pipeline, features, max_samples=max_samples, random_state=random_state, top_n=top_n
    ).iloc[
        ::-1
    ]  # reverse so the most important bar sits at the top

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, max(4.0, 0.32 * len(report))))
    axis.barh(report.index.astype(str), report["mean_abs_shap"], color="#4a7dbd")
    axis.set_xlabel("Mean |SHAP value|  (averaged over the three classes)")
    axis.set_title(f"Global feature importance - top {len(report)}")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    logger.info("SHAP importance plot written to %s", path)
    return path


def explain(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    top_n: int = 20,
) -> dict[str, Any]:
    """Produce the explainability block for the metrics report.

    Always returns permutation importance; adds SHAP when the package is available.
    Never raises because of a missing optional dependency.
    """
    block: dict[str, Any] = {
        "permutation_importance_top": permutation_importance_report(
            pipeline, features, target, top_n=top_n
        )
        .reset_index()
        .to_dict(orient="records")
    }

    try:
        block["shap_importance_top"] = (
            shap_importance_report(pipeline, features, top_n=top_n)
            .reset_index()
            .to_dict(orient="records")
        )
    except ImportError:
        logger.warning("shap is not installed; reporting permutation importance only.")
        block["shap_importance_top"] = None
    except Exception as error:  # noqa: BLE001 - explainability must never break training
        logger.warning("SHAP failed (%s); reporting permutation importance only.", error)
        block["shap_importance_top"] = None

    return block


__all__ = [
    "explain",
    "permutation_importance_report",
    "shap_importance_report",
    "shap_summary_plot",
    "transform_without_model",
    "transformed_feature_names",
]
