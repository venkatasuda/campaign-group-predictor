"""Presentation helpers for ``01_case_study_analysis.ipynb`` and ``02_optional_diagnostics.ipynb``.

Why this module exists
----------------------
A notebook has two jobs: run an analysis, and explain it. Code that *computes* serves the
first; code that *narrates* serves the second. Mixing them means a reader has to reconstruct
a forty-line cell in their head before they can reach the one sentence that matters.

Everything here is display logic: assembling a frame, formatting it, drawing a chart.
Nothing here makes a modelling decision. The decisions live in ``src/`` where they are unit
tested and where the API can reach them; this module exists so the notebook can call one
line and spend its space on the finding instead of the plumbing.

Why ``notebooks/`` rather than ``src/``
--------------------------------------
Three reasons, in order of how much they matter:

1. ``src/`` is the deployed package. Presentation helpers are not part of the product, and
   putting them there would imply the API depends on matplotlib and on table styling.
2. Coverage is measured on ``src/`` against a ``fail_under`` floor. Adding several hundred
   presentational lines there would either drop the project under its own gate or force
   tests that assert the colour of a heatmap.
3. ``.gcloudignore`` excludes ``notebooks/``, so none of this reaches the serving image.

Import from the notebook as::

    from notebooks.analysis_support import feature_contract, data_quality_report

which resolves because the setup cell puts the project root on ``sys.path``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.figure import Figure

from src.constants import (
    BASE_FEATURES,
    CLASS_LABELS,
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
    TARGET_COLUMN,
)

__all__ = [
    # section 1 - data quality
    "assert_column_contract", "data_quality_report", "dataset_facts", "feature_contract", "missing_values_report",

    # section 2 - outcome distribution
    "outcome_shares", "outcome_table", "plot_campaign_outcomes",

    # section 3 - post-campaign columns
    "leakage_association", "leakage_cv_comparison",

    # section 4 - exploratory
    "plot_feature_distributions", "redundancy_report", "show_pairwise_signal", "show_time_ordering",

    #section 5 - symmetry
    "position_bias_table","show_swap_validation","show_symmetry_diagnosis",

    # section 6 - modelling
    "champion_verdict","show_confusion", "split_summary", "style_leaderboard",

    # section 7 - calibration
    "apply_calibration_rule","calibration_table","plot_reliability",

    # section 8 - explainability
    "plot_horizontal_importance","show_shap_importance",

    # section 8.2 - minimal model
    "minimal_model_comparison","minimal_model_symmetry",

    # section 9 - business lift
    "cost_sensitivity_table","invariance_table","lift_summary","plot_strategies","strategy_table",

    # section 8.1 - decision-support diagnostics
    "decline_rule_summary","gate_summary","plot_feature_sweep","plot_learning_curve","plot_operating_points",
]

#: One palette for the whole notebook. Red is reserved for "unprofitable / nothing worked",
#: so the same colour never means two things across figures.
COLOUR_NEITHER = "#d1495b"
COLOUR_GROUP_1 = "#3da846"
COLOUR_GROUP_2 = "#3d58a1"
COLOUR_NEUTRAL = "#3d6fa8"


# --------------------------------------------------------------------------- section 1
def feature_contract() -> pd.DataFrame:
    """Describe the column blocks and which of them may be used at prediction time.

    Built from ``src.constants`` rather than typed out, so the table cannot drift from the
    definitions the training pipeline and the API schema actually enforce. If a feature is
    added to a block, this table updates itself; a hand-written version would quietly become
    a lie.
    """
    rows = [
        (
            "group 1, pre-campaign",
            f"{GROUP_1_FEATURES[0]} … {GROUP_1_FEATURES[-1]}",
            len(GROUP_1_FEATURES),
            "yes",
        ),
        (
            "group 2, pre-campaign",
            f"{GROUP_2_FEATURES[0]} … {GROUP_2_FEATURES[-1]}",
            len(GROUP_2_FEATURES),
            "yes",
        ),
        (
            "comparison, pre-campaign",
            f"{COMPARISON_FEATURES[0]} … {COMPARISON_FEATURES[-1]}",
            len(COMPARISON_FEATURES),
            "yes",
        ),
        (
            "post-campaign",
            ", ".join(LEAKAGE_FEATURES),
            len(LEAKAGE_FEATURES),
            "NO — recorded after the campaign ran",
        ),
        ("target", TARGET_COLUMN, 1, "label"),
    ]
    return pd.DataFrame(
        rows, columns=["block", "columns", "count", "usable at prediction time"]
    ).set_index("block")


def assert_column_contract(df: pd.DataFrame) -> None:
    """Raise if the loaded file does not match the columns the brief specifies.

    This is the one check in section 1 that is an *assertion* rather than an observation.
    Every downstream number assumes the 67 usable features, the 3 post-campaign columns and
    the target are all present and named as expected. If they are not, the right outcome is
    a loud failure at the top of the notebook rather than a subtly wrong model forty cells
    later.

    Raises
    ------
    ValueError
        If any expected column is absent, listing exactly which.
    """
    expected = set(BASE_FEATURES) | set(LEAKAGE_FEATURES) | {TARGET_COLUMN}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset does not match the brief: {len(missing)} column(s) missing "
            f"({', '.join(missing)}). Every downstream result assumes this contract."
        )


def dataset_facts(df: pd.DataFrame) -> dict[str, Any]:
    """Return the scalars section 1 contributes to ``findings.json``.

    Separated from the display tables so the notebook records results with one explicit
    ``RESULTS.update(...)`` rather than scattering assignments between rendering calls -
    which is how a value gets computed twice and reported inconsistently.
    """
    return {
        "n_campaigns": int(len(df)),
        "n_features_usable": len(BASE_FEATURES),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_rate": float(df.isna().sum().sum() / df.size),
    }


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise shape, contract conformance, nulls and duplicates in one table.

    Two duplicate counts are reported, not one, because they mean different things. Fully
    duplicated rows are a data-handling artefact. Rows duplicated *on the features only* -
    identical inputs carrying different labels - would instead put a ceiling on achievable
    accuracy no model can pass, so the two are worth distinguishing even when both are zero.
    """
    expected = set(BASE_FEATURES) | set(LEAKAGE_FEATURES) | {TARGET_COLUMN}
    actual = set(df.columns)
    missing_values = df.isna().sum()

    rows = {
        "shape (rows × columns)": f"{len(df):,} × {df.shape[1]}",
        "columns missing vs. brief": ", ".join(sorted(expected - actual)) or "none",
        "columns extra vs. brief": ", ".join(sorted(actual - expected)) or "none",
        "columns containing nulls": int((missing_values > 0).sum()),
        "overall null rate": f"{missing_values.sum() / df.size:.4%}",
        "fully duplicated rows": int(df.duplicated().sum()),
        "rows duplicated on features only": int(df.duplicated(subset=BASE_FEATURES).sum()),
    }
    return pd.DataFrame({"value": list(rows.values())}, index=list(rows))


def missing_values_report(df: pd.DataFrame) -> Any:
    """Per-column missingness, or an explicit note when there is none.

    Returns a styled frame when values are missing and a plain one-line frame otherwise.
    Rendering "no missing values" as a statement rather than as an empty table matters: an
    empty output reads as *the cell did not run*, which is a different claim entirely.
    """
    missing = df.isna().sum()
    nulls = missing[missing > 0].sort_values(ascending=False)

    if not len(nulls):
        return pd.DataFrame({"note": ["No missing values in any column."]})

    return (
        nulls.to_frame("missing")
        .assign(share=lambda d: d["missing"] / len(df))
        .style.format({"share": "{:.2%}"})
        .background_gradient(subset=["share"], cmap="Reds")
    )


# --------------------------------------------------------------------------- section 2


def outcome_shares(distribution: dict[str, Any]) -> dict[str, float]:
    """Extract just the percentages, for ``findings.json``."""
    return {key: value["percentage"] for key, value in distribution["by_class"].items()}


def outcome_table(distribution: dict[str, Any]) -> Any:
    """Outcome counts and shares, bar-formatted."""
    frame = (
        pd.DataFrame(distribution["by_class"])
        .T[["description", "count", "percentage"]]
        .rename(columns={"description": "outcome", "percentage": "share of campaigns (%)"})
    )
    return frame.style.format({"share of campaigns (%)": "{:.2f}"}).bar(
        subset=["share of campaigns (%)"], color=COLOUR_GROUP_1
    )


def plot_campaign_outcomes(distribution: dict[str, Any]) -> Figure:
    """Two views of the same distribution: the three-way split, and run-vs-don't.

    The second panel is not a restatement of the first. Collapsing classes 1 and 2 answers a
    question the three-way chart obscures - *should this campaign have run at all?* - which
    is the only decision where spend can be avoided entirely rather than redirected.
    """
    shares = [distribution["by_class"][CLASS_LABELS[i]]["percentage"] for i in sorted(CLASS_LABELS)]
    labels = ["Neither profitable", "Group 1 most profitable", "Group 2 most profitable"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bars = axes[0].bar(labels, shares, color=[COLOUR_NEITHER, COLOUR_GROUP_1, COLOUR_GROUP_2])
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set_ylabel("share of campaigns (%)")
    axes[0].set_title("Campaign outcomes")
    axes[0].set_ylim(0, max(shares) * 1.25)
    axes[0].tick_params(axis="x", labelrotation=12)

    worth_running = 100 - shares[0]
    axes[1].pie(
        [worth_running, shares[0]],
        labels=[f"Worth running\n{worth_running:.1f}%", f"Wasted budget\n{shares[0]:.1f}%"],
        colors=[COLOUR_NEUTRAL, COLOUR_NEITHER],
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    axes[1].set_title("Was the campaign worth running at all?")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- section 3


def leakage_association(df: pd.DataFrame) -> Any:
    """Correlation and per-class means for the post-campaign columns.

    Class-conditional means are shown alongside the correlation because correlation alone
    would miss a non-monotonic relationship - a column could separate class 0 sharply while
    correlating near zero with a 0/1/2 label whose ordering is not meaningful.
    """
    present = [c for c in LEAKAGE_FEATURES if c in df.columns]
    return (
        pd.DataFrame(
            {
                "correlation with target": [df[c].corr(df[TARGET_COLUMN]) for c in present],
                "mean | class 0": [df.loc[df[TARGET_COLUMN] == 0, c].mean() for c in present],
                "mean | class 1": [df.loc[df[TARGET_COLUMN] == 1, c].mean() for c in present],
                "mean | class 2": [df.loc[df[TARGET_COLUMN] == 2, c].mean() for c in present],
            },
            index=present,
        )
        .style.format("{:.4f}")
        .background_gradient(subset=["correlation with target"], cmap="RdBu_r", vmin=-1, vmax=1)
    )


class LeakageComparison(NamedTuple):
    """Result of the with/without post-campaign-columns experiment."""

    table: Any
    inflation_f1: float


def leakage_cv_comparison(df: pd.DataFrame, random_state: int) -> LeakageComparison:
    """Fit the same model with and without the post-campaign columns, on identical folds.

    Identical folds and an identical estimator are the point: any difference in score is
    then attributable to the three columns and nothing else.

    Uses a fast HistGradientBoosting configuration rather than the champion. This measures
    the *columns*, not the model, and a cheaper estimator answers that question at a
    fraction of the runtime.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from src.training.pipeline import build_pipeline, candidate_models, split_features_target

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model = candidate_models(random_state, fast=True)["hist_gradient_boosting"]

    x_leaky, y = split_features_target(df, drop_leakage=False)
    x_clean, _ = split_features_target(df, drop_leakage=True)

    leaky = cross_val_score(build_pipeline(model), x_leaky, y, cv=cv, scoring="f1_macro")
    clean = cross_val_score(build_pipeline(model), x_clean, y, cv=cv, scoring="f1_macro")

    table = pd.DataFrame(
        {
            "features": [x_leaky.shape[1], x_clean.shape[1]],
            "CV macro F1": [leaky.mean(), clean.mean()],
            "std": [leaky.std(), clean.std()],
        },
        index=["with post-campaign columns", "pre-campaign only (used from here)"],
    ).style.format({"CV macro F1": "{:.4f}", "std": "{:.4f}"})

    return LeakageComparison(table, round(float(leaky.mean() - clean.mean()), 4))


# --------------------------------------------------------------------------- section 4

_SAMPLE_COLUMNS = ["g1_1", "g1_2", "g2_1", "g2_2", "c_1", "c_2", "c_3", "c_4"]


def plot_feature_distributions(df: pd.DataFrame, columns: list[str] | None = None) -> Figure:
    """Histograms for a representative sample of columns.

    A sample, not all 67: the purpose is to check for pathologies a summary table would hide
    - hard bounds, spikes at a sentinel value, obvious multimodality - and eight panels show
    that as well as sixty-seven would, while remaining readable.
    """
    columns = columns or _SAMPLE_COLUMNS
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes.ravel(), columns, strict=False):
        sns.histplot(df[col].dropna(), ax=ax, bins=40, kde=True, color=COLOUR_NEUTRAL)
        ax.set_title(col)
        # Per-panel labels are cleared and replaced with one shared pair below. Repeating
        # "feature value" eight times is noise; omitting it entirely - which this function
        # previously did - leaves a reader unable to say what either axis measures.
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel("feature value (units are anonymised and differ per column)", y=-0.02)
    fig.supylabel("number of campaigns")
    fig.suptitle(
        "Representative feature distributions\n"
        "8 of 67 columns; the curve is a smoothed density, not a fitted model",
        y=1.04,
    )
    fig.tight_layout()
    return fig


class TimeOrdering(NamedTuple):
    """The split decision, and the evidence behind it."""

    strategy: str
    order_by: str | None
    candidates: list[str]


def show_time_ordering(x: pd.DataFrame) -> TimeOrdering:
    """Test whether any column encodes a time ordering, and pick the split accordingly.

    This runs *before* the split rather than after, because it decides which split is
    honest. If a column orders campaigns in time, a random split trains on the future and
    tests on the past - and the resulting score describes nothing anyone can deploy.

    Displays the evidence and the decision, and returns the decision for later cells.
    """
    from src.training.splits import detect_time_ordering

    evidence = detect_time_ordering(x)
    candidates = [e.column for e in evidence if e.looks_like_time]
    strategy = "temporal" if candidates else "stratified"

    display(
        pd.DataFrame([e.to_dict() for e in evidence[:8]])
        .set_index("column")
        .style.format({"spearman_with_row_order": "{:.3f}", "monotonic_fraction": "{:.3f}"})
    )
    display(
        pd.DataFrame(
            {"decision": [candidates or "none detected", strategy]},
            index=["time-ordering candidates", "split strategy adopted"],
        )
    )

    return TimeOrdering(strategy, candidates[0] if candidates else None, candidates)


def redundancy_report(x: pd.DataFrame, top_n: int = 10) -> tuple[Any, int]:
    """How much of the comparison block is reconstructible from the group blocks.

    Returns the styled top-N table and the count of redundant features. The count matters
    more than the table: it says whether the ``c_`` block is derived bookkeeping or carries
    information the group columns do not have.
    """
    from src.training.splits import comparison_redundancy_report

    redundancy = comparison_redundancy_report(x)
    styled = (
        redundancy.head(top_n)
        .style.format({"r2": "{:.4f}", "max_abs_corr_with_diff": "{:.3f}"})
        .background_gradient(subset=["r2"], cmap="Oranges", vmin=0, vmax=1)
    )
    return styled, int(redundancy["is_redundant"].sum())


class PairwiseSignal(NamedTuple):
    """Strength of the explicit ``g1_i - g2_i`` contrasts."""

    figure: Figure
    strongest: dict[str, Any]


def show_pairwise_signal(x: pd.DataFrame, y: pd.Series) -> PairwiseSignal:
    """Correlate each explicit pairwise difference with the target.

    The label is a comparison, so differences are the natural representation - and a tree
    needs many axis-aligned splits to approximate one subtraction. The question is whether
    the differences carry enough signal to be worth constructing. Displays the top five and
    returns the strongest for the results file.
    """
    diffs = pd.DataFrame({f"diff_{i}": x[f"g1_{i}"] - x[f"g2_{i}"] for i in range(1, 21)})
    signal = diffs.corrwith(y).sort_values(key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    signal.plot.bar(ax=axes[0], color=COLOUR_NEUTRAL)
    axes[0].set_title("Does each $g1_i - g2_i$ difference predict the outcome?")
    axes[0].set_xlabel("difference feature  (diff_i = g1_i − g2_i)")
    axes[0].set_ylabel("Pearson r with the target\n(0 = no relationship, ±1 = perfect)")
    axes[0].axhline(0, color="grey", linewidth=0.8)

    # The second panel guards against reading the first as six independent signals: if the
    # strongest differences are correlated with each other, they are one finding, not six.
    sns.heatmap(
        diffs[signal.index[:6]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1]
    )
    axes[1].set_title("...and are the strongest ones telling us different things?")
    axes[1].set_xlabel("each cell = correlation between two differences, NOT with the target")
    fig.tight_layout()

    display(signal.head(5).to_frame("correlation with target").style.format("{:.4f}"))

    return PairwiseSignal(
        fig,
        {"feature": signal.index[0], "correlation": round(float(signal.iloc[0]), 4)},
    )


# --------------------------------------------------------------------------- section 5


def show_symmetry_diagnosis(x: pd.DataFrame) -> list[str]:
    """Diagnose which comparison columns are direction-dependent, and return them.

    The columns are anonymised, so the *type* of each comparison feature has to be inferred
    from its distribution rather than read from a name. Three types behave differently under
    a mirror: a signed difference must be negated, a symmetric feature left alone, and a
    ratio **inverted** - negating a strictly positive ratio produces values that could never
    occur.
    """
    from src.symmetry import diagnose_comparison_symmetry, suggested_negate_columns

    display(
        diagnose_comparison_symmetry(x)
        .head(8)
        .style.format(
            {
                "correlation": "{:.3f}",
                "abs_correlation": "{:.3f}",
                "mean": "{:.3f}",
                "skew": "{:.3f}",
            }
        )
    )
    return suggested_negate_columns(x)


class SwapVerdict(NamedTuple):
    """Whether mirroring is valid, and why not when it is not."""

    augment: bool
    mirror_well_defined: bool
    positions_exchangeable: bool
    failures: list[str]


def show_swap_validation(x: pd.DataFrame, negate_columns: list[str]) -> SwapVerdict:
    """Decide whether symmetry augmentation is legitimate. Two conditions, both required.

    **1. Is the mirror well defined?** After negation, do the comparison columns still follow
    their original distribution? This is a check on the *operation*.

    **2. Are the positions exchangeable?** Are ``g1_`` and ``g2_`` drawn from the same
    distribution at all? This is a check on the *premise*, and it is the more fundamental of
    the two: a perfectly implemented mirror of a non-exchangeable pair still fabricates rows
    that never occur.

    An earlier version of this notebook tested only the first condition. That is a real bug
    rather than an omission - condition 1 is sample-dependent and *passes* under some split
    seeds, so augmentation would have been enabled on a dataset whose positions are provably
    not exchangeable. Both conditions are now required, which matches what the training code
    enforces.
    """
    from src.symmetry import validate_group_exchangeability, validate_swap

    swap_check = validate_swap(x, negate_columns)
    broken = swap_check.loc[~swap_check["distribution_preserved"]].index.tolist()
    suspicious = swap_check.loc[
        swap_check["negated"] & swap_check["strictly_positive"]
    ].index.tolist()
    mirror_ok = not (broken or suspicious)

    # `variable` is the index of this report, not a column - indexing it as a column raises.
    exchange = validate_group_exchangeability(x)
    not_exchangeable = exchange.loc[~exchange["exchangeable"]].index.tolist()
    positions_ok = not not_exchangeable

    augment = mirror_ok and positions_ok

    display(swap_check.head(8).style.format({"ks_statistic": "{:.4f}", "p_value": "{:.4f}"}))
    display(
        pd.DataFrame(
            {
                "result": [
                    broken or "none",
                    suspicious or "none",
                    (
                        f"{len(not_exchangeable)} of {len(exchange)} differ significantly"
                        if not_exchangeable
                        else "positions are exchangeable"
                    ),
                    (
                        "mirror is valid — augmentation enabled"
                        if augment
                        else "augmentation DISABLED"
                    ),
                ]
            },
            index=[
                "columns whose distribution changed",
                "negated but strictly positive",
                "paired variables not exchangeable",
                "verdict",
            ],
        )
    )

    return SwapVerdict(augment, mirror_ok, positions_ok, broken + suspicious + not_exchangeable)


def position_bias_table(y: pd.Series) -> tuple[pd.DataFrame, float]:
    """How much more often group 1 wins than group 2.

    This is the quantity the mirror would have destroyed, which is why it is reported
    immediately after the decision not to mirror.
    """
    bias = float((y == 1).mean() - (y == 2).mean())
    table = pd.DataFrame(
        {"value": [f"{(y == 1).mean():.2%}", f"{(y == 2).mean():.2%}", f"{bias:+.2%}"]},
        index=["P(group 1 wins)", "P(group 2 wins)", "position bias in the labels"],
    )
    return table, round(bias, 4)


# --------------------------------------------------------------------------- section 6
# Note on what is NOT moved here. The three-way split, the candidate loop and the champion
# rule stay in the notebook. They are the methodology a reviewer came to check, and hiding
# them behind a function call would defeat the point of the notebook. Only the formatting
# leaves. The rule is: decisions stay visible, presentation moves out.
def split_summary(
    n_train: int, n_fit: int, n_calib: int, n_test: int, augmented: bool
) -> pd.DataFrame:
    """Row counts for the three-way split, with each split's purpose stated."""
    return pd.DataFrame(
        {
            "rows": [n_train, n_fit, n_calib, n_test],
            "purpose": [
                "training campaigns",
                "after symmetry augmentation" if augmented else "augmentation disabled",
                "calibration (held out; calibrator, thresholds and gate selected here)",
                "held-out confirmation set; not used for champion selection",
            ],
        },
        index=["train", "fit", "calibration", "test"],
    )


def style_leaderboard(leaderboard: pd.DataFrame) -> Any:
    """Format the development leaderboard.

    Deliberately shows ``CV std`` next to ``CV accuracy``. A ranked column invites the eye to
    read first place as a winner; the spread beside it is what says whether that ranking
    means anything on this dataset.
    """
    return leaderboard.style.format(
        {
            "CV accuracy": "{:.4f}",
            "CV std": "{:.4f}",
            "CV macro F1": "{:.4f}",
            "CV balanced acc": "{:.4f}",
        }
    ).background_gradient(subset=["CV accuracy"], cmap="Greens")


def champion_verdict(leaderboard: pd.DataFrame, champion: str) -> pd.DataFrame:
    """State the winner *and* whether the win is real.

    Reporting the gap against the combined fold spread is the whole point. A champion table
    that shows only the winner implies a separation the data does not support, and on this
    dataset the honest verdict is that the top models are not distinguishable.
    """
    gap = leaderboard["CV accuracy"].iloc[0] - leaderboard["CV accuracy"].iloc[1]
    combined_std = leaderboard["CV std"].iloc[0] + leaderboard["CV std"].iloc[1]

    return pd.DataFrame(
        {
            "value": [
                champion,
                f"{leaderboard['CV accuracy'].iloc[0]:.4f}",
                f"{gap:.4f}",
                f"{combined_std:.4f}",
                (
                    "meaningful"
                    if gap > combined_std
                    else "within noise — the top models are not separable"
                ),
            ]
        },
        index=[
            "champion (CV accuracy, no override)",
            "its CV accuracy",
            "gap to runner-up",
            "combined CV std",
            "verdict",
        ],
    )


def show_confusion(y_true: Any, preds: Any, champion: str) -> tuple[Figure, pd.DataFrame]:
    """Confusion matrix in counts and row-normalised form, plus the per-class report.

    Both panels, not one. Counts show where the volume is; row-normalised shows recall per
    class, and it is the only view in which this model's weakness on class 0 is legible
    rather than hidden behind a large, easy class 1.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report

    names = ["neither", "group 1", "group 2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, preds, display_labels=names, ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"{champion} — counts")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        preds,
        display_labels=names,
        ax=axes[1],
        colorbar=False,
        cmap="Blues",
        normalize="true",
    )
    axes[1].set_title("Row-normalised — recall per class")
    fig.tight_layout()

    report = pd.DataFrame(
        classification_report(y_true, preds, target_names=names, output_dict=True, zero_division=0)
    ).T
    return fig, report


# --------------------------------------------------------------------------- section 7
def calibration_table(comparison: dict[str, Any]) -> Any:
    """Calibration metrics for each candidate method, measured on the selection half."""
    return (
        pd.DataFrame(
            {
                method: {
                    "expected calibration error": v["expected_calibration_error"],
                    "max calibration error": v["max_calibration_error"],
                    "Brier score": v["brier_multiclass"],
                    "log loss": v["log_loss"],
                }
                for method, v in comparison.items()
                if isinstance(v, dict)
            }
        )
        .T.style.format("{:.4f}")
        .highlight_min(subset=["expected calibration error"], color="#315a33")
    )


class CalibrationDecision(NamedTuple):
    """Whether calibration is applied, and the model that ships either way."""

    applied: bool
    method: str
    model: Any
    verdict: pd.DataFrame


def apply_calibration_rule(
    comparison: dict[str, Any],
    champion: Any,
    x_cal_fit: pd.DataFrame,
    y_cal_fit: pd.Series,
    min_relative_improvement: float = 0.10,
) -> CalibrationDecision:
    """Apply calibration only if it clears a stated margin.

    "Whichever option scores lowest" is not a decision rule - on a selection split of a few
    hundred rows, ECE has a standard error comparable to the differences being compared, so
    *something* always wins and the rule always fires. A minimum relative improvement is
    required instead.

    The margin is a judgement about the size of the selection sample, stated up front rather
    than discovered by trying values until the answer looked right.
    """
    from src.calibration import calibrate_pipeline

    recommended = comparison["recommended"]
    uncalibrated_ece = comparison["uncalibrated"]["expected_calibration_error"]
    best = comparison.get(recommended)

    candidate_ece = (
        best["expected_calibration_error"] if isinstance(best, dict) else uncalibrated_ece
    )
    required = uncalibrated_ece * (1 - min_relative_improvement)
    applied = recommended in ("isotonic", "sigmoid") and candidate_ece < required

    model = (
        calibrate_pipeline(champion, x_cal_fit, y_cal_fit, method=recommended)
        if applied
        else champion
    )
    relative = (uncalibrated_ece - candidate_ece) / uncalibrated_ece if uncalibrated_ece else 0.0

    verdict = pd.DataFrame(
        {
            "value": [
                recommended,
                f"{uncalibrated_ece:.4f}",
                f"{candidate_ece:.4f}",
                f"{relative:.1%}",
                f"{min_relative_improvement:.0%}",
                "APPLIED" if applied else "REJECTED — improvement is within noise",
            ]
        },
        index=[
            "lowest-ECE method on the selection half",
            "ECE uncalibrated",
            "ECE calibrated",
            "relative improvement",
            "improvement required",
            "decision",
        ],
    )

    return CalibrationDecision(applied, recommended if applied else "uncalibrated", model, verdict)


def plot_reliability(
    y_true: Any, champion: Any, x_test: pd.DataFrame, decision: CalibrationDecision
) -> Figure:
    """Reliability curve for the shipped model, and the calibrated one only if it shipped.

    Plotting a calibrated curve that was *rejected* would contradict the decision above it -
    a reader would see two lines and conclude calibration is in the serving path. The second
    series is therefore gated on ``decision.applied``, not on which method scored best.
    """
    from src.calibration import calibration_report

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

    label = "uncalibrated" if decision.applied else "uncalibrated (shipped)"
    series = [(label, champion.predict_proba(x_test), COLOUR_NEITHER)]
    if decision.applied:
        series.append((decision.method, decision.model.predict_proba(x_test), "#3f9d59"))

    for name, proba, colour in series:
        rep = calibration_report(y_true.to_numpy(), proba)
        curve = pd.DataFrame(rep.reliability_curve)
        if len(curve):
            ax.plot(
                curve["mean_confidence"],
                curve["accuracy"],
                "o-",
                color=colour,
                label=f"{name} (ECE {rep.expected_calibration_error:.3f})",
            )

    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(
        "Reliability curve"
        if decision.applied
        else "Reliability curve — calibration rejected on the selection split"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- section 8
def plot_horizontal_importance(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    xlabel: str,
    error_column: str | None = None,
    top_n: int = 20,
) -> Figure:
    """Horizontal bar chart of feature importance, largest at the top.

    Error bars are drawn when a spread column is supplied, and that is not decoration: on
    this dataset several importances are smaller than their own standard deviation, and a
    bare bar chart would present noise as a ranking.
    """
    top = frame.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.32 * len(top))))
    ax.barh(
        top.index,
        top[value_column],
        xerr=top[error_column] if error_column else None,
        color=COLOUR_NEUTRAL,
    )
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def show_shap_importance(
    pipeline: Any, features: pd.DataFrame, max_samples: int = 300, top_n: int = 15
) -> Figure | None:
    """Global SHAP importance over the engineered feature space.

    Complements permutation importance rather than duplicating it. Permutation importance
    answers *"how much does the score drop if this column is destroyed?"*. SHAP answers
    *"how much did this feature move this prediction?"* - additive per-prediction
    attributions that also work for a single campaign, which is what a campaign manager
    asking "why this group?" actually needs.

    They disagree in an informative way: permutation importance splits credit between
    correlated copies, and 15 of the 27 comparison features are reconstructible from the
    group blocks, so a genuinely strong feature can look weak. Agreement is evidence;
    disagreement points at collinearity rather than at error.

    Computed on the calibration split, not on test - a ranking derived from test labels is
    selection on test the moment anything downstream uses it.
    """
    from src.explainability import shap_importance_report

    try:
        shap_top = shap_importance_report(pipeline, features, max_samples=max_samples, top_n=top_n)
    except ImportError as error:
        display(
            pd.DataFrame(
                {"note": [f"SHAP not installed ({error}); permutation importance is the evidence."]}
            )
        )
        return None

    fig = plot_horizontal_importance(
        shap_top,
        "mean_abs_shap",
        f"SHAP importance — top {len(shap_top)} engineered features",
        "mean |SHAP value| (averaged over the three classes)",
        top_n=top_n,
    )
    display(shap_top.style.format({"mean_abs_shap": "{:.5f}"}))
    return fig


# ------------------------------------------------------------------------- section 8.1
def gate_summary(gate: dict[str, Any]) -> pd.DataFrame:
    """The confidence threshold, and the rule that chose it.

    The rule is displayed beside the number because a threshold without its selection rule
    is unfalsifiable - a reader cannot tell whether 0.60 was reasoned or tuned until the
    answer looked good.
    """
    return pd.DataFrame(
        {
            "value": [
                gate["threshold"],
                f"{gate['coverage']:.1%}",
                f"{gate['accuracy']:.4f}",
                gate["rule"],
            ]
        },
        index=[
            "threshold (chosen on calibration)",
            "coverage (calibration)",
            "accuracy (calibration)",
            "selection rule",
        ],
    )


def decline_rule_summary(
    rule: dict[str, Any], metrics: dict[str, Any], n_declined: int
) -> pd.DataFrame:
    """Outcome of the "decline if P(class 0) >= tau" experiment on the test set."""
    return pd.DataFrame(
        {
            "value": [
                rule["threshold"] if rule["threshold"] is not None else "none — argmax retained",
                rule["rule"],
                f"{metrics['accuracy']:.4f}",
                f"{metrics['per_class']['0']['recall']:.3f}",
                n_declined,
            ]
        },
        index=[
            "frozen threshold",
            "selection rule",
            "test accuracy",
            "test class-0 recall",
            "campaigns declined on test",
        ],
    )


def plot_feature_sweep(sweep: pd.DataFrame, baseline: float) -> Figure:
    """CV accuracy against the number of features retained.

    Log x-axis because the counts are geometric; on a linear axis the informative left-hand
    end is compressed into nothing.

    Error bars matter more here than anywhere else in the notebook. The middle of this curve
    moves by several points between adjacent counts, and without the spread a reader would
    take that shape for a trend rather than for noise.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(
        sweep.index,
        sweep["cv_mean"],
        yerr=sweep["cv_std"],
        marker="o",
        capsize=4,
        color=COLOUR_NEUTRAL,
    )
    ax.axhline(baseline, ls="--", c=COLOUR_NEITHER, label="always target group 1")
    ax.set_xscale("log")
    ax.set_xlabel("features used (top-k, ranked on the held-out calibration split)")
    ax.set_ylabel("CV accuracy")
    ax.set_title("How many features are actually needed?")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_learning_curve(curve: pd.DataFrame, baseline: float) -> Figure:
    """CV accuracy against training-set size, read against the rule the model must beat."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        curve["n_train"],
        curve["cv_mean"],
        yerr=curve["cv_std"],
        marker="o",
        capsize=4,
        color=COLOUR_NEUTRAL,
    )
    ax.axhline(baseline, ls="--", c=COLOUR_NEITHER, label=f"always target group 1 ({baseline:.1%})")
    ax.set_xlabel("training campaigns")
    ax.set_ylabel("CV accuracy")
    ax.set_title("Learning curve — does more data help?")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_operating_points(ops: pd.DataFrame, baseline: float) -> Figure:
    """Coverage and conditional accuracy against the confidence threshold.

    Twin axes because the two quantities trade against each other on a shared x-axis: as the
    threshold rises, accuracy on decided campaigns goes up and the share decided goes down.
    Plotting them apart would hide that the question is where the two cross.
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(ops.index, ops["coverage"], "o-", color=COLOUR_NEUTRAL, label="coverage")
    ax1.set_xlabel("minimum confidence to decide automatically")
    ax1.set_ylabel("coverage", color=COLOUR_NEUTRAL)

    ax2 = ax1.twinx()
    ax2.plot(
        ops.index, ops["accuracy_covered"], "s-", color="#3f9d59", label="accuracy when decided"
    )
    ax2.axhline(baseline, ls="--", c=COLOUR_NEITHER)
    ax2.set_ylabel("accuracy on decided campaigns", color="#3f9d59")

    ax1.set_title("Automate or escalate — coverage vs. accuracy")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------- section 8.2
class MinimalModel(NamedTuple):
    """One-feature versus all-features, scored on the test set."""

    table: pd.DataFrame
    single_models: dict[str, Any]
    top_feature: str


def minimal_model_comparison(
    top_feature: str,
    champion_name: str,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    champion_test_accuracy: float,
) -> MinimalModel:
    """Fit each candidate on one feature and on all 67, and score both on test.

    Candidates are identified by model key rather than by a hand-written label, so the label
    in the output cannot disagree with the model that produced the row. An earlier version
    mapped the label ``"mlp"`` onto whatever ``CHAMPION`` happened to be; when the champion
    changed, the label did not follow and every row was misattributed while the numbers
    stayed correct. A wrong label on a right number is the harder bug to catch.

    Asserts that the champion's all-67 row reproduces the headline test accuracy. If it does
    not, this function is not fitting what the report claims it is, and everything below is
    meaningless - so it fails loudly rather than producing a plausible table.
    """
    from sklearn.base import clone

    from src.training.evaluation import classification_metrics, estimate_business_lift
    from src.training.pipeline import build_pipeline, candidate_models

    keys = list(dict.fromkeys(["xgboost", champion_name]))  # dedupe if the champion IS xgboost
    zoo = candidate_models(random_state)

    rows: list[dict[str, Any]] = []
    single_models: dict[str, Any] = {}

    for model_key in keys:
        if model_key not in zoo:
            continue
        for feature_set, name in [
            ([top_feature], f"{top_feature} alone"),
            (list(x_fit.columns), f"all {x_fit.shape[1]}"),
        ]:
            # clone() is load-bearing, not defensive. build_pipeline() inserts the estimator
            # object it is handed without copying it, so reusing zoo[model_key] across both
            # feature sets would put ONE RandomForestClassifier in both pipelines - and
            # fitting the all-features pipeline would silently refit the same object,
            # destroying the single-feature fit that is stored below. The metrics would still
            # be right (predictions are taken immediately after each fit), but the retained
            # model would be the wrong one, which is exactly the bug the symmetry check hits.
            pipe = build_pipeline(clone(zoo[model_key]))
            pipe.fit(x_fit[feature_set], y_fit)
            preds = pipe.predict(x_test[feature_set])

            metrics = classification_metrics(y_test.to_numpy(), preds)
            lift = estimate_business_lift(y_test.to_numpy(), preds)
            rows.append(
                {
                    "model": model_key,
                    "features": name,
                    "n": len(feature_set),
                    "test accuracy": metrics["accuracy"],
                    "class-0 recall": metrics["per_class"]["0"]["recall"],
                    "class-1 recall": metrics["per_class"]["1"]["recall"],
                    "class-2 recall": metrics["per_class"]["2"]["recall"],
                    "lift (pp)": lift.absolute_lift_pp,
                }
            )
            if len(feature_set) == 1:
                single_models[model_key] = pipe

    table = pd.DataFrame(rows).set_index(["model", "features"])

    all_features_label = f"all {x_fit.shape[1]}"
    champion_all = table.loc[(champion_name, all_features_label), "test accuracy"]
    assert abs(champion_all - champion_test_accuracy) < 1e-4, (
        f"{champion_name} on {all_features_label} scored {champion_all:.4f}, but the "
        f"champion's reported test accuracy is {champion_test_accuracy:.4f} - these must agree."
    )

    display(
        table.style.format(
            {
                "test accuracy": "{:.4f}",
                "class-0 recall": "{:.3f}",
                "class-1 recall": "{:.3f}",
                "class-2 recall": "{:.3f}",
                "lift (pp)": "{:+.2f}",
            }
        ).background_gradient(subset=["test accuracy"], cmap="Greens")
    )

    return MinimalModel(table, single_models, top_feature)


def minimal_model_symmetry(
    minimal: MinimalModel,
    x_test: pd.DataFrame,
    negate_columns: list[str],
    model_key: str,
) -> float:
    """Can a one-feature model respond to a group swap at all?

    If the feature was not flagged as direction-dependent, the mirror leaves the model's only
    input unchanged, so it *cannot* flip 1 to 2. A high violation rate under those conditions
    means the diagnostic in section 5 missed a direction-dependent feature - which is a
    finding about the diagnostic, not about the model.

    ``model_key`` is required, deliberately. An earlier version defaulted to
    ``single_models.get("xgboost") or ...``, which silently reported a number for a model
    that is not the deployed one while every other figure in the analysis described the
    champion. Passing it explicitly makes the choice visible in the notebook, where a
    reviewer can see which model the reported rate belongs to.
    """
    from src.symmetry import measure_invariance

    top = minimal.top_feature
    if model_key not in minimal.single_models:
        raise KeyError(
            f"No single-feature model was fitted for '{model_key}'. "
            f"Available: {sorted(minimal.single_models)}"
        )
    model = minimal.single_models[model_key]

    invariance = measure_invariance(
        lambda frame: model.predict(frame[[top]]), x_test, negate_columns
    )

    display(
        pd.DataFrame(
            {
                "value": [
                    top,
                    f"{invariance.violation_rate:.2%}",
                    f"{invariance.position_bias:+.2%}",
                    "yes" if top in negate_columns else "no — the input is identical after a swap",
                ]
            },
            index=[
                "feature used",
                "predictions that changed under a group swap",
                "position bias, P(predict 1) - P(predict 2)",
                "is it direction-dependent?",
            ],
        )
    )
    return invariance.violation_rate


# --------------------------------------------------------------------------- section 9
def strategy_table(lift: Any, champion_name: str) -> Any:
    """Success rate of every naive strategy alongside the model.

    "Run no campaigns at all" is included because it is the baseline that is easiest to
    forget and can be the strongest: if most campaigns were unprofitable, doing nothing beats
    every targeting rule, and measuring lift against anything weaker overstates the model.
    """
    strategies = pd.DataFrame(
        {
            "success rate": [
                lift.random_choice_success_rate,
                lift.always_group_1_success_rate,
                lift.always_group_2_success_rate,
                lift.always_decline_success_rate,
                lift.model_success_rate,
            ]
        },
        index=[
            "random choice between groups",
            "always target group 1",
            "always target group 2",
            "run no campaigns at all",
            f"MODEL ({champion_name})",
        ],
    )
    return strategies


def lift_summary(lift: Any, ci: dict[str, Any], n_resamples: int) -> pd.DataFrame:
    """Headline lift with its interval and the avoided-waste count."""
    return pd.DataFrame(
        {
            "value": [
                lift.best_naive_strategy,
                f"{lift.best_naive_success_rate:.2%}",
                f"{lift.model_success_rate:.2%}",
                f"{lift.absolute_lift_pp:+.2f} pp",
                f"{ci['ci_lower_pp']:+.2f} to {ci['ci_upper_pp']:+.2f} pp",
                "yes" if ci["significantly_positive"] else "no — the interval includes zero",
                f"{lift.wasted_campaigns_avoided} ({lift.wasted_campaign_avoidance_rate:.1%})",
            ]
        },
        index=[
            "best naive strategy",
            "its success rate",
            "model success rate",
            "absolute lift",
            f"95% bootstrap CI ({n_resamples} resamples)",
            "statistically above zero",
            "unprofitable campaigns correctly declined",
        ],
    )


def plot_strategies(strategies: pd.DataFrame) -> Figure:
    """Bar chart of every strategy, with the model highlighted."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    colours = ["#9aa5b1"] * (len(strategies) - 1) + ["#3f9d59"]
    bars = ax.bar(range(len(strategies)), strategies["success rate"] * 100, color=colours)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([i.replace(" ", "\n", 1) for i in strategies.index], fontsize=9)
    ax.set_ylabel("campaign success rate (%)")
    ax.set_title("Campaign success rate by targeting strategy")
    ax.set_ylim(0, strategies["success rate"].max() * 122)
    fig.tight_layout()
    return fig


def cost_sensitivity_table(
    y_true: Any, proba: Any, scenarios: dict[str, tuple[float, float, float]]
) -> Any:
    """Decision-layer performance across several cost assumptions.

    The dataset contains no monetary values, so a single expected-ROI figure cannot honestly
    be derived from it. Reporting a range across stated ratios says what is actually known:
    the *direction* is robust, the magnitude depends on business inputs nobody has supplied
    yet. Confirming the real spend-to-margin ratio is the cheapest improvement available.
    """
    from src.decision import CostMatrix, DecisionPolicy, evaluate_policy

    rows = []
    for label, (spend, profit, weight) in scenarios.items():
        result = evaluate_policy(
            y_true,
            proba,
            DecisionPolicy(
                cost_matrix=CostMatrix.from_business_parameters(spend, profit, weight),
                review_margin=0.05,
            ),
        )
        rows.append(
            {
                "scenario": label,
                "cost/campaign — argmax": result["cost_per_campaign_argmax"],
                "cost/campaign — decision rule": result["cost_per_campaign_policy"],
                "saving/campaign": result["cost_saved_vs_argmax_per_campaign"],
                "decisions differing": result["decisions_differing_from_argmax"],
                "flagged for review": result["review_rate"],
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("scenario")
        .style.format(
            {
                "cost/campaign — argmax": "{:+.4f}",
                "cost/campaign — decision rule": "{:+.4f}",
                "saving/campaign": "{:+.4f}",
                "flagged for review": "{:.1%}",
            }
        )
    )


def invariance_table(invariance: Any) -> pd.DataFrame:
    """How much of the model's behaviour is position rather than signal.

    The row labels are worded precisely because the natural paraphrase is wrong. A
    *violation* is a failure to transform as the symmetry requires - 0 stays 0, 1 becomes 2,
    2 becomes 1. For classes 1 and 2, **changing is the correct behaviour**, so "predictions
    that changed" describes the opposite of what is measured. Four documents describing this
    project carried that inversion before it was caught.
    """
    return pd.DataFrame(
        {
            "value": [
                f"{invariance.violation_rate:.2%}",
                f"{invariance.position_bias:+.2%}",
                f"{invariance.class_0_stay_rate:.2%}",
                str(invariance.per_class_violation_rate),
            ]
        },
        index=[
            "predictions that FAILED to transform correctly under a group swap",
            "position bias, P(predict 1) − P(predict 2)",
            "class-0 predictions that correctly stayed class 0",
            "failure rate per predicted class",
        ],
    )
