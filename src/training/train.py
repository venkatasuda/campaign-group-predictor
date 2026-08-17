"""Training CLI.

Usage
-----
    # Day 1 smoke run - fast, no tuning
    python -m src.training.train --data data/customerGroups.csv --out artifacts --fast

    # Full run
    python -m src.training.train --data data/customerGroups.csv --out artifacts \
        --tune --n-iter 25 --augment-symmetry

    # Honest protocol when a time ordering exists
    python -m src.training.train --data data/customerGroups.csv --split temporal

    # Leakage demonstration for the report - never ship this model
    python -m src.training.train --data data/customerGroups.csv --keep-leakage --fast

Produces
--------
``artifacts/model.pkl``
    Champion pipeline plus metadata, loadable by ``SklearnPipelinePredictor``.
``artifacts/metrics.json``
    Outcome distribution (ML Q1), dataset diagnostics, per-model metrics, symmetry
    invariance, explainability, and business lift (ML Q3).
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src import __version__
from src.calibration import calibrate_pipeline, calibration_report
from src.decision import CostMatrix, DecisionPolicy, evaluate_policy
from src.drift import build_reference, drift_summary
from src.explainability import explain
from src.logging_config import configure_logging, get_logger
from src.symmetry import (
    augment_with_swapped_grouped,
    measure_invariance,
    suggested_negate_columns,
    validate_group_exchangeability,
    validate_swap,
)
from src.tracking import (
    end_training_run,
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    model_run,
    start_training_run,
)
from src.training.evaluation import (
    bootstrap_lift_interval,
    campaign_outcome_distribution,
    classification_metrics,
    estimate_business_lift,
)
from src.training.pipeline import (
    build_pipeline,
    candidate_models,
    dataset_fingerprint,
    load_dataset,
    split_features_target,
)
from src.training.splits import (
    comparison_redundancy_report,
    detect_time_ordering,
    make_split,
)
from src.training.tuning import tune_pipeline

logger = get_logger(__name__)

#: Model-selection criteria, all computed by cross-validation on the TRAINING split.
#:
#: 'accuracy' is the default because it *is* the campaign success rate the brief asks us
#: to improve: the share of campaigns where the chosen action matches the profitable
#: outcome. Selecting on 'f1_macro' instead is a trap on this dataset - a class-balanced
#: model maximises macro F1 by under-predicting the dominant class, which raises minority
#: recall while making the success rate *worse than the naive baseline*.
#:
#: Note what is deliberately absent: business lift. Lift is measured on the held-out test
#: set, so selecting on it would be selecting on the test set and would contaminate the
#: final estimate. Accuracy under cross-validation is the honest proxy - it ranks models
#: identically, because lift is just accuracy minus a constant baseline.
SELECTION_SCORERS: dict[str, str] = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the campaign group predictor.")
    parser.add_argument("--data", default="data/customerGroups.csv", help="Path to the CSV.")
    parser.add_argument("--out", default="artifacts", help="Output directory for artifacts.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Model and CV seed.")
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=None,
        help=(
            "Seed for the train/test split, separate from --random-state. Defaults to it "
            "when unset. Two different concerns: --random-state controls model "
            "initialisation and fold assignment, --holdout-seed decides which campaigns "
            "are never seen. Conflating them means a test set cannot be regenerated "
            "without also perturbing every model, which is exactly what is needed after "
            "a test set has been inspected too often and must be replaced."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["stratified", "temporal"],
        default="stratified",
        help="Splitting protocol. Use 'temporal' if a time ordering was detected.",
    )
    parser.add_argument(
        "--order-by",
        default=None,
        help="Column to sort by for a temporal split. Defaults to existing row order.",
    )
    parser.add_argument(
        "--select-by",
        choices=sorted(SELECTION_SCORERS),
        default="accuracy",
        help=(
            "Cross-validated metric used to pick the champion. Default 'accuracy' is the "
            "campaign success rate the brief asks us to improve. 'f1_macro' can select a "
            "class-balanced model whose success rate is worse than the naive baseline."
        ),
    )
    parser.add_argument(
        "--champion",
        default=None,
        help=(
            "Override the automatically selected champion with a named model, while still "
            "evaluating the whole zoo. Use when a model wins the selection metric but is "
            "disqualified on grounds the metric cannot express - for example a model that "
            "is accurate overall yet almost never recommends declining a campaign, and so "
            "cannot serve the 'neither group is profitable' case. The override and the "
            "automatic choice are both recorded in metrics.json."
        ),
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run a hyperparameter search over the model zoo."
    )
    parser.add_argument(
        "--n-iter", type=int, default=25, help="Search candidates sampled per model."
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "MLflow tracking URI. Defaults to a local ./mlruns directory, which needs no "
            "server. Point at a shared tracking server (or a managed endpoint) to log "
            "there instead - that is the whole migration."
        ),
    )
    parser.add_argument(
        "--search",
        choices=["random", "halving"],
        default="random",
        help=(
            "Search strategy. 'random' evaluates every candidate at full cost. 'halving' "
            "uses successive halving: candidates start on a subset of rows and only the "
            "survivors get more data, fitting roughly 3-5x more candidates in the same "
            "wall-clock at the cost of judging early rounds on less data."
        ),
    )
    parser.add_argument(
        "--augment-symmetry",
        action="store_true",
        help="Mirror the training split (group 1 <-> group 2) to enforce antisymmetry.",
    )
    parser.add_argument(
        "--negate-threshold",
        type=float,
        default=0.9,
        help="Correlation above which a c_ feature is treated as direction dependent.",
    )
    parser.add_argument(
        "--force-augment",
        action="store_true",
        help="Augment even if swap validation fails. Only with a hand-verified mirror.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model names to evaluate. Defaults to the whole zoo.",
    )
    parser.add_argument("--fast", action="store_true", help="Shrink models for a quick smoke run.")
    parser.add_argument(
        "--skip-explain", action="store_true", help="Skip permutation/SHAP importance."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate the champion's probabilities on a held-out calibration split.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method. Isotonic is the more consistent improver.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.2,
        help="Fraction of the training split reserved for calibration.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for the lift confidence interval. 0 disables.",
    )
    parser.add_argument(
        "--campaign-spend", type=float, default=1.0, help="Cost of running one campaign."
    )
    parser.add_argument(
        "--profit-if-correct",
        type=float,
        default=1.0,
        help="Net profit when the correct group is targeted.",
    )
    parser.add_argument(
        "--opportunity-weight",
        type=float,
        default=0.5,
        help="Share of profit charged for declining a campaign that would have paid off.",
    )
    parser.add_argument(
        "--review-margin",
        type=float,
        default=0.05,
        help="Expected-cost gap below which a decision is flagged for human review.",
    )
    parser.add_argument(
        "--keep-leakage",
        action="store_true",
        help="Keep post-campaign columns. For the leakage demonstration only.",
    )
    return parser.parse_args(argv)


def _diagnostics(features: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Run the dataset diagnostics that inform the modelling choices."""
    ordering = detect_time_ordering(features)
    redundancy = comparison_redundancy_report(features)

    return {
        "time_ordering_candidates": [item.to_dict() for item in ordering if item.looks_like_time][
            :10
        ],
        "time_ordering_top_evidence": [item.to_dict() for item in ordering[:5]],
        "split_strategy_used": args.split,
        "comparison_redundancy": {
            "n_redundant": int(redundancy["is_redundant"].sum()),
            "n_checked": int(len(redundancy)),
            "redundant_features": redundancy.loc[redundancy["is_redundant"]].index.tolist(),
            "top_r2": redundancy["r2"].head(5).round(6).to_dict(),
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full training routine and persist the artifacts.

    Deliberately one long linear function rather than a class or a chain of small
    helpers. A training run *is* a sequence performed once, in order, where every step
    depends on the one before it; splitting it up would scatter that order across call
    sites and hide the single most important property of the routine - that the test set
    is created at phase 2 and not touched again until phase 6. Section banners below mark
    the phases so the shape stays readable at length.

    Phases
    ------
    1. **Load and project** - read the CSV, drop the post-campaign columns, report the
       outcome distribution (ML question 1).
    2. **Split** - hold out the test set, then optionally carve a calibration split from
       what remains. Nothing after this point may look at ``x_test`` except phase 6.
    3. **Symmetry preconditions** - decide whether mirroring rows is defensible. Refuses
       to augment when the two group positions are not exchangeable.
    4. **Model zoo** - fit and cross-validate every candidate on identical folds, so the
       comparison is paired.
    5. **Champion selection** - on the cross-validated score, never on the test score.
    6. **Test evaluation** - a single pass: metrics, business lift, invariance.
    7. **Calibration and explainability** - both optional, both non-fatal.
    8. **Persist** - model artifact, drift reference and metrics, with the dataset's
       SHA-256 recorded for lineage.

    Returns
    -------
    dict
        The full metrics payload, identical to the contents of ``metrics.json``.
    """
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load and project onto the legitimate feature set -----------------------------
    frame = load_dataset(args.data)
    features, target = split_features_target(frame, drop_leakage=not args.keep_leakage)

    distribution = campaign_outcome_distribution(target)
    logger.info("Campaign outcome distribution: %s", json.dumps(distribution["by_class"], indent=2))

    diagnostics = _diagnostics(features, args)

    # -- 2. Split. After this line, x_test is untouchable until phase 6 -------------------
    holdout_seed = args.holdout_seed if args.holdout_seed is not None else args.random_state
    x_train, x_test, y_train, y_test = make_split(
        features,
        target,
        strategy=args.split,
        test_size=args.test_size,
        random_state=holdout_seed,
        order_by=args.order_by,
    )
    # Carve the calibration split off *before* symmetry augmentation: a row and its
    # mirror are not independent, so they must never straddle a split boundary.
    x_calib, y_calib = None, None
    if args.calibrate:
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_train,
            y_train,
            test_size=args.calibration_size,
            stratify=y_train,
            random_state=args.random_state,
        )

    logger.info(
        "Train rows: %d | Calibration rows: %s | Test rows: %d",
        len(x_train),
        len(x_calib) if x_calib is not None else "-",
        len(x_test),
    )

    # -- 3. Symmetry preconditions -------------------------------------------------------
    negate_columns = suggested_negate_columns(x_train, args.negate_threshold)

    # Precondition for the entire symmetry argument: are the two slots interchangeable?
    # If group 1 is systematically a different kind of group from group 2 - the incumbent,
    # the larger segment - then position carries real signal and mirroring fabricates rows
    # that never occur.
    exchangeability = validate_group_exchangeability(x_train)
    groups_exchangeable = bool(exchangeability.empty or exchangeability["exchangeable"].all())

    # Second check: a ratio-type comparison feature would be corrupted by sign flipping.
    swap_check = validate_swap(x_train, negate_columns)
    swap_ok = bool(
        swap_check.empty
        or (
            swap_check["distribution_preserved"].all()
            and not (swap_check["negated"] & swap_check["strictly_positive"]).any()
        )
    )

    groups: np.ndarray | None = None
    if args.augment_symmetry:
        if not groups_exchangeable and not args.force_augment:
            divergent = exchangeability.loc[~exchangeability["exchangeable"]].index.tolist()
            raise ValueError(
                "Symmetry augmentation refused: the group positions are NOT "
                f"exchangeable. {len(divergent)} paired variable(s) differ significantly "
                f"between the g1_ and g2_ blocks ({divergent[:8]}), which means group 1 "
                "and group 2 are systematically different kinds of group. Mirroring would "
                "fabricate rows that never occur and would destroy genuine signal.\n"
                "\nDrop --augment-symmetry. See "
                "metrics.json['diagnostics']['group_exchangeability'] for the evidence, "
                "and report it - a tested-and-rejected assumption is a finding."
            )
        if not swap_ok and not args.force_augment:
            raise ValueError(
                "Swap validation failed - mirroring changed the distribution of at least "
                "one comparison column, so the augmented rows are not realistic. Inspect "
                "the report in metrics.json['diagnostics']['swap_validation'], then either "
                "adjust --negate-threshold, drop augmentation, or pass --force-augment if "
                "you have verified the mirror by hand."
            )
        augmented = augment_with_swapped_grouped(x_train, y_train, negate_columns)
        x_train, y_train, groups = augmented.features, augmented.target, augmented.groups

    policy = DecisionPolicy(
        cost_matrix=CostMatrix.from_business_parameters(
            campaign_spend=args.campaign_spend,
            profit_if_correct=args.profit_if_correct,
            opportunity_weight=args.opportunity_weight,
        ),
        review_margin=args.review_margin,
    )

    # When the training set is mirrored, a campaign and its mirror MUST stay in the same
    # fold. StratifiedKFold would put them on opposite sides and every CV score would be
    # inflated by validating against a transformed copy of a training row.
    if groups is not None:
        cv = StratifiedGroupKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
        )
        logger.info("Using StratifiedGroupKFold: mirrored rows are kept within one fold.")
    else:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    # -- 4. Model zoo. Every candidate sees identical folds, so the comparison is paired --
    zoo = candidate_models(args.random_state, fast=args.fast)
    if args.models:
        unknown = set(args.models).difference(zoo)
        if unknown:
            raise ValueError(f"Unknown model name(s): {sorted(unknown)}. Available: {sorted(zoo)}")
        zoo = {name: zoo[name] for name in args.models}

    results: dict[str, Any] = {}
    fitted_pipelines: dict[str, Any] = {}
    best_name, best_pipeline, best_score = "", None, -np.inf

    # Experiment tracking. Degrades to a no-op when MLflow is not installed, so training
    # never fails because of the layer that observes it.
    tracker = start_training_run(
        run_name=f"train-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        tracking_uri=args.tracking_uri,
        tags={
            # The dataset hash is the lineage anchor: it answers "which data produced this
            # model?" without trusting a filename or a directory convention.
            "dataset_sha256": dataset_fingerprint(args.data),
            "split_strategy": args.split,
            "selection_metric": args.select_by,
            "tuned": str(args.tune),
        },
    )
    log_params(
        tracker,
        {
            "n_rows": len(frame),
            "n_features": x_train.shape[1],
            "test_size": args.test_size,
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
            "split": args.split,
            "select_by": args.select_by,
            "tune": args.tune,
            "search_strategy": args.search if args.tune else None,
            "n_iter": args.n_iter if args.tune else None,
            "augment_symmetry": args.augment_symmetry,
            "calibrate": args.calibrate,
            "models": ",".join(zoo),
        },
    )

    for name, model in zoo.items():
        pipeline = build_pipeline(model)

        if args.tune:
            pipeline, tuning = tune_pipeline(
                pipeline,
                model_name=name,
                features=x_train,
                target=y_train,
                cv=cv,
                n_iter=args.n_iter,
                scoring=SELECTION_SCORERS[args.select_by],
                random_state=args.random_state,
                groups=groups,
                strategy=args.search,
            )
            cv_scores = {args.select_by: tuning.best_score}
            # The winning configuration's fold-to-fold spread, not NaN: the leaderboard's
            # job is to say whether a gap between two models is real, and it cannot do
            # that from point estimates alone.
            cv_mean, cv_std = tuning.best_score, tuning.best_score_std
        else:
            # Score every criterion in one pass so the leaderboard shows all of them and
            # the selection rule is a reporting choice, not a hidden default.
            cv_result = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                scoring=SELECTION_SCORERS,
                groups=groups,
                n_jobs=None,
            )
            cv_scores = {
                key: round(float(cv_result[f"test_{key}"].mean()), 4) for key in SELECTION_SCORERS
            }
            pipeline.fit(x_train, y_train)
            tuning = None
            scores = cv_result[f"test_{args.select_by}"]
            cv_mean, cv_std = float(scores.mean()), float(scores.std())

        # DEVELOPMENT METRICS ONLY.
        #
        # `x_test` is deliberately absent from this loop. An earlier version scored every
        # candidate on the test set "for the leaderboard" while selecting on CV - which is
        # defensible in principle and indefensible in practice, because the numbers are
        # then on screen while the decision is being made. Selection bias does not require
        # intent; it only requires the figures to be visible.
        #
        # The question a leaderboard exists to answer - "were the candidates close?" - is
        # answerable from the cross-validated mean and its fold-to-fold spread, which is
        # where it belongs. Nothing here needs the test set.
        metrics: dict[str, Any] = {
            "cv_scores": cv_scores,
            "selection_metric": args.select_by,
            "cv_selection_score_mean": None if np.isnan(cv_mean) else round(cv_mean, 4),
            "cv_selection_score_std": None if np.isnan(cv_std) else round(cv_std, 4),
            "tuning": tuning.to_dict() if tuning else None,
            "evaluated_on_test": False,
        }

        results[name] = metrics
        fitted_pipelines[name] = pipeline

        # One child run per candidate, so "which models were compared in the run that
        # produced the deployed artifact?" is answerable - the question that matters when
        # a deployed model is being questioned months later.
        with model_run(tracker, name) as child:
            log_params(child, {"model": name, **(tuning.best_params if tuning else {})})
            log_metrics(child, metrics)

        logger.info(
            "%-22s CV %s=%s (+/- %s) | CV f1_macro=%s",
            name,
            args.select_by,
            metrics["cv_selection_score_mean"],
            metrics["cv_selection_score_std"],
            cv_scores.get("f1_macro", "-"),
        )

        # Selection is on the cross-validated score. There is no test score to select on.
        comparable = -np.inf if np.isnan(cv_mean) else cv_mean
        if best_pipeline is None or comparable > best_score:
            best_name, best_pipeline, best_score = name, pipeline, comparable

    if best_pipeline is None:  # pragma: no cover - defensive
        raise RuntimeError("No model could be trained.")

    # -- 5. Champion selection -----------------------------------------------------------
    # An override is permitted but never silent: both the automatic winner and the manual
    # choice are written to metrics.json, so a reader can see a judgement was made and
    # hold the report to justifying it.
    automatic_champion = best_name
    if args.champion:
        if args.champion not in results:
            raise ValueError(
                f"--champion '{args.champion}' was not evaluated. " f"Available: {sorted(results)}"
            )
        if args.champion != best_name:
            logger.warning(
                "Champion overridden: '%s' won on CV %s (%.4f) but '%s' was selected "
                "manually. Both are recorded in metrics.json - justify the override in "
                "the report.",
                best_name,
                args.select_by,
                best_score,
                args.champion,
            )
        best_name = args.champion
        best_pipeline = fitted_pipelines[best_name]
        best_score = results[best_name]["cv_selection_score_mean"] or float("nan")

    logger.info(
        "Champion locked: %s (CV %s=%.4f). The test set has not been touched yet.",
        best_name,
        args.select_by,
        best_score,
    )

    # -- 6. Test evaluation - the champion only, exactly once ----------------------------
    #
    # Everything above this line used development data. Everything below reports; nothing
    # below changes a decision. That ordering is the point: it means "the test set was
    # evaluated once" is a property of the code rather than a claim in a document.
    #
    # If you find yourself wanting a candidate's test score to justify a choice, the choice
    # belongs above this line and the evidence belongs in cross-validation.
    champion_predictions = best_pipeline.predict(x_test)
    champion_metrics = classification_metrics(y_test.to_numpy(), champion_predictions)
    champion_metrics["business_lift"] = estimate_business_lift(
        y_test.to_numpy(), champion_predictions
    ).to_dict()

    if args.bootstrap > 0:
        champion_metrics["business_lift_ci"] = bootstrap_lift_interval(
            y_test.to_numpy(),
            champion_predictions,
            n_resamples=args.bootstrap,
            random_state=args.random_state,
        )

    invariance = measure_invariance(best_pipeline.predict, x_test, negate_columns)
    champion_metrics["symmetry_invariance"] = invariance.to_dict()

    if hasattr(best_pipeline, "predict_proba"):
        probabilities = best_pipeline.predict_proba(x_test)
        champion_metrics["calibration"] = calibration_report(
            y_test.to_numpy(), probabilities, getattr(best_pipeline, "classes_", None)
        ).to_dict()
        champion_metrics["decision_policy_evaluation"] = evaluate_policy(
            y_test.to_numpy(), probabilities, policy
        )

    champion_metrics["evaluated_on_test"] = True
    results[best_name].update(champion_metrics)

    logger.info(
        "Test evaluation (%s, once): acc=%.4f | lift=%+.2f pp | class-0 recall=%.3f | inv=%.1f%%",
        best_name,
        champion_metrics["accuracy"],
        champion_metrics["business_lift"]["absolute_lift_pp"],
        champion_metrics["per_class"]["0"]["recall"],
        invariance.violation_rate * 100,
    )

    if champion_metrics["business_lift"]["absolute_lift_pp"] <= 0:
        logger.error(
            "The champion has NON-POSITIVE business lift: it picks the right action less "
            "often than the naive '%s' rule. Do not deploy this. Check whether class "
            "weighting is trading success rate for macro F1.",
            champion_metrics["business_lift"]["best_naive_strategy"],
        )

    # Explainability runs on the *uncalibrated* pipeline. Calibration wraps the estimator
    # in a CalibratedClassifierCV, which is not step-indexable, so SHAP cannot reach the
    # underlying model through it. Calibration rescales probabilities; it does not change
    # which features the model relies on, so explaining the inner pipeline is correct.
    explain_pipeline = best_pipeline

    calibration_applied = False
    if args.calibrate and x_calib is not None and hasattr(best_pipeline, "predict_proba"):
        # "Apply calibration or not" is a fitted decision, so it cannot be taken by
        # comparing calibration error on the test set - that is the same contamination as
        # selecting features or thresholds there, and it silently biases every figure
        # reported afterwards.
        #
        # The calibration split is therefore halved. One half fits the calibrator; the
        # other, which neither the model nor the calibrator has seen, decides whether the
        # calibrated version is actually better. Only then is the winner measured once on
        # the test set. Two halves are needed rather than one because a calibrator
        # evaluated on its own fitting data always looks well calibrated.
        x_cal_fit, x_cal_eval, y_cal_fit, y_cal_eval = train_test_split(
            x_calib,
            y_calib,
            test_size=0.5,
            stratify=y_calib,
            random_state=args.random_state,
        )

        before_selection = calibration_report(
            y_cal_eval.to_numpy(),
            best_pipeline.predict_proba(x_cal_eval),
            getattr(best_pipeline, "classes_", None),
        )
        try:
            calibrated = calibrate_pipeline(
                best_pipeline, x_cal_fit, y_cal_fit, method=args.calibration_method
            )
            after_selection = calibration_report(
                y_cal_eval.to_numpy(),
                calibrated.predict_proba(x_cal_eval),
                getattr(calibrated, "classes_", None),
            )
        except Exception as error:  # noqa: BLE001 - never fail training on calibration
            logger.warning("Calibration failed (%s); keeping the uncalibrated model.", error)
            after_selection, calibrated = None, None

        # "Apply if ECE improves at all" is not a decision rule, it is a coin flip with
        # extra steps. On a selection split of a few hundred rows, ECE has a standard error
        # comparable to the differences being compared, so any two options will separate by
        # *something* and the rule will always fire.
        #
        # A minimum relative improvement is required instead: calibration must reduce ECE
        # by at least this fraction to be worth adding a component to the serving path.
        # The threshold is a statistical judgement about the size of the selection sample,
        # not a tuned parameter - and it is stated here rather than discovered by trying
        # values until the answer looked right.
        #
        # This guard was added after observing the rule select a method on a margin of
        # 0.003 ECE across 530 rows. Recording *when* it was added matters: the change is
        # to the rule, justified by the sample size, and was not made by looking at which
        # option the test set preferred.
        min_relative_improvement = 0.10

        improved = False
        if after_selection is not None:
            baseline_ece = before_selection.expected_calibration_error
            candidate_ece = after_selection.expected_calibration_error
            required = baseline_ece * (1.0 - min_relative_improvement)
            improved = candidate_ece < required

        # Report the chosen model on the test set - once, after the decision is frozen.
        chosen = calibrated if improved else best_pipeline
        after = calibration_report(
            y_test.to_numpy(),
            chosen.predict_proba(x_test),
            getattr(chosen, "classes_", None),
        )
        before = calibration_report(
            y_test.to_numpy(),
            best_pipeline.predict_proba(x_test),
            getattr(best_pipeline, "classes_", None),
        )

        results[best_name]["calibration_comparison"] = {
            "method": args.calibration_method,
            "selected_on": "held-out half of the calibration split, never the test set",
            "selection_rows": int(len(x_cal_eval)),
            "selection_rule": (
                f"apply only if ECE falls by at least {min_relative_improvement:.0%} - a "
                "smaller margin is not distinguishable from noise on a split this size"
            ),
            "selection_ece_uncalibrated": before_selection.expected_calibration_error,
            "selection_ece_calibrated": (
                after_selection.expected_calibration_error if after_selection else None
            ),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "improved": bool(improved),
        }
        logger.info(
            "Calibration decision on %d held-out rows: ECE %.4f uncalibrated vs %s "
            "calibrated; %.0f%% relative improvement required -> %s.",
            len(x_cal_eval),
            before_selection.expected_calibration_error,
            (f"{after_selection.expected_calibration_error:.4f}" if after_selection else "n/a"),
            min_relative_improvement * 100,
            "apply" if improved else "keep raw probabilities",
        )
        logger.info("Calibration on test (uncalibrated): %s", before.summary())
        logger.info("Calibration on test (shipped):      %s", after.summary())

        if improved:
            best_pipeline = calibrated
            calibration_applied = True
            # Re-score the decision layer on the calibrated probabilities: the whole
            # point of calibrating is that the expected-cost arithmetic becomes valid.
            results[best_name]["decision_policy_evaluation"] = evaluate_policy(
                y_test.to_numpy(), calibrated.predict_proba(x_test), policy
            )
            results[best_name]["calibration"] = after.to_dict()

    # -- 7b. Explainability --------------------------------------------------------------
    if not args.skip_explain:
        try:
            results[best_name]["explainability"] = explain(explain_pipeline, x_test, y_test)
        except Exception as error:  # noqa: BLE001 - never fail training on explainability
            logger.warning("Explainability step failed: %s", error)

    # Capture the training distribution so drift can actually be measured later. A
    # monitoring claim without a stored reference is not a monitoring capability.
    reference = build_reference(x_train, y_train)
    holdout_drift = drift_summary(reference, x_test)
    logger.info(
        "Drift sanity check on the hold-out split: max PSI %.4f, action '%s' "
        "(should be 'none' - both splits come from the same data).",
        holdout_drift["max_psi"],
        holdout_drift["action"],
    )

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "model_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "drift_reference": reference.to_dict(),
        "metrics": {
            key: value
            for key, value in results[best_name].items()
            if key not in {"explainability", "per_class"}
        },
        "dataset_sha256": dataset_fingerprint(args.data),
        # Recorded so the serving side can detect a mismatch. A pipeline pickled by one
        # scikit-learn minor version and unpickled by another may load without error and
        # then behave differently - the failure is silent, which is what makes it worth
        # a version string in the artifact rather than a pin in a requirements file alone.
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_columns": list(features.columns),
        "negate_columns": negate_columns,
        "symmetry_augmented": bool(args.augment_symmetry),
        "calibrated": calibration_applied,
        "calibration_method": args.calibration_method if calibration_applied else None,
        "decision_policy": policy.to_dict(),
        "leakage_dropped": not args.keep_leakage,
    }
    model_path = output_dir / "model.pkl"
    joblib.dump(artifact, model_path)
    logger.info("Saved model artifact to %s", model_path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.data),
        "dataset_sha256": dataset_fingerprint(args.data),
        "n_rows": int(len(frame)),
        "n_features_used": int(features.shape[1]),
        "leakage_dropped": not args.keep_leakage,
        "holdout_seed": holdout_seed,
        # Duplicated from the model artifact, deliberately.
        #
        # The artifact already carries these, but reading them requires unpickling it -
        # which needs the very libraries whose versions you are trying to discover. That is
        # circular exactly when it matters: someone holding a model.pkl that will not load
        # and asking what would load it. metrics.json is plain text and answers without
        # executing anything.
        #
        # These four are what requirements-serve.txt pins. If they disagree with that file,
        # the container is not the environment that produced the model.
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        # Recorded so a reader can verify the protocol rather than take it on trust:
        # exactly one entry in `models` carries test metrics, and it is the champion.
        "evaluation_protocol": (
            "Candidates compared on cross-validation only. The test set was evaluated "
            "once, on the locked champion, after selection. The test set is then READ "
            "several times to report different quantities (metrics, symmetry, calibration, "
            "policy, drift) - but no choice depends on any of them. 'Selected on once' is "
            "the property that matters; 'read once' would be false."
        ),
        "symmetry_augmented": bool(args.augment_symmetry),
        "negate_columns": negate_columns,
        "tuned": bool(args.tune),
        "calibrated": calibration_applied,
        "decision_policy": policy.to_dict(),
        "holdout_drift_check": holdout_drift,
        "cv_protocol": (
            "StratifiedGroupKFold (mirrored campaigns kept within a fold)"
            if groups is not None
            else "StratifiedKFold"
        ),
        "group_exchangeability": {
            "exchangeable": groups_exchangeable,
            "n_variables_checked": int(len(exchangeability)),
            "divergent_variables": (
                exchangeability.loc[~exchangeability["exchangeable"]].index.tolist()
                if not exchangeability.empty
                else []
            ),
            "detail": (
                exchangeability.head(10).reset_index().to_dict(orient="records")
                if not exchangeability.empty
                else []
            ),
        },
        "swap_validation": {
            "passed": swap_ok,
            "n_columns_checked": int(len(swap_check)),
            "columns_with_distribution_shift": (
                swap_check.loc[~swap_check["distribution_preserved"]].index.tolist()
                if not swap_check.empty
                else []
            ),
            "negated_but_strictly_positive": (
                swap_check.loc[
                    swap_check["negated"] & swap_check["strictly_positive"]
                ].index.tolist()
                if not swap_check.empty
                else []
            ),
        },
        "diagnostics": diagnostics,
        "campaign_outcome_distribution": distribution,
        "champion_model": best_name,
        "champion_selected_automatically": automatic_champion,
        "champion_overridden": bool(args.champion and args.champion != automatic_champion),
        "selection_metric": args.select_by,
        "models": results,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)

    # Close out the tracking run: the champion's metrics at the top level (so runs are
    # comparable without opening a child), the artifacts attached, and the model itself
    # registered so a served prediction is traceable to the run that produced it.
    #
    # metrics.json is still written to disk. It travels with the artifact, is readable
    # without MLflow installed, and is diffable in version control - MLflow adds history
    # across runs, not a replacement for the file.
    if tracker is not None:
        tracker.set_tags(
            {
                "champion": best_name,
                "champion_overridden": str(
                    bool(args.champion and args.champion != automatic_champion)
                ),
                "calibrated": str(calibration_applied),
            }
        )
        log_metrics(tracker, {"champion": results[best_name]})
        log_artifact(tracker, metrics_path)
        log_model(tracker, best_pipeline)
        end_training_run(tracker)

    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    configure_logging("INFO")
    train(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
