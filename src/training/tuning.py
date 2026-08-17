"""Hyperparameter search.

Randomised search rather than grid search: with a fixed budget, random sampling covers
the important dimensions far better than an exhaustive grid over unimportant ones
(Bergstra & Bengio, 2012). Every budget here is deliberately modest - the marginal
return on heavy tuning is small compared with getting the evaluation protocol right.

All distributions are expressed over the ``model__`` step of the pipeline, so the search
tunes the estimator while the preprocessing stays fixed and leak-free inside each fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.logging_config import get_logger

logger = get_logger(__name__)


def search_spaces(random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Return the randomised-search distribution for each model in the zoo.

    Keys match the names produced by :func:`src.training.pipeline.candidate_models`.
    Models without an entry are simply used at their defaults.
    """
    del random_state  # kept for signature symmetry with candidate_models

    return {
        # 'penalty' is deliberately absent: scikit-learn deprecated it in 1.8 in favour
        # of 'l1_ratio', and searching over it emits a FutureWarning on modern versions
        # while breaking outright from 1.10. C alone controls regularisation strength,
        # which is the parameter that actually matters here.
        # 'saga' is excluded: on this feature scale it does not converge within a
        # reasonable iteration budget and floods the log with ConvergenceWarnings without
        # ever winning the search. 'lbfgs' converges reliably for dense L2 problems.
        "logistic_regression": {
            "model__C": loguniform(1e-3, 1e2),
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_leaf": randint(1, 12),
            "model__max_features": ["sqrt", "log2", 0.4],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": loguniform(0.01, 0.25),
            "model__max_iter": randint(150, 600),
            "model__max_leaf_nodes": randint(15, 63),
            "model__min_samples_leaf": randint(10, 60),
            "model__l2_regularization": loguniform(1e-4, 1e1),
        },
        "mlp": {
            "model__hidden_layer_sizes": [(128,), (256, 128), (512, 256), (256, 128, 64)],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-2),
            "model__batch_size": [32, 64, 128],
        },
        "catboost": {
            "model__iterations": randint(200, 900),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__depth": randint(4, 9),
            "model__l2_leaf_reg": loguniform(1.0, 20.0),
            "model__random_strength": uniform(0.0, 2.0),
        },
        "lightgbm": {
            "model__n_estimators": randint(200, 900),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__num_leaves": randint(15, 90),
            "model__min_child_samples": randint(5, 60),
            "model__reg_lambda": loguniform(1e-3, 1e1),
            "model__colsample_bytree": uniform(0.6, 0.4),
        },
        "xgboost": {
            "model__n_estimators": randint(200, 900),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(3, 10),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(1e-3, 1e1),
            "model__min_child_weight": randint(1, 10),
        },
    }


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a hyperparameter search for one model."""

    model_name: str
    tuned: bool
    best_score: float
    best_params: dict[str, Any]
    n_candidates: int
    best_score_std: float = float("nan")
    strategy: str = "random"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "model_name": self.model_name,
            "tuned": self.tuned,
            "strategy": self.strategy,
            "best_score": self.best_score,
            # Carried deliberately. The best score alone cannot answer "is this model
            # actually better than the runner-up, or is the gap inside fold-to-fold
            # noise?" - which is the only question the leaderboard exists to answer.
            # Without it the tuned leaderboard is less informative than the untuned one.
            "best_score_std": self.best_score_std,
            "best_params": {key: _jsonable(value) for key, value in self.best_params.items()},
            "n_candidates": self.n_candidates,
        }


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and tuples into JSON-friendly types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def tune_pipeline(
    pipeline: Pipeline,
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    cv: BaseCrossValidator | None = None,
    n_iter: int = 25,
    scoring: str = "accuracy",
    random_state: int = 42,
    n_jobs: int = -1,
    groups: np.ndarray | None = None,
    strategy: str = "random",
    factor: int = 3,
) -> tuple[Pipeline, TuningResult]:
    """Search the pipeline's estimator hyperparameters.

    Why randomised rather than grid
    -------------------------------
    Grid search spends its budget uniformly across every dimension, including the ones
    that do not matter. Bergstra & Bengio (2012) showed that because only a few
    hyperparameters usually drive performance, random sampling finds better
    configurations than a grid of the same cost - it never wastes trials re-testing an
    irrelevant axis at a fixed value of an important one.

    Strategies
    ----------
    ``"random"``
        :class:`~sklearn.model_selection.RandomizedSearchCV`. Every candidate is
        evaluated at full cost. Simple, and the fold-to-fold spread it reports is
        directly comparable to the untuned leaderboard.
    ``"halving"``
        :class:`~sklearn.model_selection.HalvingRandomSearchCV` - successive halving.
        All candidates start on a small subset of the training rows; after each round the
        worst are discarded and the survivors get ``factor`` times the data. Cost is
        concentrated on configurations that are still competitive, so roughly three to
        five times more candidates fit in the same wall-clock. The trade-off is that
        early rounds judge candidates on little data, which can eliminate a configuration
        that only pays off at full size.

    A caveat that matters more than the choice of optimiser
    -------------------------------------------------------
    A more effective search makes the winner's cross-validated score **more** optimistic,
    not less. Selecting the maximum over many estimates biases that maximum upward, and
    the harder the search, the larger the bias. Tuning therefore improves the *model*
    while degrading the honesty of the *number* attached to it. The defence is nested
    cross-validation, or - as here - reporting the tuned score as a development figure
    and quoting the held-out test set separately.

    Parameters
    ----------
    scoring:
        Defaults to ``accuracy``, which matches the campaign-success rate the business
        cares about. It is deliberately **not** ``f1_macro``: selecting on macro-F1 was
        found to favour a class-balanced model whose success rate fell below the naive
        "always target group 1" rule, so tuning toward it would optimise the wrong thing.
    groups:
        Group labels forwarded to the splitter. **Required when the training set has been
        symmetry-augmented**: a campaign and its mirror share a group id, which stops the
        search from validating a fold against the mirror of a row it trained on.

    Returns the best fitted pipeline and a :class:`TuningResult`. When no search space
    is defined for ``model_name``, the pipeline is fitted at its defaults and
    ``TuningResult.tuned`` is ``False`` - the caller does not need a special case.
    """
    spaces = search_spaces(random_state)
    space = spaces.get(model_name)
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if not space:
        logger.info("No search space for '%s'; fitting at default hyperparameters.", model_name)
        pipeline.fit(features, target)
        return pipeline, TuningResult(model_name, False, float("nan"), {}, 0)

    common = {
        "estimator": pipeline,
        "scoring": scoring,
        "cv": cv,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "refit": True,
        "error_score": "raise",
    }

    if strategy == "halving":
        # Experimental-flag import, required by scikit-learn to enable the estimator.
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        search = HalvingRandomSearchCV(
            param_distributions=space, n_candidates=n_iter, factor=factor, **common
        )
    elif strategy == "random":
        search = RandomizedSearchCV(param_distributions=space, n_iter=n_iter, **common)
    else:  # pragma: no cover - guarded by the CLI's choices=
        raise ValueError(f"unknown search strategy {strategy!r}; expected 'random' or 'halving'")

    # `groups` is only meaningful for a grouped splitter; passing it to a plain
    # StratifiedKFold is harmless but passing None keeps the halving path simple.
    search.fit(features, target, **({"groups": groups} if groups is not None else {}))

    # Recover the winning configuration's fold-to-fold spread. Without it the tuned
    # leaderboard cannot say whether one model genuinely beat another.
    std = float(search.cv_results_["std_test_score"][search.best_index_])

    logger.info(
        "Tuned %-22s best %s=%.4f (+/- %.4f) over %d candidates via %s search.",
        model_name,
        scoring,
        search.best_score_,
        std,
        n_iter,
        strategy,
    )
    return search.best_estimator_, TuningResult(
        model_name=model_name,
        tuned=True,
        best_score=round(float(search.best_score_), 4),
        best_params=dict(search.best_params_),
        n_candidates=n_iter,
        best_score_std=round(std, 4),
        strategy=strategy,
    )


def nested_cv_score(
    pipeline: Pipeline,
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_iter: int = 20,
    scoring: str = "accuracy",
    random_state: int = 42,
    n_jobs: int = -1,
    strategy: str = "random",
) -> dict[str, Any]:
    """Estimate generalisation performance of the **whole procedure**, search included.

    .. note::
       ``strategy`` defaults to ``"random"`` here, unlike :func:`tune_pipeline`. Successive
       halving evaluates its first rounds on very small subsets, and a subset can easily
       omit a class entirely. XGBoost requires labels to be contiguous from zero and
       raises ``Invalid classes inferred from unique values of y`` when handed ``[1, 2]``,
       which silently turns an outer fold into ``nan`` and poisons the mean. Full-data
       folds cost more but cannot produce that failure.

    The problem this solves
    -----------------------
    A tuned model's best cross-validated score is optimistic, and the optimism grows with
    the size of the search. The score is the *maximum* over many noisy estimates, and the
    maximum of noisy estimates is biased upward - part of what the winner "won" is luck on
    those particular folds. Reporting it as a generalisation estimate overstates the model.

    Nested cross-validation removes that bias by never scoring a model on data its own
    hyperparameter search has seen:

    1. The **outer** loop splits the data into ``outer_splits`` folds.
    2. For each outer fold, a complete hyperparameter search runs on the training part
       only, using its own **inner** cross-validation.
    3. The winning configuration is scored once on the held-out outer fold.
    4. The reported figure is the mean over outer folds.

    What the number means
    ---------------------
    It is **not** the score of any single model - each outer fold may select different
    hyperparameters. It estimates how well *this procedure* - "run this search on data of
    this size, then use whatever it picks" - generalises. That is the honest question when
    a model will be retrained periodically, because retraining reruns the search.

    Read the spread as well as the mean. A large spread across outer folds means the
    search is unstable: different data yields materially different models, which is a
    reason to prefer a simpler, lower-variance candidate even at slightly lower mean.

    Cost
    ----
    ``outer_splits`` complete searches. At the defaults that is five searches of twenty
    candidates each - roughly five times a single tuning run. This is why it is a separate
    entry point rather than a flag on the training path.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    space = search_spaces(random_state).get(model_name)
    if not space:
        raise ValueError(f"no search space defined for {model_name!r}; nothing to nest")

    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    common = {
        "estimator": clone(pipeline),
        "scoring": scoring,
        "cv": inner,
        "random_state": random_state,
        # The outer loop is parallelised instead of the inner search. Nesting two
        # `n_jobs=-1` levels oversubscribes every core and, on Windows, is often slower
        # than running serially.
        "n_jobs": 1,
        "refit": True,
        "error_score": "raise",
    }

    if strategy == "halving":
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        search = HalvingRandomSearchCV(param_distributions=space, n_candidates=n_iter, **common)
    else:
        search = RandomizedSearchCV(param_distributions=space, n_iter=n_iter, **common)

    logger.info(
        "Nested CV for %s: %d outer folds x (%d-fold inner search over %d candidates).",
        model_name,
        outer_splits,
        inner_splits,
        n_iter,
    )
    scores = cross_val_score(search, features, target, cv=outer, scoring=scoring, n_jobs=n_jobs)

    result = {
        "model_name": model_name,
        "nested_mean": round(float(scores.mean()), 4),
        "nested_std": round(float(scores.std()), 4),
        "outer_scores": [round(float(value), 4) for value in scores],
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "n_candidates": n_iter,
        "scoring": scoring,
        "strategy": strategy,
    }
    logger.info(
        "Nested CV %s: %s = %.4f +/- %.4f over %d outer folds %s",
        model_name,
        scoring,
        scores.mean(),
        scores.std(),
        outer_splits,
        result["outer_scores"],
    )
    return result


__all__ = ["TuningResult", "nested_cv_score", "search_spaces", "tune_pipeline"]
