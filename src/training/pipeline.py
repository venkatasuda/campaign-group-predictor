"""Pipeline construction and dataset loading.

Every preprocessing step lives inside the ``Pipeline`` object that gets serialised.
Serving therefore replays the exact transformations used in training, which eliminates
training/serving skew by construction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import BASE_FEATURES, CLASS_LABELS, LEAKAGE_FEATURES, TARGET_COLUMN
from src.features import PairwiseFeatureBuilder
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read ``customerGroups.csv`` and return it as a DataFrame.

    Raises
    ------
    FileNotFoundError
        With an actionable message rather than a bare stack trace. The dataset is
        confidential and therefore deliberately absent from the repository, so a missing
        file is the expected first-run state, not a defect - the error should say so.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path.resolve()}'.\n"
            "\n"
            "The dataset is confidential and is deliberately not committed to this "
            "repository. Place the provided 'customerGroups.csv' in the 'data/' "
            "directory, or point --data at its location.\n"
            f"\nExpected 71 columns: {len(BASE_FEATURES)} pre-campaign features, "
            f"{len(LEAKAGE_FEATURES)} post-campaign columns, and '{TARGET_COLUMN}'."
        )

    frame = pd.read_csv(dataset_path)
    logger.info(
        "Loaded %d rows x %d columns from %s (sha256=%s, %d bytes)",
        len(frame),
        frame.shape[1],
        dataset_path,
        dataset_fingerprint(dataset_path)[:16],
        dataset_path.stat().st_size,
    )
    return frame


def dataset_fingerprint(path: str | Path) -> str:
    """Return the SHA-256 of the dataset file.

    Serves two purposes:

    * **Lineage.** The hash is stored in the model artifact, so any prediction can be
      traced to the exact bytes it was trained on - not just to a filename that may have
      been overwritten.
    * **Integrity.** Re-running this function proves the source data is byte-identical to
      what training consumed. The pipeline only ever reads the file, but "the code looks
      read-only" is an argument; a matching hash is evidence.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_features_target(
    frame: pd.DataFrame,
    drop_leakage: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split into the 67 legitimate pre-campaign features and the target.

    Parameters
    ----------
    drop_leakage:
        When True (default) the post-campaign columns are removed. Set to False *only*
        to reproduce the leakage demonstration in the report.

    Notes
    -----
    The allowed column set depends on ``drop_leakage``. This matters: the final
    projection below keeps only whitelisted columns, so if the whitelist were always
    ``BASE_FEATURES`` the post-campaign columns would be stripped even when the caller
    asked to keep them - and the leakage demonstration would silently compare a model
    against an identical copy of itself, showing no difference where a dramatic one is
    expected.
    """
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Dataset does not contain the '{TARGET_COLUMN}' column.")

    # The target is validated before it is coerced, not after.
    #
    # `.astype(int)` is silent and lossy: it turns 1.7 into 1, NaN into a platform-dependent
    # sentinel, and "2" into 2. Any of those means the file is not what this pipeline
    # assumes, and every metric computed downstream would be arithmetically valid and
    # meaningless. A training job that inherits a corrupted label column should stop, not
    # produce a model.
    raw_target = frame[TARGET_COLUMN]
    if raw_target.isna().any():
        raise ValueError(
            f"'{TARGET_COLUMN}' contains {int(raw_target.isna().sum())} null value(s). "
            "The label cannot be imputed - a campaign with an unknown outcome is not a "
            "training example."
        )

    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    if numeric_target.isna().any():
        bad = raw_target[numeric_target.isna()].unique()[:5]
        raise ValueError(f"'{TARGET_COLUMN}' contains non-numeric value(s): {list(bad)}.")

    if not (numeric_target == numeric_target.round()).all():
        raise ValueError(
            f"'{TARGET_COLUMN}' contains fractional value(s). The label is categorical; "
            "a fraction means the column is not the target this pipeline expects."
        )

    target = numeric_target.astype(int)
    unexpected_classes = sorted(set(target.unique()) - set(CLASS_LABELS))
    if unexpected_classes:
        raise ValueError(
            f"'{TARGET_COLUMN}' contains value(s) outside {sorted(CLASS_LABELS)}: "
            f"{unexpected_classes}. 0 = neither profitable, 1 = group 1, 2 = group 2."
        )

    features = frame.drop(columns=[TARGET_COLUMN])
    allowed = list(BASE_FEATURES)

    if drop_leakage:
        present = [column for column in LEAKAGE_FEATURES if column in features.columns]
        if present:
            logger.info("Dropping post-campaign leakage columns: %s", present)
            features = features.drop(columns=present)
    else:
        allowed += list(LEAKAGE_FEATURES)
        logger.warning(
            "KEEPING post-campaign columns %s. This is valid only for the leakage "
            "demonstration - a model trained on these cannot be deployed, because the "
            "values do not exist when a campaign is being planned.",
            LEAKAGE_FEATURES,
        )

    # A missing expected feature is a hard failure, not a warning.
    #
    # The previous behaviour selected whatever intersection happened to exist, so a file
    # missing ten columns trained a ten-columns-smaller model that reported its accuracy
    # without complaint - and the artifact would then fail at serving time, where the API
    # schema *does* require all 67. Failing here converts a production-time surprise into a
    # training-time error message naming the columns.
    missing = [column for column in allowed if column not in features.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} expected feature column(s): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. "
            "The API schema requires all of them at prediction time, so a model trained "
            "without them could not be served."
        )

    # Unexpected columns stay a warning rather than an error. An extra column is ignorable -
    # it is projected away below and cannot reach the model - whereas a missing one changes
    # what the model is. Treating both as fatal would refuse a file that is a superset of
    # the contract, which is a legitimate thing for an upstream system to send.
    extra = [column for column in features.columns if column not in allowed]
    if extra:
        logger.warning(
            "Ignoring %d unexpected column(s) not in the contract: %s", len(extra), extra[:10]
        )

    return features[allowed], target


def build_pipeline(model: Any, add_ratios: bool = True) -> Pipeline:
    """Assemble the full preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("pairwise", PairwiseFeatureBuilder(add_differences=True, add_ratios=add_ratios)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def candidate_models(random_state: int = 42, fast: bool = False) -> dict[str, Any]:
    """Return the model zoo evaluated during training.

    The zoo deliberately spans four model families so the champion is chosen on
    evidence rather than on fashion:

    ``logistic_regression``
        Linear, interpretable benchmark. If it is competitive, that is a finding worth
        reporting - it means the decision boundary is close to linear in the pairwise
        differences.
    ``random_forest``
        Bagged trees. Low-variance reference point for the boosted models.
    ``mlp``
        Neural baseline. Included because well-regularised MLPs have repeatedly been
        shown to be competitive with boosted trees on tabular data; omitting a neural
        model without testing one would be an unsupported assumption.
    ``catboost`` / ``lightgbm`` / ``xgboost`` / ``hist_gradient_boosting``
        Gradient-boosted trees. CatBoost is listed first because published comparisons
        favour it on *probability quality* (log-loss, Brier), which matters more than
        raw accuracy here: the downstream decision rule consumes probabilities.

    Optional dependencies are guarded, so the zoo silently shrinks to whatever is
    installed and training never fails because of a missing package.

    Parameters
    ----------
    fast:
        Shrink the expensive models for a quick smoke run (used by the test suite).
    """
    n_estimators = 100 if fast else 400
    boost_iters = 100 if fast else 300

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            # n_jobs=1, not -1, and this is a *serving* decision made at training time.
            #
            # `n_jobs` is a fitted attribute: it is pickled with the estimator and governs
            # inference as well as fitting. The champion runs on Cloud Run with **one vCPU**
            # and request-level concurrency, so `-1` means every concurrent request spawns
            # workers for cores that do not exist. The result is thread oversubscription and
            # latency that degrades sharply under exactly the load it was meant to handle -
            # invisible in single-request benchmarking, which is how it survived.
            #
            # The cost is training time on a multi-core laptop. That is paid once, offline,
            # by someone who can wait. Inference latency is paid on every request by someone
            # who cannot.
            n_jobs=1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=boost_iters,
            learning_rate=0.08,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128) if not fast else (32,),
            alpha=1e-3,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=100 if fast else 600,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        ),
    }

    try:  # optional dependency - primary candidate
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("catboost is not installed; the primary candidate is unavailable.")

    try:  # optional dependency
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("lightgbm is not installed; continuing without it.")

    try:  # optional dependency
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            tree_method="hist",
            random_state=random_state,
            verbosity=0,
        )
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("xgboost is not installed; continuing without it.")

    return models
