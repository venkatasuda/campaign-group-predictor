"""Three experiments the evidence asks for, beyond the main training run.

Run once the champion is settled:

    python scripts/advanced_experiments.py --data data/customerGroups.csv

Writes ``reports/advanced_experiments.json``. The dataset is opened read-only and is
never modified, copied or written back.

1. Nested cross-validation
   An honest generalisation estimate of the *whole procedure*, hyperparameter search
   included. Directly answers the standing criticism that a tuned model's best CV score
   is optimistic.

2. Contrast-only features
   The label is a comparison, so does the model need the raw levels of each group, or
   only the differences between them? Fitting on ``g1_i - g2_i`` alone answers it - and
   the answer bears on the symmetry-violation result, because a contrast-only model
   cannot read a group's absolute position in the same way.

3. How small can the model get?
   The feature sweep showed a handful of features matching all 67 on the *untuned*
   champion. This repeats it on the tuned one. If three features match sixty-seven, the
   deployable recommendation changes: a smaller data contract, a cheaper pipeline and a
   model a marketer can actually be talked through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Running `python scripts/advanced_experiments.py` puts *this* directory on sys.path, not
# the project root, so `import src` fails. Prepending the root makes the script work both
# as a file and as `python -m scripts.advanced_experiments`, without requiring the caller
# to set PYTHONPATH or install the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import configure_logging, get_logger  # noqa: E402
from src.training.pipeline import (  # noqa: E402
    build_pipeline,
    candidate_models,
    load_dataset,
    split_features_target,
)
from src.training.splits import make_split  # noqa: E402
from src.training.tuning import nested_cv_score  # noqa: E402

logger = get_logger(__name__)

GROUP_PAIRS = 20  # g1_1..g1_20 paired with g2_1..g2_20


def contrast_only(features: pd.DataFrame) -> pd.DataFrame:
    """Return only the pairwise contrasts ``g1_i - g2_i``.

    Each column negates when the two groups are swapped, so the representation carries
    *which group is ahead on each measure* while discarding both groups' absolute levels.
    If this matches the full feature set, the levels were adding nothing the contrasts did
    not already say - and the model is reading a comparison rather than a group profile.
    """
    columns = {}
    for index in range(1, GROUP_PAIRS + 1):
        left, right = f"g1_{index}", f"g2_{index}"
        if left in features.columns and right in features.columns:
            columns[f"diff_{index}"] = features[left] - features[right]
    return pd.DataFrame(columns, index=features.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/customerGroups.csv")
    parser.add_argument("--out", default="reports/advanced_experiments.json")
    parser.add_argument("--model", default="xgboost", help="Champion model name.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--nested-outer", type=int, default=5)
    parser.add_argument("--nested-inner", type=int, default=3)
    parser.add_argument("--nested-candidates", type=int, default=20)
    parser.add_argument(
        "--skip-nested", action="store_true", help="Skip experiment 1 (the expensive one)."
    )
    args = parser.parse_args()

    configure_logging()
    results: dict[str, Any] = {}

    frame = load_dataset(Path(args.data))
    features, target = split_features_target(frame, drop_leakage=True)
    x_train, _, y_train, _ = make_split(
        features, target, strategy="stratified", test_size=0.2, random_state=args.random_state
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    zoo = candidate_models(args.random_state)

    # -- 1. Nested cross-validation ------------------------------------------------------
    if not args.skip_nested:
        nested = []
        for name in [args.model, "random_forest"]:
            if name not in zoo:
                logger.warning("Model %s is not installed; skipping its nested run.", name)
                continue
            nested.append(
                nested_cv_score(
                    build_pipeline(zoo[name]),
                    model_name=name,
                    features=x_train,
                    target=y_train,
                    outer_splits=args.nested_outer,
                    inner_splits=args.nested_inner,
                    n_iter=args.nested_candidates,
                    random_state=args.random_state,
                )
            )
        results["nested_cv"] = nested

    # -- 2. Which block of features carries the signal? -----------------------------------
    #
    # `build_pipeline` inserts PairwiseFeatureBuilder, which derives diff_i and ratio_i
    # from any g1_i/g2_i pair it finds. That is right for production but wrong for this
    # experiment: "group levels only" would silently become levels *plus* the very
    # contrasts the experiment is trying to isolate. Each representation below is
    # therefore fitted through a bare pipeline that adds nothing.
    def bare_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", clone(zoo[args.model])),
            ]
        )

    def score(frame_in: pd.DataFrame, label: str, derive: bool = False) -> dict[str, Any]:
        estimator = build_pipeline(clone(zoo[args.model])) if derive else bare_pipeline()
        scores = cross_val_score(estimator, frame_in, y_train, cv=cv, scoring="accuracy")
        row = {
            "representation": label,
            "n_features": int(frame_in.shape[1]),
            "derived_features_added": derive,
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }
        logger.info(
            "%-38s %2d cols -> %.4f +/- %.4f",
            label,
            row["n_features"],
            row["cv_mean"],
            row["cv_std"],
        )
        return row

    contrasts = contrast_only(x_train)
    group_columns = [c for c in x_train.columns if c.startswith(("g1_", "g2_"))]
    comparison_columns = [c for c in x_train.columns if c.startswith("c_")]

    results["representation"] = [
        score(x_train, "production pipeline (67 + derived)", derive=True),
        score(x_train, "all 67, no derived features"),
        score(x_train[comparison_columns], "comparison block c_ only"),
        score(x_train[group_columns], "group levels only"),
        score(contrasts, "contrasts only (g1_i - g2_i)"),
        score(
            pd.concat([x_train[group_columns], contrasts], axis=1),
            "group levels + contrasts",
        ),
    ]

    # -- 3. How small can the model get? -------------------------------------------------
    ranking_path = Path("reports/findings.json")
    if ranking_path.exists():
        ranked = json.loads(ranking_path.read_text(encoding="utf-8")).get("top_features_clean", [])
    else:  # pragma: no cover - only when the notebook has not been run
        ranked = []

    if ranked:
        sizes = []
        for count in (1, 3, 5, 10):
            subset = [name for name in ranked[:count] if name in x_train.columns]
            if subset:
                sizes.append(score(x_train[subset], f"top {len(subset)} (clean ranking)"))
        results["minimal_feature_sets"] = sizes
    else:
        logger.warning("No clean ranking found in reports/findings.json; skipping experiment 3.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
