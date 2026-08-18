"""Build a small, deterministic model artifact so CI can test the real serving path.

Why this exists
---------------
``artifacts/*.pkl`` is gitignored - the champion is 59 MB and is produced from a confidential
dataset - so a CI runner has no model. The container job therefore booted with
``ALLOW_BASELINE_FALLBACK=true`` and proved only that the *degraded* application starts. The
path that actually runs in production - artifact loads, canary passes, ``/ready`` turns 200,
a real pipeline answers ``/predict`` - was verified by hand against the deployed revision and
by nothing on every commit.

That is the wrong thing to leave unautomated. A missing entry in ``requirements-serve.txt``,
a pickle that needs a library the serving image does not install, a preprocessing step that
fails outside the training environment: each of those passes the baseline smoke test and
fails on the first real request.

What this is not
----------------
**Not the champion, and it must never be mistaken for it.** The pipeline here is trained on
synthetic noise and its predictions are meaningless. It exists to exercise machinery, not to
be evaluated: same estimator family, same preprocessing, same 67-column contract, same
artifact keys the loader expects - and a ``model_name`` that says so out loud, so a stray
copy cannot be confused with a real run.

Determinism matters for a different reason than usual: a flaky artifact would produce a
container job that fails intermittently, and an intermittently failing smoke test gets
disabled rather than fixed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.constants import BASE_FEATURES, TARGET_COLUMN
from src.training.pipeline import build_pipeline, candidate_models

RANDOM_STATE = 7
N_ROWS = 400


def build(destination: Path) -> Path:
    rng = np.random.default_rng(RANDOM_STATE)

    data: dict[str, np.ndarray] = {column: rng.normal(size=N_ROWS) for column in BASE_FEATURES}

    # A learnable signal, so the estimator fits something rather than degenerating to a
    # constant. A constant-prediction pipeline would still serve, and would still pass a
    # smoke test - while hiding the fact that predict_proba had collapsed.
    signal = data["g1_1"] + data["g1_2"] - data["g2_1"] - data["g2_2"]
    target = np.where(signal > 0.6, 1, np.where(signal < -0.6, 2, 0))

    frame = pd.DataFrame(data)
    frame[TARGET_COLUMN] = target

    features = frame[list(BASE_FEATURES)]
    pipeline = build_pipeline(candidate_models(RANDOM_STATE, fast=True)["random_forest"])
    pipeline.fit(features, frame[TARGET_COLUMN])

    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            # Named so it cannot be mistaken for a real run, in a log line or a filename.
            "model_name": "SYNTHETIC-CI-ARTIFACT-NOT-THE-CHAMPION",
            "model_version": "0.0.0-ci",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {"note": "Trained on synthetic noise. Predictions are meaningless."},
        },
        destination,
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Required, with no default, and the default that would have been obvious is exactly the
    # one to refuse: artifacts/model.pkl is the champion. A script whose accident case
    # silently overwrites a 59 MB artifact rebuilt from confidential data - and whose output
    # looks superficially identical - should not have a convenient shorthand.
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination path. Required: there is deliberately no default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists.",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"Refusing to overwrite existing {args.out}. Pass --force if that is intended.")
        return 1

    written = build(args.out)
    size_mb = written.stat().st_size / (1024 * 1024)
    print(f"Wrote synthetic CI artifact to {written} ({size_mb:.1f} MiB).")
    print("This is NOT the champion. Do not evaluate it, deploy it, or report its metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
