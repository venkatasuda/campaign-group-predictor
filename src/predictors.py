"""Predictor implementations.

Design pattern: **Strategy**. ``BasePredictor`` defines the contract; each concrete
subclass is an interchangeable strategy. The API depends only on the abstraction, so
swapping the champion model - or falling back to a baseline - requires no API changes.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.constants import (
    CLASS_DESCRIPTIONS,
    CLASS_LABELS,
    HISTORICAL_CLASS_DISTRIBUTION,
    RECOMMENDED_ACTIONS,
)
from src.exceptions import ModelArtifactError
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Immutable outcome of a single comparison."""

    predicted_class: int
    label: str
    description: str
    recommended_action: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_class(
        cls,
        predicted_class: int,
        probabilities: dict[str, float] | None = None,
    ) -> PredictionResult:
        """Build a result from a class index, filling in the business metadata.

        ``confidence`` is the predicted probability of the chosen class. When the caller
        supplies no probabilities - the estimator has no ``predict_proba`` - it falls back
        to ``1.0``, which means *"no probability information available"* and **not**
        *"certain"*. Downstream consumers that weight by confidence (the cost-sensitive
        decision layer, the automation gate) are only meaningful for estimators that do
        expose calibrated probabilities; every model in the zoo does.
        """
        probabilities = probabilities or {}
        confidence = probabilities.get(CLASS_LABELS[predicted_class], 1.0)
        return cls(
            predicted_class=predicted_class,
            label=CLASS_LABELS[predicted_class],
            description=CLASS_DESCRIPTIONS[predicted_class],
            recommended_action=RECOMMENDED_ACTIONS[predicted_class],
            confidence=round(float(confidence), 6),
            probabilities={key: round(float(value), 6) for key, value in probabilities.items()},
        )


class BasePredictor(ABC):
    """Abstract strategy every predictor must implement."""

    #: Registry key used by :class:`src.factory.ModelFactory`.
    name: str = "base"

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> list[PredictionResult]:
        """Return one :class:`PredictionResult` per row of ``frame``."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata exposed by the ``/model/info`` endpoint."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"


class MajorityClassPredictor(BasePredictor):
    """Baseline strategy that always predicts the historically most frequent class.

    Serves two purposes:

    1. It is the honest benchmark that the ML model must beat (ML question 3).
    2. It lets the API start and stay healthy when no trained artifact is available,
       which keeps CI green without committing a binary model to the repository.
    """

    name = "majority_class"

    def __init__(
        self,
        majority_class: int = 1,
        class_distribution: dict[str, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        class_distribution:
            Probabilities reported alongside every prediction. Defaults to the observed
            historical frequencies rather than a one-hot vector on ``majority_class``.

            The distinction matters. A one-hot default would make this baseline report
            ``confidence: 1.0`` on every call while being correct 46% of the time - and
            that value is not decorative. It is consumed by the cost-sensitive decision
            layer, which multiplies it by money, and by the automation gate, which uses it
            to decide what a human never sees. A baseline that overstates its certainty is
            more dangerous than one that simply guesses, because it guesses *and* suppresses
            review.
        """
        if majority_class not in CLASS_LABELS:
            raise ValueError(f"majority_class must be one of {sorted(CLASS_LABELS)}")
        self.majority_class = majority_class
        self.class_distribution = class_distribution or dict(HISTORICAL_CLASS_DISTRIBUTION)

    def predict(self, frame: pd.DataFrame) -> list[PredictionResult]:
        """Predict the majority class for every row."""
        return [
            PredictionResult.from_class(self.majority_class, self.class_distribution)
            for _ in range(len(frame))
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the baseline."""
        return {
            "predictor": self.name,
            "model_type": "baseline",
            "majority_class": self.majority_class,
            "class_distribution": self.class_distribution,
            "is_baseline": True,
        }


class SklearnPipelinePredictor(BasePredictor):
    """Production strategy wrapping a fitted scikit-learn ``Pipeline``."""

    name = "sklearn_pipeline"

    def __init__(
        self,
        pipeline: Any,
        model_name: str = "unknown",
        model_version: str = "0.0.0",
        trained_at: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(pipeline, "predict"):
            raise ModelArtifactError("Loaded artifact does not expose a 'predict' method.")
        self.pipeline = pipeline
        self.model_name = model_name
        self.model_version = model_version
        self.trained_at = trained_at or datetime.now(timezone.utc).isoformat()
        self.metrics = metrics or {}

        # Lineage, populated by `from_artifact`. Declared here so the attributes always
        # exist: a predictor constructed directly in a test has them as None rather than
        # raising AttributeError when /model/info reads them.
        self.artifact_sha256: str | None = None
        self.dataset_sha256: str | None = None
        self.sklearn_version: str | None = None
        self.holdout_seed: int | None = None

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_artifact(cls, path: str | Path) -> SklearnPipelinePredictor:
        """Load a predictor from a joblib artifact written by ``src.training.train``.

        The artifact is a dict so metadata travels with the weights. A bare pipeline
        is also accepted for convenience.
        """
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise ModelArtifactError(f"Model artifact not found at '{artifact_path}'.")

        # Fingerprint the bytes actually loaded.
        #
        # `model_version` is "1.0.0" and stays "1.0.0" across every retrain, so it cannot
        # answer the question that matters six weeks after a decision: *which artifact
        # produced this recommendation?* Two builds can carry the same version string, the
        # same model name and different weights.
        #
        # The hash is of the file on disk, computed at load time, so it describes what this
        # process is actually serving rather than what a build step recorded. Surfaced at
        # /model/info and logged at startup, which makes a prediction traceable to a
        # specific file without a model registry existing yet.
        digest = hashlib.sha256()
        with artifact_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        artifact_sha256 = digest.hexdigest()

        try:
            payload = joblib.load(artifact_path)
        except Exception as error:  # noqa: BLE001 - surfaced as a domain error
            raise ModelArtifactError(f"Failed to deserialise '{artifact_path}': {error}") from error

        if isinstance(payload, dict):
            pipeline = payload.get("pipeline")
            if pipeline is None:
                raise ModelArtifactError("Artifact dict does not contain a 'pipeline' key.")

            # A pipeline pickled by one scikit-learn minor version and unpickled by
            # another usually loads without error and may then behave differently -
            # attribute defaults change, private internals move. The failure is silent,
            # which is precisely why it is worth announcing at load time rather than
            # trusting a version range in a requirements file.
            # A MAJOR or MINOR mismatch fails closed; a patch mismatch warns.
            #
            # The previous behaviour warned on any difference and loaded anyway. That is the
            # wrong default for this failure: the container starts, passes its readiness
            # probe, serves traffic, and returns predictions that are *arithmetically valid
            # and quietly different* from the ones every number in the report describes.
            # Nobody notices, because nothing errors.
            #
            # The split is deliberate rather than blanket strictness. scikit-learn's own
            # policy is that pickles are not guaranteed across minor versions - estimator
            # attributes are added, private internals move - so 1.9 vs 1.8 is a real risk.
            # Patch releases are bug fixes to the same estimator layout, so 1.9.0 vs 1.9.1
            # is worth recording and not worth refusing: failing there would block a security
            # patch for no safety gain, which is the same reasoning that governs how
            # requirements-serve.txt is pinned.
            trained_with = payload.get("sklearn_version")
            if trained_with and trained_with != sklearn.__version__:
                trained_series = trained_with.split(".")[:2]
                running_series = sklearn.__version__.split(".")[:2]

                if trained_series != running_series:
                    raise ModelArtifactError(
                        f"Artifact was trained with scikit-learn {trained_with} but "
                        f"{sklearn.__version__} is installed. Unpickling across minor "
                        "versions is not guaranteed and can change predictions silently, so "
                        "this artifact is refused rather than served. Align the serving "
                        "image with the training environment - the versions are recorded in "
                        "artifacts/metrics.json under 'environment'."
                    )

                logger.warning(
                    "Artifact was trained with scikit-learn %s but %s is installed. Patch "
                    "versions differ only; loading, but the serving image should match the "
                    "training environment.",
                    trained_with,
                    sklearn.__version__,
                )

            logger.info(
                "Loaded %s v%s (artifact sha256=%s, dataset sha256=%s, trained %s).",
                payload.get("model_name", "unknown"),
                payload.get("model_version", "0.0.0"),
                artifact_sha256[:16],
                str(payload.get("dataset_sha256", "unknown"))[:16],
                payload.get("trained_at", "unknown"),
            )

            predictor = cls(
                pipeline=pipeline,
                model_name=payload.get("model_name", "unknown"),
                model_version=payload.get("model_version", "0.0.0"),
                trained_at=payload.get("trained_at"),
                metrics=payload.get("metrics", {}),
            )
            predictor.artifact_sha256 = artifact_sha256
            predictor.dataset_sha256 = payload.get("dataset_sha256")
            predictor.sklearn_version = trained_with
            predictor.holdout_seed = payload.get("holdout_seed")
            return predictor

        predictor = cls(pipeline=payload)
        predictor.artifact_sha256 = artifact_sha256
        return predictor

    # ------------------------------------------------------------------ inference

    def predict(self, frame: pd.DataFrame) -> list[PredictionResult]:
        """Score every row, returning class probabilities when the model exposes them."""
        if frame.empty:
            return []

        classes = self._classes()
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = np.asarray(self.pipeline.predict_proba(frame))
            predicted_indices = probabilities.argmax(axis=1)
            return [
                PredictionResult.from_class(
                    predicted_class=int(classes[index]),
                    probabilities=self._label_probabilities(classes, row),
                )
                for index, row in zip(predicted_indices, probabilities, strict=True)
            ]

        # ravel(): CatBoost returns a (n, 1) column vector for multiclass problems, which
        # would otherwise yield arrays rather than scalars when iterated.
        predictions = np.asarray(self.pipeline.predict(frame)).ravel()
        return [PredictionResult.from_class(int(value)) for value in predictions]

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the deployed model, including enough lineage to trace a prediction.

        The ``lineage`` block answers a question ``model_version`` cannot. That string is
        "1.0.0" on every retrain, so it identifies the *contract*, not the *artifact*. Six
        weeks after a campaign, "why did it recommend group 2?" needs the specific file:
        which bytes, trained on which dataset, with which library, under which split seed.

        All four travel with the artifact and are surfaced here rather than living only in a
        build log, because a build log is a different system that may not still exist.
        """
        return {
            "predictor": self.name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "lineage": {
                # SHA-256 of the loaded file, computed at load time - so it describes what
                # this process is serving, not what a build step believed it wrote.
                "artifact_sha256": self.artifact_sha256,
                "dataset_sha256": self.dataset_sha256,
                "sklearn_version": self.sklearn_version,
                "holdout_seed": self.holdout_seed,
            },
            "metrics": self.metrics,
            "classes": [int(value) for value in self._classes()],
            "is_baseline": False,
        }

    # ------------------------------------------------------------------ internals

    def _classes(self) -> np.ndarray:
        classes = getattr(self.pipeline, "classes_", None)
        if classes is None:
            final_step = getattr(self.pipeline, "_final_estimator", None)
            classes = getattr(final_step, "classes_", None)
        if classes is None:
            classes = np.array(sorted(CLASS_LABELS))
        return np.asarray(classes)

    @staticmethod
    def _label_probabilities(classes: np.ndarray, row: np.ndarray) -> dict[str, float]:
        return {
            CLASS_LABELS[int(class_value)]: float(probability)
            for class_value, probability in zip(classes, row, strict=True)
        }
