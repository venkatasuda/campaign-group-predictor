"""Model registry.

Design pattern: **Singleton**. The trained pipeline is expensive to deserialise, so it
is loaded once per process and shared by every request. The API never touches this class
directly - it receives the predictor through dependency injection, which makes the
predictor trivially replaceable in tests.
"""

from __future__ import annotations

import threading
from typing import Any

from src.config import Settings, get_settings
from src.exceptions import ModelNotLoadedError
from src.factory import ModelFactory
from src.logging_config import get_logger
from src.predictors import BasePredictor

logger = get_logger(__name__)


class ModelRegistry:
    """Thread-safe, process-wide holder for the active predictor."""

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._predictor = None
                    cls._instance = instance
        return cls._instance

    # ---------------------------------------------------------------------- state

    @property
    def is_loaded(self) -> bool:
        """Whether a predictor is currently available."""
        return self._predictor is not None

    @property
    def predictor(self) -> BasePredictor:
        """Return the active predictor, or raise if none has been loaded."""
        if self._predictor is None:
            raise ModelNotLoadedError(
                "No model has been loaded. Call ModelRegistry().load() during startup."
            )
        return self._predictor

    # --------------------------------------------------------------------- actions

    def load(self, settings: Settings | None = None) -> BasePredictor:
        """Load the configured predictor, optionally degrading to the baseline.

        Returns the predictor that ended up active so callers can log it.
        """
        settings = settings or get_settings()

        try:
            predictor = ModelFactory.create_from_artifact(
                name=settings.model_type,
                artifact_path=settings.model_path,
            )
            logger.info(
                "Loaded predictor '%s' from '%s'.", settings.model_type, settings.model_path
            )
        except Exception as error:  # noqa: BLE001 - any load failure is handled identically
            if not settings.allow_baseline_fallback:
                logger.error("Model loading failed and fallback is disabled: %s", error)
                raise
            logger.warning(
                "Model loading failed (%s). Falling back to the majority-class baseline. "
                "This must not happen in production.",
                error,
            )
            predictor = ModelFactory.create("majority_class")

        self.set_predictor(predictor)
        return predictor

    def set_predictor(self, predictor: BasePredictor) -> None:
        """Inject a predictor directly. Used by ``load`` and by tests."""
        if not isinstance(predictor, BasePredictor):
            raise TypeError("predictor must implement BasePredictor")
        self._predictor = predictor

    def metadata(self) -> dict[str, Any]:
        """Return the active predictor's metadata."""
        return self.predictor.metadata

    def reset(self) -> None:
        """Drop the active predictor. Intended for tests."""
        self._predictor = None

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton entirely. Intended for tests."""
        with cls._lock:
            cls._instance = None
