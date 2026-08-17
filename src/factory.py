"""Predictor factory.

Design pattern: **Factory** + a registration decorator. New predictor strategies are
added by decorating the class - no ``if/elif`` chain has to be edited, which keeps the
module closed for modification and open for extension (Open/Closed Principle).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from src.exceptions import PredictorNotRegisteredError
from src.predictors import BasePredictor, MajorityClassPredictor, SklearnPipelinePredictor

PredictorType = TypeVar("PredictorType", bound=BasePredictor)


class ModelFactory:
    """Create predictor instances from a registered name."""

    _registry: dict[str, type[BasePredictor]] = {}

    @classmethod
    def register(
        cls, name: str | None = None
    ) -> Callable[[type[PredictorType]], type[PredictorType]]:
        """Class decorator registering a predictor implementation under ``name``."""

        def decorator(predictor_cls: type[PredictorType]) -> type[PredictorType]:
            key = name or predictor_cls.name
            cls._registry[key] = predictor_cls
            return predictor_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BasePredictor:
        """Instantiate the predictor registered under ``name``."""
        try:
            predictor_cls = cls._registry[name]
        except KeyError as error:
            raise PredictorNotRegisteredError(
                f"Unknown predictor '{name}'. Available: {sorted(cls._registry)}"
            ) from error
        return predictor_cls(**kwargs)

    @classmethod
    def create_from_artifact(cls, name: str, artifact_path: str) -> BasePredictor:
        """Instantiate a predictor from a serialised artifact when it supports one."""
        predictor_cls = cls._registry.get(name)
        if predictor_cls is None:
            raise PredictorNotRegisteredError(
                f"Unknown predictor '{name}'. Available: {sorted(cls._registry)}"
            )
        if hasattr(predictor_cls, "from_artifact"):
            return predictor_cls.from_artifact(artifact_path)
        return predictor_cls()

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted list of registered predictor names."""
        return sorted(cls._registry)

    @classmethod
    def reset(cls) -> None:
        """Clear the registry. Intended for tests only."""
        cls._registry.clear()


# Register the built-in strategies.
ModelFactory.register()(MajorityClassPredictor)
ModelFactory.register()(SklearnPipelinePredictor)
