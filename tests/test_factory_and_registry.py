"""Unit tests for the Factory and Singleton components."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Settings
from src.exceptions import ModelNotLoadedError, PredictorNotRegisteredError
from src.factory import ModelFactory
from src.predictors import BasePredictor, MajorityClassPredictor, PredictionResult
from src.registry import ModelRegistry


class TestModelFactory:
    def test_builtin_strategies_are_registered(self) -> None:
        assert "majority_class" in ModelFactory.available()
        assert "sklearn_pipeline" in ModelFactory.available()

    def test_creates_a_registered_predictor(self) -> None:
        predictor = ModelFactory.create("majority_class", majority_class=2)
        assert isinstance(predictor, MajorityClassPredictor)
        assert predictor.majority_class == 2

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(PredictorNotRegisteredError, match="Unknown predictor"):
            ModelFactory.create("does_not_exist")

    def test_create_from_artifact_with_unknown_name_raises(self) -> None:
        with pytest.raises(PredictorNotRegisteredError):
            ModelFactory.create_from_artifact("does_not_exist", "model.pkl")

    def test_create_from_artifact_loads_a_pipeline(self, trained_artifact: Path) -> None:
        predictor = ModelFactory.create_from_artifact("sklearn_pipeline", str(trained_artifact))
        assert predictor.metadata["model_name"] == "logistic_regression"

    def test_create_from_artifact_ignores_path_for_artifactless_predictors(self) -> None:
        predictor = ModelFactory.create_from_artifact("majority_class", "irrelevant.pkl")
        assert isinstance(predictor, MajorityClassPredictor)

    def test_new_strategies_can_be_registered(self, feature_frame: pd.DataFrame) -> None:
        @ModelFactory.register("always_group_2_test")
        class AlwaysGroupTwo(BasePredictor):
            name = "always_group_2_test"

            def predict(self, frame: pd.DataFrame) -> list[PredictionResult]:
                return [PredictionResult.from_class(2) for _ in range(len(frame))]

            @property
            def metadata(self) -> dict:
                return {"predictor": self.name}

        predictor = ModelFactory.create("always_group_2_test")
        assert predictor.predict(feature_frame.head(2))[0].predicted_class == 2


class TestModelRegistry:
    def test_is_a_singleton(self) -> None:
        assert ModelRegistry() is ModelRegistry()

    def test_predictor_access_before_load_raises(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(ModelNotLoadedError):
            _ = registry.predictor

    def test_is_loaded_reflects_state(self) -> None:
        registry = ModelRegistry()
        assert registry.is_loaded is False
        registry.set_predictor(MajorityClassPredictor())
        assert registry.is_loaded is True

    def test_set_predictor_rejects_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="BasePredictor"):
            ModelRegistry().set_predictor("not a predictor")  # type: ignore[arg-type]

    def test_load_falls_back_to_baseline_when_artifact_is_missing(self, settings: Settings) -> None:
        predictor = ModelRegistry().load(settings)
        assert predictor.metadata["is_baseline"] is True

    def test_load_raises_when_fallback_is_disabled(self, tmp_path: Path) -> None:
        strict = Settings(
            model_path=str(tmp_path / "missing.pkl"),
            model_type="sklearn_pipeline",
            allow_baseline_fallback=False,
        )
        with pytest.raises(Exception):  # noqa: B017 - ModelArtifactError subclass
            ModelRegistry().load(strict)

    def test_load_uses_the_artifact_when_present(self, trained_artifact: Path) -> None:
        settings = Settings(
            model_path=str(trained_artifact),
            model_type="sklearn_pipeline",
            allow_baseline_fallback=False,
        )
        predictor = ModelRegistry().load(settings)
        assert predictor.metadata["is_baseline"] is False

    def test_metadata_delegates_to_the_predictor(self) -> None:
        registry = ModelRegistry()
        registry.set_predictor(MajorityClassPredictor(majority_class=1))
        assert registry.metadata()["majority_class"] == 1

    def test_reset_clears_the_predictor(self) -> None:
        registry = ModelRegistry()
        registry.set_predictor(MajorityClassPredictor())
        registry.reset()
        assert registry.is_loaded is False
