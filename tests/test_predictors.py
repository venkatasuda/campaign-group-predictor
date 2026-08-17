"""Unit tests for the predictor strategies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.constants import BASE_FEATURES, CLASS_LABELS, HISTORICAL_CLASS_DISTRIBUTION
from src.exceptions import ModelArtifactError
from src.predictors import (
    BasePredictor,
    MajorityClassPredictor,
    PredictionResult,
    SklearnPipelinePredictor,
)


class TestPredictionResult:
    def test_from_class_fills_business_metadata(self) -> None:
        result = PredictionResult.from_class(1, {"group_1": 0.7, "group_2": 0.2})
        assert result.label == "group_1"
        assert result.description == "Group 1 was the most profitable"
        assert result.recommended_action == "Target customer group 1."
        assert result.confidence == pytest.approx(0.7)

    def test_class_zero_recommends_skipping_the_campaign(self) -> None:
        result = PredictionResult.from_class(0)
        assert "Do not run" in result.recommended_action

    def test_to_dict_is_serialisable(self) -> None:
        payload = PredictionResult.from_class(2, {"group_2": 1.0}).to_dict()
        assert set(payload) == {
            "predicted_class",
            "label",
            "description",
            "recommended_action",
            "confidence",
            "probabilities",
        }

    def test_is_immutable(self) -> None:
        result = PredictionResult.from_class(1)
        with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
            result.predicted_class = 2  # type: ignore[misc]


class TestMajorityClassPredictor:
    def test_is_a_base_predictor(self) -> None:
        assert isinstance(MajorityClassPredictor(), BasePredictor)

    def test_predicts_the_configured_class_for_every_row(self, feature_frame: pd.DataFrame) -> None:
        results = MajorityClassPredictor(majority_class=2).predict(feature_frame.head(4))
        assert len(results) == 4
        assert {result.predicted_class for result in results} == {2}

    def test_rejects_an_invalid_class(self) -> None:
        with pytest.raises(ValueError, match="majority_class"):
            MajorityClassPredictor(majority_class=9)

    def test_metadata_flags_it_as_a_baseline(self) -> None:
        assert MajorityClassPredictor().metadata["is_baseline"] is True

    def test_empty_frame_returns_no_results(self) -> None:
        assert MajorityClassPredictor().predict(pd.DataFrame(columns=BASE_FEATURES)) == []

    def test_does_not_claim_certainty_it_does_not_have(self, feature_frame: pd.DataFrame) -> None:
        """The baseline must report its historical hit rate, not 1.0.

        A one-hot default would have this predictor answer ``confidence: 1.0`` on every
        call while being right 46% of the time. That value is not cosmetic - the
        cost-sensitive decision layer multiplies it by money, and the automation gate uses
        it to decide which campaigns a human never reviews. A baseline that overstates its
        certainty suppresses exactly the review that should catch it.
        """
        result = MajorityClassPredictor(majority_class=1).predict(feature_frame.head(1))[0]

        assert result.confidence < 1.0
        assert result.confidence == pytest.approx(
            HISTORICAL_CLASS_DISTRIBUTION["group_1"], abs=1e-4
        )

    def test_reported_probabilities_form_a_distribution(self, feature_frame: pd.DataFrame) -> None:
        result = MajorityClassPredictor().predict(feature_frame.head(1))[0]
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


class TestSklearnPipelinePredictor:
    def test_loads_from_artifact(self, trained_artifact: Path) -> None:
        predictor = SklearnPipelinePredictor.from_artifact(trained_artifact)
        assert predictor.model_name == "logistic_regression"
        assert predictor.metadata["is_baseline"] is False

    def test_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelArtifactError, match="not found"):
            SklearnPipelinePredictor.from_artifact(tmp_path / "nope.pkl")

    def test_corrupt_artifact_raises(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.pkl"
        broken.write_bytes(b"this is not a joblib file")
        with pytest.raises(ModelArtifactError, match="deserialise"):
            SklearnPipelinePredictor.from_artifact(broken)

    def test_artifact_without_pipeline_key_raises(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "no_pipeline.pkl"
        joblib.dump({"model_name": "x"}, path)
        with pytest.raises(ModelArtifactError, match="pipeline"):
            SklearnPipelinePredictor.from_artifact(path)

    def test_object_without_predict_is_rejected(self) -> None:
        with pytest.raises(ModelArtifactError, match="predict"):
            SklearnPipelinePredictor(pipeline=object())

    def test_predicts_valid_classes_with_probabilities(
        self, trained_artifact: Path, feature_frame: pd.DataFrame
    ) -> None:
        predictor = SklearnPipelinePredictor.from_artifact(trained_artifact)
        results = predictor.predict(feature_frame.head(10))

        assert len(results) == 10
        for result in results:
            assert result.predicted_class in CLASS_LABELS
            assert 0.0 <= result.confidence <= 1.0
            assert result.probabilities
            assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_empty_frame_returns_no_results(self, trained_artifact: Path) -> None:
        predictor = SklearnPipelinePredictor.from_artifact(trained_artifact)
        assert predictor.predict(pd.DataFrame(columns=BASE_FEATURES)) == []

    def test_metadata_exposes_version_and_metrics(self, trained_artifact: Path) -> None:
        metadata = SklearnPipelinePredictor.from_artifact(trained_artifact).metadata
        assert metadata["model_version"] == "test"
        assert metadata["metrics"]["f1_macro"] == 0.9
        assert sorted(metadata["classes"]) == [0, 1, 2]
