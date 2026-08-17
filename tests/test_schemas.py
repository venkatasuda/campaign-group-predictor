"""Unit tests for the Pydantic request/response contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.constants import GROUP_1_FEATURES
from src.schemas import (
    BatchComparisonRequest,
    ComparisonRequest,
    HealthResponse,
    PredictionResponse,
)


class TestComparisonRequest:
    def test_accepts_a_valid_payload(self, valid_payload: dict) -> None:
        request = ComparisonRequest(**valid_payload)
        assert len(request.group_1) == len(GROUP_1_FEATURES)

    def test_rejects_missing_keys(self, valid_payload: dict) -> None:
        del valid_payload["group_2"]["g2_1"]
        with pytest.raises(ValidationError, match="missing required key"):
            ComparisonRequest(**valid_payload)

    def test_rejects_unexpected_keys(self, valid_payload: dict) -> None:
        valid_payload["group_1"]["g1_42"] = 1.0
        with pytest.raises(ValidationError, match="unexpected key"):
            ComparisonRequest(**valid_payload)

    def test_rejects_post_campaign_variables(self, valid_payload: dict) -> None:
        valid_payload["comparison"]["c_28"] = 1.0
        with pytest.raises(ValidationError, match="post-campaign"):
            ComparisonRequest(**valid_payload)

    def test_allows_null_values_for_missing_measurements(self, valid_payload: dict) -> None:
        valid_payload["group_1"]["g1_1"] = None
        assert ComparisonRequest(**valid_payload).group_1["g1_1"] is None

    def test_rejects_a_missing_block(self, valid_payload: dict) -> None:
        del valid_payload["comparison"]
        with pytest.raises(ValidationError):
            ComparisonRequest(**valid_payload)


class TestBatchComparisonRequest:
    def test_accepts_a_batch(self, valid_payload: dict) -> None:
        request = BatchComparisonRequest(comparisons=[valid_payload, valid_payload])
        assert len(request.comparisons) == 2

    def test_rejects_an_empty_batch(self) -> None:
        with pytest.raises(ValidationError):
            BatchComparisonRequest(comparisons=[])


class TestResponses:
    def test_prediction_response_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_class=1,
                label="group_1",
                description="Group 1 was the most profitable",
                recommended_action="Target customer group 1.",
                confidence=1.5,
            )

    def test_health_response_round_trips(self) -> None:
        payload = HealthResponse(
            status="ok", app_name="api", api_version="1.0.0", model_loaded=True
        ).model_dump()
        assert payload["model_loaded"] is True
