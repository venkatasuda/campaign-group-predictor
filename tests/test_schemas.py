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


class TestNumericValidation:
    """Values that are syntactically float but cannot be measurements.

    Pydantic's ``float`` accepts anything IEEE-754 can express. Without these checks the
    request validates, the pipeline overflows during standardisation, and the caller receives
    a 500 - a server error for what is unambiguously a client error.
    """

    def test_rejects_nan(self, valid_payload: dict) -> None:
        valid_payload["group_1"]["g1_1"] = float("nan")
        with pytest.raises(ValidationError, match="NaN"):
            ComparisonRequest(**valid_payload)

    def test_rejects_positive_infinity(self, valid_payload: dict) -> None:
        valid_payload["comparison"]["c_1"] = float("inf")
        with pytest.raises(ValidationError, match="infinite"):
            ComparisonRequest(**valid_payload)

    def test_rejects_negative_infinity(self, valid_payload: dict) -> None:
        valid_payload["group_2"]["g2_5"] = float("-inf")
        with pytest.raises(ValidationError, match="infinite"):
            ComparisonRequest(**valid_payload)

    def test_rejects_an_implausible_magnitude(self, valid_payload: dict) -> None:
        """1e308 is finite, so it passes every type check and breaks arithmetic instead."""
        valid_payload["group_1"]["g1_2"] = 1e308
        with pytest.raises(ValidationError, match="plausible range"):
            ComparisonRequest(**valid_payload)

    def test_accepts_a_large_but_plausible_value(self, valid_payload: dict) -> None:
        """The bound is a sanity check, not a modelling constraint.

        A tight bound derived from training quantiles would reject legitimate drift and turn
        a monitoring signal into an outage.
        """
        valid_payload["group_1"]["g1_2"] = 999_999.0
        assert ComparisonRequest(**valid_payload).group_1["g1_2"] == 999_999.0

    def test_null_is_still_allowed(self, valid_payload: dict) -> None:
        """A genuinely absent measurement is legitimate; the pipeline imputes it.

        Only NaN is refused - it is indistinguishable from null downstream but arrives
        through a different path, usually a failed upstream computation.
        """
        valid_payload["group_1"]["g1_3"] = None
        assert ComparisonRequest(**valid_payload).group_1["g1_3"] is None


class TestMissingnessLimit:
    """A campaign must never be approved from an empty request."""

    def test_rejects_an_entirely_null_payload(self, valid_payload: dict) -> None:
        """The headline case.

        Every value null: the median imputer fills all 67, the model returns a well-formed
        prediction with a confidence score, and that prediction is built entirely from
        training medians. It is indistinguishable from a real answer and it can allocate
        budget.
        """
        for block in ("group_1", "group_2", "comparison"):
            valid_payload[block] = dict.fromkeys(valid_payload[block], None)

        with pytest.raises(ValidationError, match="above the limit"):
            ComparisonRequest(**valid_payload)

    def test_rejects_just_over_the_limit(self, valid_payload: dict) -> None:
        # 20% of 67 features is 13; 14 nulls must fail.
        for key in list(valid_payload["group_1"])[:14]:
            valid_payload["group_1"][key] = None

        with pytest.raises(ValidationError, match="above the limit"):
            ComparisonRequest(**valid_payload)

    def test_accepts_sparse_but_tolerable_missingness(self, valid_payload: dict) -> None:
        """Occasional gaps are what the imputer is for. Ten nulls is under the limit."""
        for key in list(valid_payload["group_1"])[:10]:
            valid_payload["group_1"][key] = None

        assert ComparisonRequest(**valid_payload) is not None

    def test_the_limit_counts_across_all_three_blocks(self, valid_payload: dict) -> None:
        """Spread thinly, the same total is still too much.

        Twenty nulls in one block and twenty across sixty-seven features are different
        situations, and only the whole request shows which happened - which is why this
        cannot be a field validator.
        """
        for block in ("group_1", "group_2", "comparison"):
            for key in list(valid_payload[block])[:5]:
                valid_payload[block][key] = None

        with pytest.raises(ValidationError, match="above the limit"):
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
