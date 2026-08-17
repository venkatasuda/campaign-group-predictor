"""Pydantic request/response models.

These schemas are the public contract of the API. They validate input *before* it ever
reaches the model, and they generate the OpenAPI documentation served at ``/docs``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.constants import (
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
)

# Annotated as dict[str, Any] rather than inferred. Pydantic types `json_schema_extra` as
# a JSON-value mapping, and an inferred dict[str, dict[str, float]] does not satisfy that
# union - the example is valid JSON, the stub is simply not precise enough to see it.
# Declaring the intent here is more honest than three suppressions at the use site.
_REQUEST_EXAMPLE: dict[str, Any] = {
    "group_1": dict.fromkeys(GROUP_1_FEATURES, 1.0),
    "group_2": dict.fromkeys(GROUP_2_FEATURES, 0.8),
    "comparison": dict.fromkeys(COMPARISON_FEATURES, 0.5),
}


def _validate_keys(
    values: dict[str, float | None],
    required: list[str],
    block_name: str,
) -> dict[str, float | None]:
    """Shared key-set validation used by the three feature blocks."""
    keys = set(values)

    leaked = keys.intersection(LEAKAGE_FEATURES)
    if leaked:
        raise ValueError(
            f"{block_name} must not contain post-campaign variables {sorted(leaked)}; "
            "they are unavailable at prediction time."
        )

    missing = [key for key in required if key not in keys]
    if missing:
        raise ValueError(f"{block_name} is missing required key(s): {', '.join(missing)}")

    unexpected = sorted(keys.difference(required))
    if unexpected:
        raise ValueError(f"{block_name} contains unexpected key(s): {', '.join(unexpected)}")

    return values


class ComparisonRequest(BaseModel):
    """One campaign comparison: the pre-campaign characteristics of both groups."""

    # extra="forbid" makes the contract strict at the top level as well as inside each
    # block. Without it, Pydantic silently discards unknown fields, so a caller sending
    # `{"group_1": ..., "group_2": ..., "comparison": ..., "g1_21": 0.9}` would get a
    # 200 and never learn that the post-campaign value they supplied was ignored rather
    # than used. Rejecting nested unknown keys while accepting top-level ones is an
    # inconsistency a client would eventually be bitten by.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _REQUEST_EXAMPLE},
    )

    group_1: dict[str, float | None] = Field(
        ..., description="Pre-campaign variables g1_1 .. g1_20 for customer group 1."
    )
    group_2: dict[str, float | None] = Field(
        ..., description="Pre-campaign variables g2_1 .. g2_20 for customer group 2."
    )
    comparison: dict[str, float | None] = Field(
        ..., description="Pre-campaign comparison variables c_1 .. c_27."
    )

    @field_validator("group_1")
    @classmethod
    def _check_group_1(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        return _validate_keys(value, GROUP_1_FEATURES, "group_1")

    @field_validator("group_2")
    @classmethod
    def _check_group_2(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        return _validate_keys(value, GROUP_2_FEATURES, "group_2")

    @field_validator("comparison")
    @classmethod
    def _check_comparison(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        return _validate_keys(value, COMPARISON_FEATURES, "comparison")


class BatchComparisonRequest(BaseModel):
    """A batch of campaign comparisons scored in a single call."""

    comparisons: list[ComparisonRequest] = Field(
        ..., min_length=1, max_length=1000, description="Between 1 and 1000 comparisons."
    )


class DecisionResponse(BaseModel):
    """Cost-optimal action, which can differ from the most likely class."""

    action: int = Field(..., description="0 = do not run, 1 = target group 1, 2 = target group 2.")
    action_label: str = Field(..., description="Machine-readable action label.")
    recommended_action: str = Field(..., description="What the campaign manager should do.")
    expected_costs: dict[str, float] = Field(
        default_factory=dict, description="Expected cost of each available action."
    )
    expected_value: float = Field(..., description="Expected value of the chosen action.")
    margin: float = Field(..., description="Expected-cost gap to the runner-up action.")
    argmax_class: int = Field(..., description="The single most likely class.")
    differs_from_argmax: bool = Field(
        ..., description="True when the cost-optimal action is not the most likely class."
    )
    review_required: bool = Field(
        ..., description="True when the decision is too close to automate; route to a human."
    )
    exploration: bool = Field(
        default=False,
        description=(
            "True when a 'do not run' recommendation was deliberately overridden so the "
            "outcome is still observed. Exclude these from performance reporting."
        ),
    )
    rationale: str = Field(..., description="Plain-English justification.")


class PredictionResponse(BaseModel):
    """Prediction for one comparison."""

    predicted_class: int = Field(..., description="0 = neither, 1 = group 1, 2 = group 2.")
    label: str = Field(..., description="Machine-readable class label.")
    description: str = Field(..., description="Business meaning of the predicted class.")
    recommended_action: str = Field(..., description="What the campaign manager should do.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the winning class.")
    probabilities: dict[str, float] = Field(
        default_factory=dict, description="Probability per class label."
    )
    decision: DecisionResponse | None = Field(
        default=None,
        description=(
            "Cost-sensitive recommendation. Present when the model exposes probabilities "
            "and the decision layer is enabled."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch of comparisons."""

    predictions: list[PredictionResponse]
    count: int = Field(..., description="Number of comparisons scored.")


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = Field(
        ...,
        description=(
            "'ok' only when a trained model is serving. 'degraded' when nothing is loaded "
            "or when the majority-class fallback is active."
        ),
    )
    app_name: str
    api_version: str
    model_loaded: bool = Field(
        ...,
        description=(
            "True only for a real trained artifact. Deliberately false when the baseline "
            "fallback is serving, so a monitoring rule on this field cannot be satisfied "
            "by the fallback."
        ),
    )
    serving_baseline: bool = Field(
        default=False,
        description=(
            "True when the majority-class baseline is active because the artifact could "
            "not be loaded. Should never be true in production."
        ),
    )


class ModelInfoResponse(BaseModel):
    """Metadata about the currently deployed model."""

    metadata: dict[str, Any]
    available_predictors: list[str]
    expected_features: dict[str, list[str]]
    excluded_leakage_features: list[str]
    decision_policy: dict[str, Any] | None = Field(
        default=None, description="Active cost matrix and review margin, when enabled."
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error: str = Field(..., description="Exception class name.")
    detail: str = Field(..., description="Human-readable explanation.")
