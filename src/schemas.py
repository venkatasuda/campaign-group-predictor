"""Pydantic request/response models.

These schemas are the public contract of the API. They validate input *before* it ever
reaches the model, and they generate the OpenAPI documentation served at ``/docs``.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.constants import (
    BASE_FEATURES,
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
)

#: Widest value any feature may take.
#:
#: Not a modelling constraint - a sanity bound. The training data lies well inside this, and
#: a value outside it is a broken upstream join or a unit error, not a campaign. The bound
#: exists because Pydantic's ``float`` accepts anything IEEE-754 can express: ``1e308``
#: validates, reaches the pipeline, and overflows during standardisation. The caller then
#: receives an opaque 500 for what is really a bad request.
#:
#: Deliberately loose. A tight bound derived from training quantiles would reject legitimate
#: drift and turn a monitoring signal into an outage.
_FEATURE_ABS_LIMIT = 1e6

#: Share of the 67 features that may be null before the request is refused.
#:
#: The median imputer will fill anything, silently and confidently - which is the problem. A
#: request carrying 67 nulls currently returns a prediction indistinguishable from a real
#: one, and that prediction can allocate budget. Imputation is a tool for the occasional
#: gap, not a way to manufacture a campaign from nothing.
#:
#: 20% is a judgement, stated rather than tuned: below it the model is interpolating within
#: evidence, above it the answer is mostly the training median wearing a confidence score.
_MAX_MISSING_FRACTION = 0.20

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

    _validate_numeric_range(values, block_name)
    return values


def _validate_numeric_range(values: dict[str, float | None], block_name: str) -> None:
    """Reject NaN, infinity and implausible magnitudes.

    ``None`` is allowed - a genuinely missing value is a legitimate input and the pipeline
    imputes it. ``NaN`` is not: it is indistinguishable from a missing value downstream but
    arrives through a different path, usually a failed upstream computation. Accepting both
    means the caller has two ways to say "no value" and only one of them is deliberate.

    Infinity and extreme magnitudes are rejected here rather than allowed to fail inside the
    pipeline, because the failure there is an unhandled overflow and the caller sees a 500 -
    a server error for what is unambiguously a client error.
    """
    for key, value in values.items():
        if value is None:
            continue
        if math.isnan(value):
            raise ValueError(
                f"{block_name}.{key} is NaN. Use null for a missing value; NaN usually "
                "indicates a failed computation upstream rather than an absent measurement."
            )
        if math.isinf(value):
            raise ValueError(f"{block_name}.{key} is infinite, which cannot be a measurement.")
        if abs(value) > _FEATURE_ABS_LIMIT:
            raise ValueError(
                f"{block_name}.{key} = {value:g} exceeds the plausible range "
                f"(|value| <= {_FEATURE_ABS_LIMIT:g}). This is usually a unit error or a "
                "broken upstream join rather than a real campaign."
            )


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

    @model_validator(mode="after")
    def _check_missingness(self) -> ComparisonRequest:
        """Refuse a request that is mostly absent.

        This check spans all three blocks, so it cannot live in a field validator - twenty
        nulls in one block is a different situation from twenty spread across sixty-seven
        features, and only the whole request shows which one happened.

        Why it matters more than it looks: the pipeline's median imputer fills every gap
        without complaint, so a request of 67 nulls produces a perfectly well-formed
        prediction carrying a confidence score - built entirely from training medians. It is
        indistinguishable from a real answer, and it can allocate budget. **A campaign must
        never be approved from an empty request.**

        Returning 422 with the count and the threshold makes this a data-quality problem the
        caller can fix, rather than a silent one nobody notices.
        """
        missing = sum(
            1
            for block in (self.group_1, self.group_2, self.comparison)
            for value in block.values()
            if value is None
        )
        limit = int(len(BASE_FEATURES) * _MAX_MISSING_FRACTION)

        if missing > limit:
            raise ValueError(
                f"{missing} of {len(BASE_FEATURES)} features are null, above the limit of "
                f"{limit} ({_MAX_MISSING_FRACTION:.0%}). The prediction would be built "
                "mostly from training medians rather than from this campaign."
            )
        return self


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
    model_version: str = Field(
        default="unknown",
        description=(
            "Version of the artifact that produced this prediction. Present on the "
            "prediction itself, not only on /model/info: reconciling a logged decision "
            "months later requires knowing which model made it, and a separate endpoint "
            "answers only what is loaded *now*."
        ),
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
