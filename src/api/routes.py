"""API routes exposing the prediction service."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Response, status

from src.api.dependencies import (
    get_app_settings,
    get_decision_policy,
    get_feature_transformer,
    get_predictor,
    get_registry,
)
from src.config import Settings
from src.constants import (
    CLASS_LABELS,
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
    LEAKAGE_FEATURES,
)
from src.decision import DecisionPolicy
from src.factory import ModelFactory
from src.features import FeatureTransformer
from src.logging_config import get_logger
from src.predictors import BasePredictor, PredictionResult
from src.registry import ModelRegistry
from src.schemas import (
    BatchComparisonRequest,
    BatchPredictionResponse,
    ComparisonRequest,
    DecisionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = get_logger(__name__)

router = APIRouter()

#: Canonical class order used when turning the probability dict back into a vector.
_CLASS_ORDER = [CLASS_LABELS[class_value] for class_value in sorted(CLASS_LABELS)]


def _to_response(
    result: PredictionResult,
    policy: DecisionPolicy | None,
    automation_threshold: float = 0.0,
    model_version: str = "unknown",
) -> PredictionResponse:
    """Convert a model result into the API response, attaching the decision if enabled.

    The decision layer is skipped silently when the predictor exposes no probabilities
    (for example the majority-class baseline), so a degraded model still serves valid
    responses instead of erroring.

    Two independent reasons to route a campaign to a person, and both are applied:

    * **The cost margin** - the best and second-best actions are nearly tied, so the
      recommendation is not robust to small errors in the probabilities.
    * **The confidence gate** - the model's own probability for its chosen class is below
      the threshold selected on calibration data. This is the operating point the analysis
      identifies: the model is materially more accurate on the subset it is confident
      about, and escalating the remainder is what makes that accuracy usable rather than
      merely reportable.

    A response can clear the margin test and still fail the gate. Applying only one of
    them would automate exactly the campaigns the analysis says a person should see.
    """
    payload = result.to_dict()

    decision_payload = None
    if policy is not None and result.probabilities:
        vector = np.array([float(result.probabilities.get(label, 0.0)) for label in _CLASS_ORDER])
        total = vector.sum()
        if total > 0:
            decision = policy.decide(vector / total)
            decision_dict = decision.to_dict()

            if automation_threshold > 0.0 and result.confidence < automation_threshold:
                decision_dict["review_required"] = True
                decision_dict["rationale"] = (
                    f"{decision_dict['rationale']} Confidence {result.confidence:.2f} is "
                    f"below the automation threshold of {automation_threshold:.2f}, so "
                    "this campaign is routed to a human."
                )

            decision_payload = DecisionResponse(**decision_dict)

    return PredictionResponse(**payload, model_version=model_version, decision=decision_payload)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Liveness and readiness probe",
)
def health(
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Report whether the service is able to serve predictions.

    Three states, not two. A predictor being *loaded* is not the same as the *trained
    model* being loaded: when the artifact cannot be read and
    ``ALLOW_BASELINE_FALLBACK`` is on, the majority-class baseline loads successfully and
    ``is_loaded`` becomes true.

    Reporting that as ``ok`` is the worst available failure mode. Monitoring stays green,
    readiness probes pass, traffic is routed, and campaign budget gets allocated by a rule
    that is right 46% of the time - and nobody finds out until someone audits outcomes
    weeks later. ``degraded`` is what a fallback deserves.
    """
    serving_baseline = registry.is_loaded and bool(
        registry.predictor.metadata.get("is_baseline", False)
    )

    status = "degraded" if (not registry.is_loaded or serving_baseline) else "ok"

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        api_version=settings.api_version,
        # True only for a real trained artifact. A monitoring rule written against this
        # field must not be satisfied by the fallback.
        model_loaded=registry.is_loaded and not serving_baseline,
        serving_baseline=serving_baseline,
    )


@router.get("/live", tags=["operations"], summary="Liveness probe")
def live() -> dict[str, str]:
    """Is the process running? Nothing more.

    Deliberately checks nothing except that the event loop can answer. Liveness and readiness
    answer different questions and an orchestrator responds to them differently: a failed
    liveness probe means *restart the container*, a failed readiness probe means *stop
    sending it traffic*.

    Conflating them makes both wrong. A liveness probe that also checks the model would
    restart a container whose only problem is a missing artifact - producing a crash loop
    that fixes nothing, because the next container will be missing the same artifact.
    """
    return {"status": "alive"}


@router.get(
    "/ready",
    tags=["operations"],
    summary="Readiness probe - 503 when the service cannot serve predictions",
)
def ready(
    response: Response,
    registry: ModelRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Can this instance serve a real prediction right now?

    Returns **503** rather than 200 when it cannot, which is the difference between this and
    ``/health``. ``/health`` is for humans and dashboards - it reports `degraded` in a 200 so
    the detail is readable. A readiness probe is consumed by a load balancer that only looks
    at the status code, so a degraded instance answering 200 keeps receiving traffic it
    should not receive.

    "Cannot serve" includes serving the majority-class baseline. An instance that answers
    every request with a 46%-accurate rule is not ready; it is a fallback that should be
    drained, investigated, and replaced.
    """
    serving_baseline = registry.is_loaded and bool(
        registry.predictor.metadata.get("is_baseline", False)
    )
    model_ready = registry.is_loaded and not serving_baseline

    if not model_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": model_ready,
        "reason": (
            "ready"
            if model_ready
            else ("serving the majority-class baseline" if serving_baseline else "no model loaded")
        ),
    }


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["operations"],
    summary="Metadata about the deployed model",
)
def model_info(
    predictor: BasePredictor = Depends(get_predictor),
    policy: DecisionPolicy | None = Depends(get_decision_policy),
) -> ModelInfoResponse:
    """Expose model name, version, training date, offline metrics and the cost matrix."""
    return ModelInfoResponse(
        metadata=predictor.metadata,
        available_predictors=ModelFactory.available(),
        expected_features={
            "group_1": GROUP_1_FEATURES,
            "group_2": GROUP_2_FEATURES,
            "comparison": COMPARISON_FEATURES,
        },
        excluded_leakage_features=LEAKAGE_FEATURES,
        decision_policy=policy.to_dict() if policy else None,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    summary="Predict which customer group to target",
)
def predict(
    request: ComparisonRequest,
    predictor: BasePredictor = Depends(get_predictor),
    transformer: FeatureTransformer = Depends(get_feature_transformer),
    policy: DecisionPolicy | None = Depends(get_decision_policy),
    settings: Settings = Depends(get_app_settings),
) -> PredictionResponse:
    """Score a single comparison between two customer groups."""
    frame = transformer.from_payload(
        group_1=request.group_1,
        group_2=request.group_2,
        comparison=request.comparison,
    )
    result = predictor.predict(frame)[0]
    response = _to_response(
        result,
        policy,
        settings.automation_confidence_threshold,
        model_version=str(predictor.metadata.get("model_version", "unknown")),
    )

    # `extra` rather than interpolation only, for the same reason the access log uses it:
    # on Cloud Run these become top-level fields, so "what did the model predict for the
    # campaigns we ran in March, and how confident was it?" is a log query rather than a
    # text search. Interpolated values are readable but not aggregatable, and the questions
    # asked of this service six weeks later are aggregate questions.
    #
    # The feature payload is deliberately NOT logged. It describes real customer segments,
    # and nothing here needs it: the prediction, the confidence and the request ID are enough
    # to explain a decision, while 67 columns per request would be a retention liability
    # nobody has agreed to hold.
    logger.info(
        "Prediction served: class=%s confidence=%.4f action=%s",
        result.predicted_class,
        result.confidence,
        response.decision.action_label if response.decision else "n/a",
        extra={
            "predicted_class": result.predicted_class,
            "predicted_label": response.label,
            "confidence": round(float(result.confidence), 4),
            "model_version": response.model_version,
            "decision_action": response.decision.action_label if response.decision else None,
            "review_required": response.decision.review_required if response.decision else None,
            "batch_size": 1,
        },
    )
    return response


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    summary="Score many comparisons in one call",
)
def predict_batch(
    request: BatchComparisonRequest,
    predictor: BasePredictor = Depends(get_predictor),
    transformer: FeatureTransformer = Depends(get_feature_transformer),
    policy: DecisionPolicy | None = Depends(get_decision_policy),
    settings: Settings = Depends(get_app_settings),
) -> BatchPredictionResponse:
    """Score a batch of comparisons - used for campaign planning over many segments."""
    frame = transformer.from_payloads(
        (item.group_1, item.group_2, item.comparison) for item in request.comparisons
    )
    results = predictor.predict(frame)
    threshold = settings.automation_confidence_threshold
    version = str(predictor.metadata.get("model_version", "unknown"))

    # Batch size as a field, not only in the message: it is the input that determines cost
    # per request, so it is what a latency percentile has to be read against. A slow batch of
    # 900 and a slow batch of 2 are different incidents.
    logger.info(
        "Batch prediction served: %d comparison(s).",
        len(results),
        extra={
            "batch_size": len(results),
            "model_version": version,
            "class_counts": {
                str(class_value): sum(1 for r in results if r.predicted_class == class_value)
                for class_value in sorted({r.predicted_class for r in results})
            },
        },
    )
    return BatchPredictionResponse(
        predictions=[
            _to_response(result, policy, threshold, model_version=version) for result in results
        ],
        count=len(results),
    )
