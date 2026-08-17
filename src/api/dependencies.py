"""FastAPI dependency providers.

Dependency injection keeps the route handlers free of construction logic and lets tests
override any collaborator with ``app.dependency_overrides``.

Single source of configuration
------------------------------
Settings are read from ``request.app.state.settings``, which ``create_app`` populates.
An earlier version called the module-level ``get_settings()`` here, which meant the app
could be built with one ``Settings`` object while its dependencies silently used another -
a split-brain that made tests pass against configuration the app was not running with.
The application object is now the only source of truth, with the global cache as a
fallback for code paths that construct a dependency outside a request.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from src.config import Settings, get_settings
from src.decision import CostMatrix, DecisionPolicy
from src.features import FeatureTransformer
from src.predictors import BasePredictor
from src.registry import ModelRegistry

# The adapter is stateless, so a single shared instance is safe.
_feature_transformer = FeatureTransformer()


@lru_cache
def _build_policy(
    campaign_spend: float,
    profit_if_correct: float,
    opportunity_weight: float,
    review_margin: float,
    exploration_rate: float,
    currency: str,
) -> DecisionPolicy:
    """Construct (and cache) a decision policy from primitive settings values.

    Cached on the primitives rather than on the ``Settings`` object so that two
    equivalent configurations share one instance and the cache key stays hashable.

    **The cache is required for correctness, not only for speed.** ``DecisionPolicy`` holds
    a persistent random stream for exploration draws, advanced across calls. Removing
    ``@lru_cache`` would construct a fresh policy per request, every one restarting that
    stream from ``exploration_seed`` and drawing the same first value - which means
    exploration either fires on every request or on none, and never at the configured rate.

    That is not hypothetical: the equivalent defect existed inside ``decide_batch`` and
    silently disabled exploration in production while batch-based tests passed. If this
    decorator is ever removed, the exploration stream has to move somewhere with an
    application-scoped lifetime instead.
    """
    return DecisionPolicy(
        cost_matrix=CostMatrix.from_business_parameters(
            campaign_spend=campaign_spend,
            profit_if_correct=profit_if_correct,
            opportunity_weight=opportunity_weight,
            currency=currency,
        ),
        review_margin=review_margin,
        exploration_rate=exploration_rate,
    )


def get_app_settings(request: Request) -> Settings:
    """Return the settings the running application was built with."""
    settings = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def get_registry() -> ModelRegistry:
    """Return the singleton model registry."""
    return ModelRegistry()


def get_predictor() -> BasePredictor:
    """Return the currently loaded predictor."""
    return ModelRegistry().predictor


def get_feature_transformer() -> FeatureTransformer:
    """Return the payload -> DataFrame adapter."""
    return _feature_transformer


def get_decision_policy(request: Request) -> DecisionPolicy | None:
    """Return the active cost-sensitive decision policy, or ``None`` when disabled.

    Built from the *application's* settings, so an app constructed with a custom
    ``Settings`` object gets a policy matching it.
    """
    settings = get_app_settings(request)
    if not settings.enable_decision_layer:
        return None
    return _build_policy(
        settings.campaign_spend,
        settings.profit_if_correct,
        settings.opportunity_weight,
        settings.decision_review_margin,
        settings.exploration_rate,
        settings.decision_currency,
    )
