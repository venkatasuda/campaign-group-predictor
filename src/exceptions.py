"""Domain-specific exceptions.

Keeping a dedicated exception hierarchy lets the API layer translate failures into
HTTP responses without leaking implementation details of the ML layer.
"""

from __future__ import annotations


class CampaignPredictorError(Exception):
    """Base class for every error raised by this application."""


class ModelNotLoadedError(CampaignPredictorError):
    """Raised when a prediction is requested before a model has been loaded."""


class ModelArtifactError(CampaignPredictorError):
    """Raised when a model artifact is missing or cannot be deserialised."""


class PredictorNotRegisteredError(CampaignPredictorError):
    """Raised when an unknown predictor name is requested from the factory."""


class InvalidFeaturePayloadError(CampaignPredictorError):
    """Raised when an incoming payload does not match the expected feature contract."""
