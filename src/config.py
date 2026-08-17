"""Application configuration.

All settings are environment-driven (12-factor). ``get_settings`` is cached so the
object is constructed once per process and can be injected as a FastAPI dependency.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Campaign Group Predictor API"
    api_version: str = "1.0.0"
    log_level: str = "INFO"

    #: Path to the serialised sklearn pipeline produced by ``src.training.train``.
    model_path: str = "artifacts/model.pkl"

    #: Name of the predictor implementation registered in ``src.factory``.
    model_type: str = "sklearn_pipeline"

    #: When True, a missing or unreadable artifact degrades to the majority-class baseline
    #: instead of preventing the service from starting.
    #:
    #: **Defaults to False: fail closed.** This default was flipped after the deployed
    #: service was found serving the baseline for three consecutive "successful" deploys.
    #: The cause was a malformed `--set-env-vars` argument that folded two variables into
    #: one, so `MODEL_PATH` pointed at a path that did not exist. With a permissive default
    #: the service started anyway, answered its health probe, and returned confident
    #: predictions from a majority-class rule - while every number in the report described a
    #: random forest that was not loaded.
    #:
    #: A fallback that engages by default converts a loud failure into a quiet wrong answer,
    #: and a quiet wrong answer in a system that allocates budget is the worse outcome. The
    #: permissive behaviour is still available, but it now has to be asked for: CI and local
    #: development set `ALLOW_BASELINE_FALLBACK=true` explicitly, because neither has a
    #: trained model and both want the service to start regardless.
    allow_baseline_fallback: bool = False

    #: Base URL the Streamlit frontend uses to reach the backend.
    api_base_url: str = "http://localhost:8000"

    #: Comma-separated CORS origins. "*" is a development convenience only - set this to
    #: the deployed frontend URL in production. Credentials are disabled while it is "*",
    #: because wildcard-plus-credentials is invalid per the CORS specification.
    allowed_origins: str = "*"

    # ------------------------------------------------------------------ decisions
    #: Attach the cost-sensitive recommendation to prediction responses.
    enable_decision_layer: bool = True

    #: Average cost of running one campaign, in `decision_currency`.
    campaign_spend: float = 1.0

    #: Average net profit when the correct group is targeted.
    profit_if_correct: float = 1.0

    #: Fraction of the profit charged for declining a campaign that would have paid off.
    opportunity_weight: float = 0.5

    #: Expected-cost gap below which a decision is flagged for human review.
    decision_review_margin: float = 0.05

    #: Minimum predicted probability required to decide a campaign automatically. Below
    #: this, the response is flagged `review_required` regardless of the cost margin.
    #:
    #: **Defaults to 0.0: disabled.** This is a *fitted parameter*, not a constant. It has
    #: to be selected on a held-out calibration split for the specific deployed model - the
    #: analysis in `notebooks/01_analysis.ipynb` §8.1 does exactly that - and a value
    #: chosen for one champion is meaningless for another. Shipping a hard-coded number
    #: here would silently apply one model's operating point to a different model.
    #:
    #: Set it deliberately from `findings.json["automation_gate"]["threshold"]` after the
    #: analysis, and re-derive it whenever the model is retrained.
    automation_confidence_threshold: float = 0.0

    #: Fraction of "do not run" recommendations overridden so the outcome is still
    #: observed. Without exploration, declining a campaign censors the data that trains the
    #: next model - the policy stops generating evidence about the decisions it makes.
    #:
    #: **Defaults to 0.0: off.** Exploration means deliberately spending budget on
    #: campaigns the model expects to lose money on, which is a business decision with a
    #: real cost, not a modelling default. It also has to be off for the reported figures
    #: to describe the deployed system: training evaluates the decision policy with
    #: exploration disabled, so a non-zero default here would mean the service behaves
    #: differently from everything measured in `metrics.json`.
    #:
    #: Turn it on deliberately, with a rate agreed with the marketing team, and exclude
    #: flagged decisions from performance reporting - the API marks them `exploration: true`
    #: for exactly that purpose.
    exploration_rate: float = 0.0

    decision_currency: str = "EUR"

    def allowed_origins_list(self) -> list[str]:
        """Parse ``allowed_origins`` into a list, preserving the wildcard sentinel."""
        origins = [origin.strip() for origin in self.allowed_origins.split(",")]
        origins = [origin for origin in origins if origin]
        return origins or ["*"]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
