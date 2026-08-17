"""Application entry point.

Design pattern: **Application Factory**. ``create_app`` builds a fully configured
instance, which lets tests spin up isolated apps instead of importing global state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config import Settings, get_settings
from src.exceptions import (
    InvalidFeaturePayloadError,
    ModelArtifactError,
    ModelNotLoadedError,
    PredictorNotRegisteredError,
)
from src.logging_config import configure_logging, get_logger
from src.registry import ModelRegistry
from src.schemas import ErrorResponse

logger = get_logger(__name__)


def _register_exception_handlers(app: FastAPI) -> None:
    """Translate domain exceptions into consistent HTTP responses."""

    def _envelope(exc: Exception, code: int) -> JSONResponse:
        payload = ErrorResponse(error=type(exc).__name__, detail=str(exc))
        return JSONResponse(status_code=code, content=payload.model_dump())

    @app.exception_handler(InvalidFeaturePayloadError)
    async def _invalid_payload(_: Request, exc: InvalidFeaturePayloadError) -> JSONResponse:
        logger.warning("Invalid payload: %s", exc)
        return _envelope(exc, status.HTTP_422_UNPROCESSABLE_ENTITY)

    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
        logger.error("Model not loaded: %s", exc)
        return _envelope(exc, status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(ModelArtifactError)
    async def _artifact_error(_: Request, exc: ModelArtifactError) -> JSONResponse:
        logger.error("Artifact error: %s", exc)
        return _envelope(exc, status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(PredictorNotRegisteredError)
    async def _unknown_predictor(_: Request, exc: PredictorNotRegisteredError) -> JSONResponse:
        logger.error("Unknown predictor: %s", exc)
        return _envelope(exc, status.HTTP_500_INTERNAL_SERVER_ERROR)

    # The catch-all, and the only handler that deliberately does NOT echo the exception.
    #
    # The four handlers above translate *anticipated* failures, and their messages are safe
    # because we wrote them. This one catches everything else - a numpy overflow, a shape
    # mismatch, a corrupt artifact - and those messages are written by libraries. They can
    # carry file paths, column names and array internals. Returning `str(exc)` to an
    # unauthenticated caller turns an incident into disclosure.
    #
    # So the response is deliberately opaque and the diagnosis goes to the log, joined by
    # `request_id`. A caller who reports "request abc123 failed" can be answered exactly;
    # a caller probing the service learns nothing.
    #
    # `exc_info=True` records the traceback in the structured log entry. Without it this
    # handler would swallow the one piece of information that makes the failure fixable.
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "Unhandled %s on %s %s",
            type(exc).__name__,
            request.method,
            request.url.path,
            exc_info=True,
            extra={
                "request_id": request_id,
                "http_path": request.url.path,
                "exception_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                detail=(
                    "The request could not be completed. Quote request "
                    f"{request_id} when reporting this."
                ),
            ).model_dump(),
            # The header is set HERE, not by the correlation-ID middleware, and that is not
            # duplication.
            #
            # Starlette's ServerErrorMiddleware - which invokes this handler - sits *outside*
            # the user middleware stack. When an exception propagates, `_attach_request_id`
            # never resumes after `call_next`, so its header assignment never runs. The
            # result is that X-Request-ID is present on every successful response and absent
            # on exactly the responses where a caller needs it to report a fault.
            #
            # Found by a test asserting the header on a 500 rather than on a 200.
            headers={"X-Request-ID": request_id},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a configured FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Load the model once at startup and release it on shutdown."""
        registry = ModelRegistry()
        predictor = registry.load(settings)
        logger.info("Startup complete. Active predictor: %s", predictor.metadata.get("predictor"))
        yield
        registry.reset()
        logger.info("Shutdown complete.")

    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description=(
            "Predicts which of two customer groups should be targeted by a marketing "
            "campaign. Classes: 0 = neither group profitable, 1 = group 1, 2 = group 2."
        ),
        lifespan=lifespan,
    )

    # Single source of configuration for every dependency (see src/api/dependencies.py).
    app.state.settings = settings

    # CORS. A wildcard origin combined with credentials is both invalid per the CORS
    # specification (browsers reject it) and a poor default for a service that authorises
    # campaign spend. Credentials are therefore only enabled when explicit origins are
    # configured; the permissive default is for local development only.
    allowed_origins = settings.allowed_origins_list()
    wildcard = allowed_origins == ["*"]
    if wildcard:
        logger.warning(
            "CORS is configured to allow any origin. Set ALLOWED_ORIGINS to the frontend "
            "URL before exposing this service publicly."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=not wildcard,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Correlation ID. Every response carries X-Request-ID, and the same value is bound to
    # the log record for the request that produced it.
    #
    # This service recommends how to spend campaign budget. When a campaign manager asks
    # six weeks later "why did it tell us to target group 2?", the answer has to be
    # recoverable - which model version, which probabilities, which decision rule. Logs
    # that record *what* was decided but not *which request decided it* cannot answer that.
    # An inbound X-Request-ID is honoured so the ID survives across the frontend, the load
    # balancer and this service rather than being regenerated at each hop.
    @app.middleware("http")
    async def _attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id
        start = perf_counter()

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        # `extra` rather than string interpolation: on Cloud Run these become top-level
        # fields in the structured log entry, so `jsonPayload.request_id="..."` retrieves
        # every line from one request and latency can be aggregated without parsing text.
        logger.info(
            "%s %s -> %d in %.1f ms",
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - start) * 1000,
            extra={
                "request_id": request_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status": response.status_code,
                "latency_ms": round((perf_counter() - start) * 1000, 2),
            },
        )
        return response

    _register_exception_handlers(app)
    app.include_router(router)

    @app.get("/", tags=["operations"], summary="Service banner")
    def root() -> dict[str, str]:
        """Return a small banner pointing at the interactive docs."""
        return {
            "service": settings.app_name,
            "version": settings.api_version,
            "docs": "/docs",
        }

    return app


#: ASGI entry point used by uvicorn: ``uvicorn src.api.main:app``
app = create_app()
