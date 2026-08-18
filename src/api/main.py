"""Application entry point.

Design pattern: **Application Factory**. ``create_app`` builds a fully configured
instance, which lets tests spin up isolated apps instead of importing global state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
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

    # Request validation, handled explicitly because FastAPI's default cannot serialise the
    # one input this schema most needs to reject.
    #
    # `schemas.py` refuses NaN and infinity - they are indistinguishable from a missing value
    # downstream and usually mean an upstream computation failed. But the default handler
    # builds a 422 body that echoes the offending value, and Starlette encodes responses with
    # `allow_nan=False`. Encoding `nan` therefore raises inside the error handler, the
    # catch-all converts that to a **500**, and the API answers a malformed request with
    # "internal server error" - blaming itself for the caller's payload, and returning the
    # wrong status code for a documented rejection.
    #
    # The validator was correct and untested; the bug lived in the path that reports it.
    #
    # This handler also stops echoing caller input altogether. The field path and the reason
    # are what a caller needs to fix the request; replaying their own payload back to them
    # adds nothing and widens what an unauthenticated probe can extract.
    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        categories: list[str] = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            problems.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
            categories.append(str(error.get("type", "unknown")))

        detail = "; ".join(problems) or "Request failed validation."

        # The failure *category* is a field, because the useful question about 422s is which
        # kind is spiking. A rise in `missing` means a caller changed their payload; a rise in
        # `value_error` from the NaN guard means something upstream is computing badly. Both
        # look identical in a count of 422s, and they need different people.
        logger.warning(
            "Request validation failed: %s",
            detail,
            extra={
                "validation_failure_categories": sorted(set(categories)),
                "validation_error_count": len(problems),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(error="RequestValidationError", detail=detail).model_dump(),
        )

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


def _run_startup_canary(predictor: Any, settings: Settings) -> None:
    """Score one synthetic campaign through the full pipeline before accepting traffic.

    The payload is built from ``src.constants`` rather than hardcoded, so it exercises the
    real 67-column contract. If a feature is added, the canary carries it automatically
    instead of silently testing an outdated shape.

    Failure behaviour depends on what is loaded, and the asymmetry is deliberate:

    * A **real artifact** that cannot predict is a broken deployment. Raise, so the container
      never becomes ready and Cloud Run keeps the previous revision serving traffic.
    * The **baseline fallback** failing is logged, not raised. The fallback exists precisely
      for environments with no model - CI, local development - and crashing the container
      there would break the case it was built to serve.
    """
    import pandas as pd

    from src.constants import BASE_FEATURES

    is_baseline = bool(predictor.metadata.get("is_baseline", False))
    probe = pd.DataFrame([dict.fromkeys(BASE_FEATURES, 0.5)])

    try:
        results = predictor.predict(probe)
        if not results or results[0].predicted_class not in (0, 1, 2):
            raise ValueError(f"Canary returned an unusable result: {results!r}")
    except Exception as error:
        if is_baseline:
            logger.warning(
                "Startup canary failed on the baseline predictor (%s). Continuing, because "
                "the baseline is what runs where no artifact exists.",
                error,
            )
            return
        logger.error("Startup canary FAILED: %s", error, exc_info=True)
        raise RuntimeError(
            "The model artifact loaded but could not produce a prediction, so this instance "
            "would fail on its first real request. Refusing to become ready. Original "
            f"error: {error}"
        ) from error

    logger.info(
        "Startup canary passed: predicted class %s with confidence %.3f.",
        results[0].predicted_class,
        results[0].confidence,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a configured FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Load the model once at startup, prove it works, then serve.

        The canary below is the point. Loading an artifact and *being able to predict with
        it* are different properties, and only the second one matters. A pickle can
        deserialise cleanly and still fail on the first request - a feature-name mismatch, a
        transformer expecting a column the schema no longer sends, an estimator built against
        a different library version.

        Without the canary that failure surfaces on a real caller's request, after the
        container has already passed its readiness probe and been given traffic. With it, the
        instance refuses to start, Cloud Run keeps the previous revision serving, and the
        failure lands on the deploy rather than on a campaign manager.
        """
        registry = ModelRegistry()
        predictor = registry.load(settings)

        _run_startup_canary(predictor, settings)

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

    # Reject oversized bodies before parsing them.
    #
    # `BatchComparisonRequest` caps the batch at 1,000 comparisons, but that limit is
    # enforced by Pydantic *after* the body has been read and deserialised. A caller sending
    # a 500 MB payload therefore consumes memory and CPU on a 1-vCPU instance before being
    # told the batch is too large - the validation is correct and arrives too late to be a
    # defence.
    #
    # 8 MB is generous for the legitimate maximum: 1,000 comparisons x 67 floats is roughly
    # 1.5 MB of JSON. The limit exists to bound the worst case, not to constrain real use.
    #
    # Content-Length can be absent or wrong on a chunked request, so this is a cheap first
    # filter rather than a complete guard - the real ceiling for an internet-facing service
    # belongs at the load balancer, where it can be applied before the request reaches the
    # application at all.
    max_body_bytes = 8 * 1024 * 1024

    @app.middleware("http")
    async def _limit_body_size(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_body_bytes:
            logger.warning(
                "Rejected an oversized request body: %s bytes on %s",
                declared,
                request.url.path,
                extra={"http_path": request.url.path, "content_length": int(declared)},
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=ErrorResponse(
                    error="PayloadTooLarge",
                    detail=(
                        f"Request body of {int(declared):,} bytes exceeds the "
                        f"{max_body_bytes:,}-byte limit. Send at most 1,000 comparisons "
                        "per batch."
                    ),
                ).model_dump(),
            )
        return await call_next(request)

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
