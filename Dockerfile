# syntax=docker/dockerfile:1
# Backend prediction API - deployed to Google Cloud Run.
#
# 3.12 is not arbitrary: artifacts/model.pkl was produced on Python 3.12.4, and this image
# deserialises it. A pickle carries references to the classes that created it, and while
# unpickling across minor Python versions usually works, "usually" is not a property to rely
# on for the step between a healthy container and a served prediction.
#
# Serving on 3.11 while training on 3.12 was the state before this change. It worked, and it
# was luck rather than design. The versions in requirements-serve.txt are pinned for the
# same reason and must move together with this line.
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
#
# This uses requirements-serve.txt, not requirements.txt. The latter is the *development*
# environment - JupyterLab, Streamlit, matplotlib, pytest, black, ruff - none of which
# serve a prediction. Installing them here would add roughly 2 GB, slow the Cloud Run cold
# start (the image is pulled before the first request is answered) and widen the CVE
# surface for packages the process never imports.
#
# The list is dictated by the champion artifact: a scikit-learn pipeline needs
# scikit-learn, numpy, scipy, joblib and pandas. If a future `--champion` selects a
# CatBoost/LightGBM/XGBoost model, its library must be added to requirements-serve.txt -
# a boosted pipeline cannot be unpickled without it, and the failure surfaces on the first
# request rather than at build time. See the header of that file.
COPY requirements-serve.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-serve.txt

COPY src ./src
COPY artifacts ./artifacts

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Cloud Run injects $PORT; the shell form expands it.
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
