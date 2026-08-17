"""Logging setup.

Two formats, chosen by environment rather than by preference.

**Locally**, a human reads the logs, so lines are plain text and aligned.

**On Cloud Run**, nothing human reads them first. Cloud Logging ingests stdout and parses
any line that is valid JSON into structured fields; anything else becomes an opaque
``textPayload`` string. The difference is not cosmetic:

* a JSON ``severity`` field drives log-level filtering and alerting policies. Without it
  every line is INFO, so an alert on errors cannot be written at all;
* extra keys become queryable — ``jsonPayload.request_id="abc123"`` retrieves every line
  from one request, which is what makes a correlation ID useful rather than decorative;
* ``logging.googleapis.com/trace`` links a log line to a Cloud Trace span, so latency and
  logs are one view rather than two.

Emitting text and then writing regex-based log parsers is the alternative, and it is the
thing this avoids.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Python level names are not the strings Cloud Logging recognises. WARNING happens to
#: match; the rest do not, and an unrecognised value silently degrades to DEFAULT.
_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}

#: Attributes present on every LogRecord. Anything outside this set was attached by the
#: caller via `extra=` and is therefore worth promoting to a queryable field.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class CloudLoggingFormatter(logging.Formatter):
    """Render a record as the single-line JSON object Cloud Logging expects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": _SEVERITY.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            # Included so an error can be traced to a line without attaching a debugger.
            "source": f"{record.module}:{record.funcName}:{record.lineno}",
        }

        # Anything passed as `logger.info("...", extra={"request_id": ...})` becomes a
        # top-level queryable field rather than being interpolated into the message.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # default=str rather than letting json.dumps raise: a log call must never be the
        # thing that breaks a request. An unserialisable value is worth less than the line
        # it appears in.
        return json.dumps(payload, default=str)


def _use_json_logs() -> bool:
    """JSON on Cloud Run, text locally, overridable either way.

    ``K_SERVICE`` is injected by the Cloud Run runtime, so the default is correct in both
    environments without anyone having to remember to set anything.
    """
    override = os.getenv("JSON_LOGS")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes"}
    return bool(os.getenv("K_SERVICE"))


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once, writing to stdout (required by Cloud Run)."""
    formatter: logging.Formatter = (
        CloudLoggingFormatter() if _use_json_logs() else logging.Formatter(_TEXT_FORMAT)
    )

    root = logging.getLogger()
    if root.handlers:
        # Already configured - by uvicorn, by a previous call, or by pytest. Re-apply the
        # formatter anyway: uvicorn installs its own, and leaving it in place would mean
        # the application's own lines are structured while the server's are not.
        for handler in root.handlers:
            handler.setFormatter(formatter)
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
