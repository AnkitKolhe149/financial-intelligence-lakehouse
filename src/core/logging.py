"""Structured logging.

Every log line carries machine-readable context (``pipeline``, ``run_id``,
``stage``, ``records``, ``duration``) rather than a prose sentence, because the
same lines are what you would ship to a log aggregator in production and query
by ``run_id`` when a nightly load misbehaves.

Two renderers are supported:

* ``console`` - colourised key=value, for humans running the demo.
* ``json``    - one JSON object per line, for log shipping.

Selected with ``FIL_LOG_FORMAT``.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

_CONFIGURED = False

# Keys that must never reach a log sink, regardless of what a caller binds.
_REDACTED_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "dsn",
    "neo4j_password",
    "postgres_dsn",
    "kaggle_key",
}
_REDACTED_PLACEHOLDER = "***redacted***"


def _redact_secrets(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Processor that scrubs credential-looking keys from every event.

    Defence in depth: the platform never logs secrets deliberately, but a
    ``logger.info("connecting", **config)`` a year from now should not be able
    to leak one either.
    """
    for key in list(event_dict):
        if key.lower() in _REDACTED_KEYS:
            event_dict[key] = _REDACTED_PLACEHOLDER
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog + stdlib logging. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )
    # Third-party libraries are chatty; keep their INFO noise out of demo output.
    for noisy in ("py4j", "urllib3", "neo4j", "botocore", "matplotlib", "faker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            _redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Return a logger, configuring logging on first use.

    ``initial_context`` is bound to every line the logger emits - typically
    ``pipeline=`` and ``run_id=``.
    """
    if not _CONFIGURED:
        # Import locally to avoid a circular import at module load time.
        from src.core.config import get_settings

        try:
            settings = get_settings()
            configure_logging(settings.log_level, settings.log_format)
        except Exception:  # pragma: no cover - configuration errors log plainly
            configure_logging()

    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_run_context(**context: Any) -> None:
    """Bind context for the remainder of the current execution.

    Uses contextvars, so a value bound at the top of a pipeline appears on every
    line emitted by every module below it without being threaded through calls.
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()


@contextmanager
def log_stage(
    logger: structlog.stdlib.BoundLogger, stage: str, **context: Any
) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage and emit start/finish (or failure) lines.

    Yields a mutable dict; anything the body puts in it is included in the
    completion line. That is how record counts reach the logs::

        with log_stage(log, "silver") as ctx:
            df = transform(...)
            ctx["records"] = len(df)
    """
    logger.info("stage_started", stage=stage, **context)
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    try:
        yield payload
    except Exception as exc:
        duration = round(time.perf_counter() - started, 3)
        logger.error(
            "stage_failed",
            stage=stage,
            duration_seconds=duration,
            error=str(exc),
            error_type=type(exc).__name__,
            **context,
            **payload,
        )
        raise
    duration = round(time.perf_counter() - started, 3)
    logger.info("stage_completed", stage=stage, duration_seconds=duration, **context, **payload)
