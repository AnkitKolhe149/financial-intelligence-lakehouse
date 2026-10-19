"""Exception hierarchy.

A single root (:class:`LakehouseError`) lets pipeline stages catch "anything our
own code raised" without also swallowing genuine programming errors such as
``AttributeError``. Each subclass carries structured context so that failures
land in the pipeline-run metadata table in a queryable shape rather than as a
free-text blob.
"""

from __future__ import annotations

from typing import Any


class LakehouseError(Exception):
    """Base class for every error raised deliberately by this platform."""

    error_code: str = "LAKEHOUSE_ERROR"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        """Render the error in the shape stored in ``pipeline_runs.error_message``."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        ctx = " ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({ctx})"


class ConfigurationError(LakehouseError):
    """Raised when settings are missing, malformed or mutually inconsistent."""

    error_code = "CONFIGURATION_ERROR"


class IngestionError(LakehouseError):
    """Raised when a source system cannot be read at all.

    Note this is a *source-level* failure. A single unparseable record must not
    raise - it is quarantined instead (see :mod:`src.silver.quarantine`).
    """

    error_code = "INGESTION_ERROR"


class SchemaValidationError(LakehouseError):
    """Raised when a dataset violates its declared contract in ``schemas/``."""

    error_code = "SCHEMA_VALIDATION_ERROR"


class DataQualityError(LakehouseError):
    """Raised when a blocking data-quality gate fails.

    Only checks declared with ``severity: blocking`` raise; everything else is
    recorded and the pipeline continues, because an oversight platform that
    stops on the first warning stops being an oversight platform.
    """

    error_code = "DATA_QUALITY_ERROR"


class EngineError(LakehouseError):
    """Raised when the requested execution engine is unavailable or misused."""

    error_code = "ENGINE_ERROR"


class GraphError(LakehouseError):
    """Raised by the graph layer. Always caught by callers - the graph is an
    enrichment component and must never be a single point of failure."""

    error_code = "GRAPH_ERROR"
