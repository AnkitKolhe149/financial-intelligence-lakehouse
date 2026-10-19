"""Cross-cutting primitives shared by every layer of the platform.

Nothing in :mod:`src.core` may import from a pipeline layer - the dependency
arrow points one way (layers -> core) so that core stays testable in isolation.
"""

from src.core.config import Settings, get_settings, project_root
from src.core.exceptions import (
    ConfigurationError,
    DataQualityError,
    IngestionError,
    LakehouseError,
    SchemaValidationError,
)
from src.core.hashing import deterministic_uuid, record_hash, stable_hash
from src.core.logging import configure_logging, get_logger
from src.core.run_context import RunContext, new_run_id

__all__ = [
    "ConfigurationError",
    "DataQualityError",
    "IngestionError",
    "LakehouseError",
    "RunContext",
    "SchemaValidationError",
    "Settings",
    "configure_logging",
    "deterministic_uuid",
    "get_logger",
    "get_settings",
    "new_run_id",
    "project_root",
    "record_hash",
    "stable_hash",
]
