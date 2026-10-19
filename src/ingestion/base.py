"""Reader contract and Bronze audit stamping.

Provenance is attached here, at the moment of reading, because it is the only
moment the information exists. Once a frame is concatenated with three other
sources, nobody can reconstruct which row came from which file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.hashing import record_hash
from src.core.logging import get_logger
from src.core.schema import get_schema

log = get_logger(__name__)


@dataclass
class IngestionResult:
    """What every reader returns, whatever it read."""

    dataset: str
    source_system: str
    source_format: str
    source_file: str
    frame: pd.DataFrame
    schema_version: str = "1.0"
    #: Source-level problems that did not stop the read (a skipped bad file, an
    #: abandoned API page). Record-level problems are not errors - they are data.
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def record_count(self) -> int:
        return len(self.frame)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_system": self.source_system,
            "source_format": self.source_format,
            "source_file": self.source_file,
            "records": self.record_count,
            "errors": len(self.errors),
            **self.metrics,
        }


class SourceReader(ABC):
    """Base class for every source reader.

    Subclasses implement :meth:`read`. Everything shared - column mapping,
    audit stamping, canonical column alignment - lives here so that adding a
    sixth source format is a small class, not a copy of an existing one.
    """

    source_format: str = "unknown"

    def __init__(
        self,
        dataset: str,
        source_system: str,
        *,
        run_id: str,
        column_map: dict[str, str] | None = None,
        schema_version: str = "1.0",
    ) -> None:
        self.dataset = dataset
        self.source_system = source_system
        self.run_id = run_id
        self.column_map = column_map or {}
        self.schema_version = schema_version
        self.schema = get_schema(dataset)

    @abstractmethod
    def read(self) -> list[IngestionResult]:
        """Read the source and return one result per physical source unit."""

    # -- shared helpers -----------------------------------------------------

    def normalise_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename source columns onto canonical names.

        Applies the explicit ``column_map`` first, then a generic
        camelCase/PascalCase/spaced -> snake_case fallback. Unmapped columns are
        **kept**: Bronze preserves fields we do not yet understand, which is what
        makes additive schema evolution a non-event.
        """
        renamed = frame.rename(columns=self.column_map)
        fallback = {
            column: _to_snake(column)
            for column in renamed.columns
            if column not in self.schema.column_names and _to_snake(column) != column
        }
        if fallback:
            renamed = renamed.rename(columns=fallback)
        return renamed

    def finalise(
        self,
        frame: pd.DataFrame,
        source_file: str,
        *,
        schema_version: str | None = None,
        errors: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Apply column mapping and audit stamping, then wrap in a result."""
        mapped = self.normalise_columns(frame)
        stamped = stamp_audit_columns(
            mapped,
            dataset=self.dataset,
            source_system=self.source_system,
            source_format=self.source_format,
            source_file=source_file,
            run_id=self.run_id,
            schema_version=schema_version or self.schema_version,
        )
        result = IngestionResult(
            dataset=self.dataset,
            source_system=self.source_system,
            source_format=self.source_format,
            source_file=source_file,
            frame=stamped,
            schema_version=schema_version or self.schema_version,
            errors=errors or [],
            metrics=metrics or {},
        )
        log.info("source_read", **result.summary())
        return result


def stamp_audit_columns(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_system: str,
    source_format: str,
    source_file: str,
    run_id: str,
    schema_version: str = "1.0",
) -> pd.DataFrame:
    """Attach the Bronze audit columns to every row.

    ``_record_hash`` is computed over the canonical hash columns declared in the
    schema, using only the columns actually present. That last detail matters:
    a v1.0 file lacking ``merchant_category`` must still produce the same hash as
    a v1.1 file with the same business content, or schema evolution would create
    phantom duplicates.
    """
    stamped = frame.copy()
    ingestion_time = datetime.now(UTC)

    schema = get_schema(dataset)
    hash_columns = [c for c in schema.effective_hash_columns() if c in stamped.columns]
    if hash_columns:
        subset = stamped[hash_columns]
        stamped["_record_hash"] = [
            record_hash(row, hash_columns) for row in subset.to_dict(orient="records")
        ]
    else:
        stamped["_record_hash"] = None

    stamped["_source_system"] = source_system
    stamped["_source_file"] = source_file
    stamped["_source_format"] = source_format
    stamped["_ingestion_timestamp"] = ingestion_time
    stamped["_ingestion_run_id"] = run_id
    stamped["_schema_version"] = schema_version
    stamped["_source_row_number"] = range(1, len(stamped) + 1)
    # Partition key for Bronze. Ingestion date, not transaction date: Bronze is
    # organised by when we received data, which is what makes "replay everything
    # we loaded on the 14th" a cheap operation.
    stamped["ingestion_date"] = ingestion_time.date().isoformat()
    return stamped


def _to_snake(name: str) -> str:
    """``contractValueUsd`` / ``Contract Value`` -> ``contract_value_usd``."""
    text = str(name).strip().replace("-", " ").replace(".", " ")
    if " " in text:
        return "_".join(part.lower() for part in text.split())
    out: list[str] = []
    for index, char in enumerate(text):
        if char.isupper() and index > 0 and not text[index - 1].isupper():
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def discover_files(directory: Path, pattern: str) -> list[Path]:
    """List matching source files in a stable, reproducible order.

    Sorted because unordered file discovery makes runs non-reproducible: with
    ``_source_row_number`` derived from read order, an OS-dependent glob order
    would change record hashes between machines.
    """
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(pattern) if p.is_file())
