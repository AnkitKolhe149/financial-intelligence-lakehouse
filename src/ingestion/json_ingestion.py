"""JSON ingestion.

Handles the three shapes JSON exports actually arrive in:

1. an **enveloped document** - ``{"metadata": {...}, "accounts": [...]}``
2. a **bare array** - ``[{...}, {...}]``
3. **newline-delimited JSON** - one object per line, the streaming inbox format

The envelope's declared ``record_count`` is reconciled against the number of
records actually parsed. A mismatch means a truncated file, which is a source
error worth surfacing rather than a data-quality issue to discover three layers
later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging import get_logger
from src.ingestion.base import IngestionResult, SourceReader, discover_files

log = get_logger(__name__)


class JsonSourceReader(SourceReader):
    """Read JSON / NDJSON sources into canonical Bronze frames."""

    source_format = "json"

    def __init__(
        self,
        dataset: str,
        source_system: str,
        *,
        run_id: str,
        directory: Path | None = None,
        files: list[Path] | None = None,
        pattern: str = "*.json",
        record_key: str | None = None,
        column_map: dict[str, str] | None = None,
        ndjson: bool = False,
    ) -> None:
        super().__init__(dataset, source_system, run_id=run_id, column_map=column_map)
        if directory is None and not files:
            raise ValueError("JsonSourceReader needs either a directory or an explicit file list")
        self.directory = directory
        self.files = files
        self.pattern = pattern
        self.record_key = record_key
        self.ndjson = ndjson

    def resolve_files(self) -> list[Path]:
        if self.files:
            return [Path(f) for f in self.files]
        return discover_files(Path(self.directory), self.pattern)  # type: ignore[arg-type]

    def read(self) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        for path in self.resolve_files():
            try:
                results.append(self._read_one(path))
            except Exception as exc:
                log.error("json_file_failed", path=str(path), error=str(exc))
                results.append(
                    IngestionResult(
                        dataset=self.dataset,
                        source_system=self.source_system,
                        source_format=self.source_format,
                        source_file=path.name,
                        frame=pd.DataFrame(),
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                )
        return results

    def _read_one(self, path: Path) -> IngestionResult:
        errors: list[str] = []
        if self.ndjson or path.suffix in {".ndjson", ".jsonl"}:
            records, line_errors = _read_ndjson(path)
            errors.extend(line_errors)
            declared: int | None = None
            source_system = self.source_system
        else:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            records, declared, envelope_system = _extract_records(document, self.record_key)
            source_system = envelope_system or self.source_system

        frame = pd.DataFrame(records)
        # Preserve source fidelity: JSON's native types would let a numeric
        # amount bypass the string-parsing path the CSV feed goes through, so
        # both feeds behave identically in Silver.
        for column in frame.columns:
            if frame[column].dtype != object:
                frame[column] = frame[column].astype("string")

        if declared is not None and declared != len(records):
            message = f"envelope declared {declared} records, parsed {len(records)}"
            errors.append(message)
            log.warning("json_record_count_mismatch", path=path.name, detail=message)

        self.source_system = source_system
        return self.finalise(
            frame,
            source_file=path.name,
            errors=errors,
            metrics={"declared_records": declared, "bytes": path.stat().st_size},
        )


def _extract_records(
    document: Any, record_key: str | None
) -> tuple[list[dict[str, Any]], int | None, str | None]:
    """Pull the record array out of whatever shape the document has."""
    if isinstance(document, list):
        return document, None, None
    if not isinstance(document, dict):
        raise ValueError(f"Unsupported JSON root type: {type(document).__name__}")

    metadata = document.get("metadata") or {}
    declared = metadata.get("record_count") if isinstance(metadata, dict) else None
    source_system = metadata.get("source_system") if isinstance(metadata, dict) else None

    if record_key and record_key in document:
        records = document[record_key]
    else:
        # Fall back to the first list-valued key - handles exports whose array
        # name we were not told in advance.
        records = next((value for value in document.values() if isinstance(value, list)), [])
    if not isinstance(records, list):
        raise ValueError("Resolved record container is not a list")
    return records, declared, source_system


def _read_ndjson(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read newline-delimited JSON, skipping and counting unparseable lines."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc.msg}")
    return records, errors
