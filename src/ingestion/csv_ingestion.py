"""CSV ingestion.

Everything is read as **string** (``dtype=str``). That is not laziness - it is
the central design decision of the Bronze layer. If pandas infers types at read
time it will silently coerce ``"not-a-date"`` to ``NaT`` and ``"-500"`` to a
valid number, destroying the evidence that the source sent something wrong. By
keeping the raw text, the invalid value survives into quarantine where an
analyst can see exactly what arrived.

Large files are read in chunks so that memory stays bounded regardless of file
size, and malformed lines are skipped and *counted* rather than aborting the
read.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.core.logging import get_logger
from src.ingestion.base import IngestionResult, SourceReader, discover_files

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 100_000


class CsvSourceReader(SourceReader):
    """Read one or more CSV files into canonical Bronze frames."""

    source_format = "csv"

    def __init__(
        self,
        dataset: str,
        source_system: str,
        *,
        run_id: str,
        directory: Path | None = None,
        files: list[Path] | None = None,
        pattern: str = "*.csv",
        column_map: dict[str, str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        delimiter: str = ",",
    ) -> None:
        super().__init__(dataset, source_system, run_id=run_id, column_map=column_map)
        if directory is None and not files:
            raise ValueError("CsvSourceReader needs either a directory or an explicit file list")
        self.directory = directory
        self.files = files
        self.pattern = pattern
        self.chunk_size = chunk_size
        self.delimiter = delimiter

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
                log.error(
                    "csv_file_failed", path=str(path), error=str(exc), error_type=type(exc).__name__
                )
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
        frames: list[pd.DataFrame] = []
        bad_lines = 0

        def on_bad_line(_line: list[str]) -> None:
            nonlocal bad_lines
            bad_lines += 1
            return None

        reader: Iterator[pd.DataFrame] = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            chunksize=self.chunk_size,
            sep=self.delimiter,
            engine="python",  # required for a callable bad-line handler
            on_bad_lines=on_bad_line,
            encoding="utf-8",
        )
        for chunk in reader:
            frames.append(chunk)

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # The schema version is inferred from the columns actually present.
        # This is how the 2025 (v1.0) and 2026 (v1.1) files are told apart
        # without hard-coding filenames.
        schema_version = "1.1" if "merchant_category" in frame.columns else "1.0"

        return self.finalise(
            frame,
            source_file=path.name,
            schema_version=schema_version,
            errors=[f"{bad_lines} malformed lines skipped"] if bad_lines else [],
            metrics={"chunks": len(frames), "bad_lines": bad_lines, "bytes": path.stat().st_size},
        )
