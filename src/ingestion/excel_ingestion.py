"""Excel ingestion.

Spreadsheets from finance teams are rarely clean rectangles. This reader handles
the two habits that break naive ``read_excel`` calls:

* **Title rows above the header.** The header is located by scanning the first
  few rows for the one that looks like a header (mostly non-empty, mostly
  unique, mostly text) instead of assuming row 0.
* **Multiple sheets carrying different datasets** in one workbook, each mapping
  to a different canonical dataset.

As with CSV, cells are read as strings so that Excel's helpful auto-typing (dates
as floats, ids losing leading zeros) does not corrupt values before Silver sees
them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.logging import get_logger
from src.ingestion.base import IngestionResult, SourceReader, discover_files

log = get_logger(__name__)

MAX_HEADER_SCAN_ROWS = 8


class ExcelSourceReader(SourceReader):
    """Read one sheet of one or more workbooks into a canonical Bronze frame."""

    source_format = "xlsx"

    def __init__(
        self,
        dataset: str,
        source_system: str,
        *,
        run_id: str,
        sheet_name: str,
        directory: Path | None = None,
        files: list[Path] | None = None,
        pattern: str = "*.xlsx",
        column_map: dict[str, str] | None = None,
        header_row: int | None = None,
    ) -> None:
        super().__init__(dataset, source_system, run_id=run_id, column_map=column_map)
        if directory is None and not files:
            raise ValueError("ExcelSourceReader needs either a directory or an explicit file list")
        self.directory = directory
        self.files = files
        self.pattern = pattern
        self.sheet_name = sheet_name
        #: When None, the header row is detected rather than assumed.
        self.header_row = header_row

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
                    "excel_file_failed", path=str(path), sheet=self.sheet_name, error=str(exc)
                )
                results.append(
                    IngestionResult(
                        dataset=self.dataset,
                        source_system=self.source_system,
                        source_format=self.source_format,
                        source_file=f"{path.name}#{self.sheet_name}",
                        frame=pd.DataFrame(),
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                )
        return results

    def _read_one(self, path: Path) -> IngestionResult:
        header_row = (
            self.header_row
            if self.header_row is not None
            else detect_header_row(path, self.sheet_name)
        )
        frame = pd.read_excel(
            path,
            sheet_name=self.sheet_name,
            header=header_row,
            dtype=str,
            engine="openpyxl",
        )
        # Drop the fully-empty spacer columns Excel loves to leave behind.
        frame = frame.dropna(axis=1, how="all")
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]

        return self.finalise(
            frame,
            source_file=f"{path.name}#{self.sheet_name}",
            metrics={"header_row": header_row, "bytes": path.stat().st_size},
        )


def detect_header_row(path: Path, sheet_name: str, max_scan: int = MAX_HEADER_SCAN_ROWS) -> int:
    """Find the header row by scoring the first few rows.

    A header row is mostly populated, mostly unique and mostly non-numeric. The
    first row scoring above the threshold wins; if none does we fall back to row
    0, which is the correct answer for a well-formed sheet.
    """
    probe = pd.read_excel(
        path, sheet_name=sheet_name, header=None, nrows=max_scan, dtype=str, engine="openpyxl"
    )
    best_row, best_score = 0, -1.0

    for index in range(len(probe)):
        values = [
            str(v).strip()
            for v in probe.iloc[index].tolist()
            if str(v).strip() not in {"", "nan", "None"}
        ]
        if len(values) < 2:
            continue
        populated = len(values) / max(probe.shape[1], 1)
        unique = len(set(values)) / len(values)
        textual = sum(not _looks_numeric(v) for v in values) / len(values)
        score = populated * 0.4 + unique * 0.3 + textual * 0.3
        if score > best_score:
            best_row, best_score = index, score

    return best_row if best_score >= 0.7 else 0


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ""))
    except ValueError:
        return False
    return True
