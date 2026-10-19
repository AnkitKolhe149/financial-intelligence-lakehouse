"""Multi-format source publishing.

The same business domain is written to five different shapes because that is the
condition the platform exists to handle. Each format also gets the quirks that
make it awkward in real life, so the ingestion layer has to do actual work:

* **CSV** - a UTF-8 file with a header, quoted free text, and no type
  information whatsoever. Everything arrives as a string.
* **JSON** - a wrapper object with metadata and a nested records array, the way
  a real export endpoint returns it, not a bare list.
* **XML** - elements *and* attributes, with the natural key held as an
  attribute, so the parser cannot just read child text nodes.
* **XLSX** - multiple sheets in one workbook, with the header on row 1 and a
  title row above it on some sheets - the classic finance-team layout.
* **NDJSON** - one event per file in the streaming inbox, which is what a
  file-drop landing zone looks like.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Render a frame as text, the way a file-based export would."""
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return output


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV source file. Nulls are written as empty fields, not 'NaN'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare(frame).to_csv(path, index=False, na_rep="", encoding="utf-8")
    log.info("source_written", format="csv", path=path.name, records=len(frame))
    return path


def write_json(frame: pd.DataFrame, path: Path, *, record_key: str, source_system: str) -> Path:
    """Write a JSON export with an envelope around the records.

    The envelope carries record counts and an extraction timestamp - metadata a
    real export API supplies and that ingestion can reconcile against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(_prepare(frame).to_json(orient="records", date_format="iso"))
    document = {
        "metadata": {
            "source_system": source_system,
            "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "record_count": len(records),
            "schema_version": "1.0",
        },
        record_key: records,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
    log.info("source_written", format="json", path=path.name, records=len(frame))
    return path


def write_xml(
    frame: pd.DataFrame,
    path: Path,
    *,
    root_tag: str,
    record_tag: str,
    id_column: str,
    source_system: str,
) -> Path:
    """Write an XML source file with the natural key held as an attribute.

    Mixing attributes and child elements is deliberate: a parser that only reads
    child text would silently drop the identifier, which is a real and common
    XML ingestion bug.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(frame)

    root = ET.Element(root_tag)
    root.set("sourceSystem", source_system)
    root.set("extractedAt", datetime.now(UTC).isoformat(timespec="seconds"))
    root.set("recordCount", str(len(prepared)))
    root.set("schemaVersion", "1.0")

    for record in prepared.to_dict(orient="records"):
        element = ET.SubElement(root, record_tag)
        identifier = record.get(id_column)
        element.set("id", "" if identifier is None else str(identifier))
        for key, value in record.items():
            if key == id_column:
                continue
            child = ET.SubElement(element, _to_camel(key))
            child.text = "" if value is None or pd.isna(value) else str(value)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    log.info("source_written", format="xml", path=path.name, records=len(frame))
    return path


def _to_camel(snake: str) -> str:
    """``contract_value_usd`` -> ``contractValueUsd``.

    Source systems rarely share our naming convention; camel-cased XML tags
    force the ingestion layer to map names rather than assume them.
    """
    head, *tail = snake.split("_")
    return head + "".join(part.capitalize() for part in tail)


def write_excel(
    sheets: dict[str, pd.DataFrame], path: Path, *, title_rows: set[str] | None = None
) -> Path:
    """Write a multi-sheet workbook.

    Sheets named in ``title_rows`` get a merged title line above the header -
    the layout a finance team actually sends, and the reason the Excel reader
    has to support a configurable header offset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    title_rows = title_rows or set()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            prepared = _prepare(frame)
            offset = 2 if sheet_name in title_rows else 0
            prepared.to_excel(writer, sheet_name=sheet_name, index=False, startrow=offset)
            if offset:
                worksheet = writer.sheets[sheet_name]
                worksheet.cell(row=1, column=1).value = (
                    f"{sheet_name.upper()} EXTRACT - GENERATED "
                    f"{datetime.now(UTC).strftime('%Y-%m-%d')}"
                )
    log.info(
        "source_written",
        format="xlsx",
        path=path.name,
        sheets=list(sheets),
        records=sum(len(f) for f in sheets.values()),
    )
    return path


def write_ndjson(records: list[dict[str, Any]], path: Path) -> Path:
    """Write newline-delimited JSON (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    log.info("source_written", format="ndjson", path=path.name, records=len(records))
    return path


def write_streaming_batches(
    events: list[dict[str, Any]], directory: Path, batch_size: int = 25
) -> list[Path]:
    """Split events into small files in the streaming inbox.

    A landing zone receives many small files, not one big one; the streaming
    pipeline's micro-batching and the small-file problem both depend on that
    shape being real.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    for index in range(0, len(events), batch_size):
        chunk = events[index : index + batch_size]
        path = directory / f"events-{stamp}-{index // batch_size:05d}.json"
        write_ndjson(chunk, path)
        written.append(path)
    return written
