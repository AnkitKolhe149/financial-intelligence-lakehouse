"""XML ingestion.

Parsed with :mod:`defusedxml`, not the standard library parser. XML from an
external system is untrusted input, and the stock parser is vulnerable to entity
expansion attacks ("billion laughs") and external entity resolution. Using the
hardened parser costs one import and removes a real class of vulnerability.

The reader handles the two ways XML carries data - **attributes** on the element
and **child text nodes** - and merges them into one flat record, with attributes
taking precedence when a name appears in both. It also converts the camelCase
tag names that XML sources favour into canonical snake_case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from defusedxml import ElementTree as SafeET

from src.core.logging import get_logger
from src.ingestion.base import IngestionResult, SourceReader, _to_snake, discover_files

log = get_logger(__name__)


class XmlSourceReader(SourceReader):
    """Read XML documents into canonical Bronze frames."""

    source_format = "xml"

    def __init__(
        self,
        dataset: str,
        source_system: str,
        *,
        run_id: str,
        record_tag: str,
        directory: Path | None = None,
        files: list[Path] | None = None,
        pattern: str = "*.xml",
        id_attribute: str = "id",
        id_column: str | None = None,
        column_map: dict[str, str] | None = None,
    ) -> None:
        super().__init__(dataset, source_system, run_id=run_id, column_map=column_map)
        if directory is None and not files:
            raise ValueError("XmlSourceReader needs either a directory or an explicit file list")
        self.directory = directory
        self.files = files
        self.pattern = pattern
        self.record_tag = record_tag
        self.id_attribute = id_attribute
        #: Canonical column the element's id attribute maps onto.
        self.id_column = id_column or f"{dataset}_id"

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
                log.error("xml_file_failed", path=str(path), error=str(exc))
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
        tree = SafeET.parse(str(path))
        root = tree.getroot()

        declared = root.attrib.get("recordCount")
        envelope_system = root.attrib.get("sourceSystem")
        schema_version = root.attrib.get("schemaVersion", self.schema_version)

        records: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, element in enumerate(root.iter(self.record_tag), start=1):
            try:
                records.append(self._element_to_record(element))
            except Exception as exc:
                errors.append(f"record {index}: {exc}")

        frame = pd.DataFrame(records)

        if declared is not None and str(declared).isdigit() and int(declared) != len(records):
            message = f"root declared {declared} records, parsed {len(records)}"
            errors.append(message)
            log.warning("xml_record_count_mismatch", path=path.name, detail=message)

        if envelope_system:
            self.source_system = envelope_system

        return self.finalise(
            frame,
            source_file=path.name,
            schema_version=schema_version,
            errors=errors,
            metrics={"declared_records": declared, "bytes": path.stat().st_size},
        )

    def _element_to_record(self, element: Any) -> dict[str, Any]:
        """Flatten one record element into a dict.

        Child elements that themselves have children are serialised depth-first
        with dotted keys (``address.city``), so nested structures survive into
        Bronze instead of being dropped.
        """
        record: dict[str, Any] = {}

        for key, value in element.attrib.items():
            column = self.id_column if key == self.id_attribute else _to_snake(key)
            record[column] = value if value != "" else None

        for child in element:
            self._flatten(child, record, prefix="")

        return record

    def _flatten(self, element: Any, record: dict[str, Any], prefix: str) -> None:
        name = _to_snake(element.tag)
        key = f"{prefix}{name}"
        children = list(element)
        if children:
            for child in children:
                self._flatten(child, record, prefix=f"{key}.")
            return
        text = (element.text or "").strip()
        record.setdefault(key, text if text else None)
        for attr_key, attr_value in element.attrib.items():
            record.setdefault(f"{key}.{_to_snake(attr_key)}", attr_value or None)
