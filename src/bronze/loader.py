"""Bronze loader: ingestion results in, partitioned Bronze datasets out.

The two hard problems handled here are idempotency and schema union.

**Idempotency.** Every source unit carries a content hash. Before writing, the
loader asks the ingestion ledger whether that exact ``(source_file,
content_hash)`` pair has been loaded before; if so the unit is skipped. This is
what makes "re-run last night's job" safe. Note that the hash is over *content*,
not the filename - a file that is re-sent with the same name but new content is
correctly treated as new data, and a file that is renamed but unchanged is
correctly treated as already loaded.

**Schema union.** The 2025 transaction file has 16 columns and the 2026 file has
17. Concatenating them naively would either fail or silently drop the extra
column. The loader unions the schemas, back-filling nulls, and records which
schema version each row came from - additive evolution absorbed without a
migration, which is the entire promise of schema-on-read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.hashing import file_content_hash, stable_hash
from src.core.io import dataset_row_count, dataset_schema, dataset_size_bytes, write_parquet
from src.core.logging import get_logger
from src.core.schema import BRONZE_AUDIT_COLUMNS, get_schema
from src.core.schema import table_name as build_table_name
from src.ingestion.base import IngestionResult
from src.metadata.store import MetadataStore, get_metadata_store

log = get_logger(__name__)


@dataclass
class BronzeLoadResult:
    """Outcome of loading one Bronze table."""

    table: str
    dataset: str
    records_written: int
    records_skipped: int
    sources: list[str] = field(default_factory=list)
    schema_versions: list[str] = field(default_factory=list)
    path: str = ""
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "records_written": self.records_written,
            "records_skipped": self.records_skipped,
            "sources": len(self.sources),
            "schema_versions": sorted(set(self.schema_versions)),
            "errors": len(self.errors),
        }


class BronzeLoader:
    """Writes :class:`IngestionResult` batches into the Bronze layer."""

    def __init__(
        self,
        run_id: str,
        settings: Settings | None = None,
        store: MetadataStore | None = None,
        *,
        force: bool = False,
    ) -> None:
        self.run_id = run_id
        self.settings = settings or get_settings()
        self.store = store if store is not None else get_metadata_store()
        #: When True, previously-loaded files are re-ingested anyway.
        self.force = force

    # -- public API ---------------------------------------------------------

    def load(
        self,
        dataset: str,
        results: list[IngestionResult],
        *,
        table: str | None = None,
        mode: str = "overwrite_partitions",
    ) -> BronzeLoadResult:
        """Write every result for ``dataset`` into its Bronze table."""
        table_name = table or build_table_name("bronze", dataset)
        path = self.settings.dataset_path("bronze", table_name)

        frames: list[pd.DataFrame] = []
        ledger_entries: list[dict[str, Any]] = []
        sources: list[str] = []
        schema_versions: list[str] = []
        errors: list[str] = []
        skipped = 0

        for result in results:
            errors.extend(f"{result.source_file}: {message}" for message in result.errors)
            if result.frame.empty:
                continue

            content_hash = self._content_hash(result)
            if not self.force and self.store.already_ingested(result.source_file, content_hash):
                skipped += result.record_count
                log.info(
                    "bronze_skip_duplicate_source",
                    table=table_name,
                    source_file=result.source_file,
                    records=result.record_count,
                    reason="identical content already ingested",
                )
                continue

            frames.append(result.frame)
            sources.append(result.source_file)
            schema_versions.append(result.schema_version)
            ledger_entries.append(
                {
                    "dataset": dataset,
                    "source_system": result.source_system,
                    "source_file": result.source_file,
                    "content_hash": content_hash,
                    "run_id": self.run_id,
                    "records": result.record_count,
                    "ingested_at": datetime.now(UTC),
                    "schema_version": result.schema_version,
                }
            )

        if not frames:
            log.info("bronze_nothing_to_load", table=table_name, skipped=skipped)
            return BronzeLoadResult(
                table=table_name,
                dataset=dataset,
                records_written=0,
                records_skipped=skipped,
                path=str(path),
                errors=errors,
            )

        combined = self._union(frames, dataset)
        partitions = self.settings.storage.partitions_for(table_name) or ["ingestion_date"]
        written = write_parquet(
            combined,
            path,
            partition_by=partitions,
            mode=mode,
            compression=self.settings.storage.compression,
        )

        self.store.record_ingestion(ledger_entries)
        self._catalog(table_name, path, dataset)

        result = BronzeLoadResult(
            table=table_name,
            dataset=dataset,
            records_written=written,
            records_skipped=skipped,
            sources=sources,
            schema_versions=schema_versions,
            path=str(path),
            errors=errors,
        )
        log.info("bronze_loaded", **result.summary())
        return result

    # -- internals ----------------------------------------------------------

    def _union(self, frames: list[pd.DataFrame], dataset: str) -> pd.DataFrame:
        """Concatenate frames with differing schemas, aligning on the union.

        Column order is stabilised as *canonical columns first, then unexpected
        source columns, then audit columns*. A deterministic order matters
        because Parquet schema comparisons and the Spark reader both care.
        """
        schema = get_schema(dataset)
        all_columns: list[str] = []
        for frame in frames:
            for column in frame.columns:
                if column not in all_columns:
                    all_columns.append(column)

        canonical = [c for c in schema.column_names if c in all_columns]
        audit = [c for c in [*BRONZE_AUDIT_COLUMNS, "ingestion_date"] if c in all_columns]
        extra = [c for c in all_columns if c not in canonical and c not in audit]
        ordered = canonical + extra + audit

        aligned = []
        for frame in frames:
            missing = [c for c in ordered if c not in frame.columns]
            if missing:
                # Absent columns become explicit nulls. This is the schema
                # evolution path: v1.0 rows get a null merchant_category rather
                # than being rejected or dropping the column for everyone.
                filler = pd.DataFrame({c: [None] * len(frame) for c in missing}, index=frame.index)
                frame = pd.concat([frame, filler], axis=1)
            aligned.append(frame[ordered])

        combined = pd.concat(aligned, ignore_index=True)
        # Everything except the ingestion timestamp is stored as text in Bronze:
        # preserving the source's exact bytes is the layer's contract.
        for column in combined.columns:
            if column == "_ingestion_timestamp":
                continue
            if combined[column].dtype == object or str(combined[column].dtype).startswith("str"):
                continue
            combined[column] = combined[column].astype("string")
        return combined

    def _content_hash(self, result: IngestionResult) -> str:
        """Hash a source unit's content.

        For files, the bytes on disk. For API pulls there is no file, so the hash
        is taken over the record hashes actually received - which correctly
        identifies an unchanged API response as a duplicate load.
        """
        candidate = resolve_source_path(result, self.settings)
        if candidate is not None:
            return file_content_hash(candidate)
        if "_record_hash" in result.frame.columns:
            return stable_hash(*sorted(result.frame["_record_hash"].astype(str).tolist()))
        return stable_hash(result.source_file, result.record_count)

    def _catalog(self, table_name: str, path: Path, dataset: str) -> None:
        """Register the written table in the local catalogue (the Unity Catalog
        stand-in described in docs/databricks_deployment.md)."""
        schema = get_schema(dataset)
        self.store.upsert_dataset(
            {
                "dataset": table_name,
                "layer": "bronze",
                "description": f"Raw {dataset} records exactly as received from source systems.",
                "owner": schema.owner,
                "classification": schema.classification,
                "schema_version": schema.schema_version,
                "row_count": dataset_row_count(path),
                "size_bytes": dataset_size_bytes(path),
                "partition_columns": self.settings.storage.partitions_for(table_name)
                or ["ingestion_date"],
                "column_schema": dataset_schema(path),
                "last_updated": datetime.now(UTC),
                "last_run_id": self.run_id,
            }
        )


def resolve_source_path(result: IngestionResult, settings: Settings) -> Path | None:
    """Best-effort mapping from a result's ``source_file`` back to a real path.

    Readers store bare file names (they are what an analyst wants to see in a
    quarantine row), so the loader searches the raw tree to find the file whose
    bytes should be hashed.
    """
    # Excel results are labelled "workbook.xlsx#sheet"; the bytes to hash belong
    # to the workbook, while the ledger key keeps the sheet so two sheets of one
    # file are tracked independently.
    name = Path(result.source_file.split("#", 1)[0]).name
    if not name or name.startswith("http"):
        return None
    for candidate in Path(settings.paths.raw).rglob(name):
        if candidate.is_file():
            return candidate
    return None
