"""Silver orchestration: Bronze in, validated Silver out.

The stage order is fixed and each step exists for a reason:

    align -> normalise -> cast -> derive -> validate -> deduplicate -> write

*Normalise before cast* so that ``" 1,234.50 "`` becomes a number instead of a
null. *Derive before validate* so that a derived column (``amount_usd``) is
checked by the same rules as a source column. *Validate before deduplicate* so
that an invalid record is reported as invalid rather than as a duplicate of
another invalid record - the error code an analyst sees should describe the
actual problem.

Dimensions are built before the fact table because the fact's referential checks
read the Silver dimensions, not the Bronze ones. Enforcing integrity against
validated dimensions is the difference between "this vendor exists in a file
somewhere" and "this vendor exists in our system of record".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.io import (
    dataset_exists,
    dataset_row_count,
    dataset_schema,
    dataset_size_bytes,
    read_parquet,
    read_parquet_or_empty,
    write_parquet,
)
from src.core.logging import get_logger
from src.core.run_context import RunContext
from src.core.schema import SILVER_AUDIT_COLUMNS
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore, get_metadata_store
from src.silver.deduplication import deduplicate
from src.silver.quarantine import build_quarantine_frame, write_quarantine
from src.silver.spec import SILVER_BUILD_ORDER, SilverSpec, get_silver_spec
from src.silver.validation import (
    ValidationOutcome,
    add_transaction_derivations,
    cast_types,
    normalise_strings,
    validate_frame,
)

log = get_logger(__name__)


@dataclass
class SilverResult:
    """Outcome of building one Silver table."""

    dataset: str
    table: str
    records_in: int
    records_out: int
    records_quarantined: int
    duplicates_removed: int
    duration_seconds: float
    path: str = ""
    errors_by_code: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "records_in": self.records_in,
            "records_out": self.records_out,
            "quarantined": self.records_quarantined,
            "duplicates_removed": self.duplicates_removed,
            "duration_seconds": round(self.duration_seconds, 3),
            **{f"error_{k.lower()}": v for k, v in self.errors_by_code.items()},
        }

    @property
    def retention_rate(self) -> float:
        return round(self.records_out / self.records_in * 100, 2) if self.records_in else 100.0


def build_silver(
    dataset: str,
    context: RunContext,
    settings: Settings | None = None,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    write: bool = True,
) -> SilverResult:
    """Build one Silver table from its Bronze source."""
    settings = settings or get_settings()
    store = store if store is not None else get_metadata_store()
    spec = get_silver_spec(dataset)
    started = time.perf_counter()

    source_path = settings.dataset_path("bronze", spec.source_table)
    if not dataset_exists(source_path):
        log.warning("silver_source_missing", table=spec.source_table)
        return SilverResult(dataset, spec.target_table, 0, 0, 0, 0, 0.0)

    bronze = read_parquet(source_path)
    records_in = len(bronze)

    # ---- clean ------------------------------------------------------------
    # The normalised-but-uncast frame is kept: validation needs it to tell
    # "the source sent something unparseable" from "the source sent nothing".
    normalised = normalise_strings(bronze, spec.normalisation)
    frame = cast_types(normalised, spec)

    # ---- derive ------------------------------------------------------------
    if dataset == "transaction":
        frame = add_transaction_derivations(frame)
    elif dataset == "transaction_event":
        frame = _add_event_derivations(frame)

    # ---- validate ----------------------------------------------------------
    references = _load_references(spec, settings)
    outcome: ValidationOutcome = validate_frame(
        frame, spec, settings, references, raw_frame=normalised
    )

    # ---- deduplicate --------------------------------------------------------
    dedup = deduplicate(outcome.valid, spec.dedup, dataset=dataset)

    # ---- quarantine ---------------------------------------------------------
    rejected = (
        pd.concat([outcome.rejected, dedup.duplicates], ignore_index=True)
        if not dedup.duplicates.empty
        else outcome.rejected
    )

    quarantined = 0
    if settings.quality.quarantine_invalid_records and not rejected.empty:
        quarantine_frame = build_quarantine_frame(
            rejected,
            dataset=dataset,
            run_id=context.run_id,
            business_key=spec.schema.business_key[0] if spec.schema.business_key else None,
        )
        quarantined = write_quarantine(quarantine_frame, dataset, settings).records

    # ---- finalise -----------------------------------------------------------
    silver = _apply_silver_audit(dedup.survivors, context, spec)
    silver = _select_output_columns(silver, spec)

    path = settings.dataset_path("silver", spec.target_table)
    if write:
        partitions = settings.storage.partitions_for(spec.target_table)
        if not partitions and spec.partition_column and spec.partition_column in silver.columns:
            partitions = [spec.partition_column]
        write_parquet(
            silver,
            path,
            partition_by=partitions,
            mode="overwrite",
            compression=settings.storage.compression,
        )
        _catalog(store, spec, path, context.run_id)

    duration = time.perf_counter() - started
    errors_by_code = dict(outcome.counts_by_code)
    if dedup.removed_total:
        errors_by_code["DUPLICATE_RECORD"] = (
            errors_by_code.get("DUPLICATE_RECORD", 0) + dedup.removed_total
        )

    result = SilverResult(
        dataset=dataset,
        table=spec.target_table,
        records_in=records_in,
        records_out=len(silver),
        records_quarantined=quarantined,
        duplicates_removed=dedup.removed_total,
        duration_seconds=duration,
        path=str(path),
        errors_by_code=errors_by_code,
    )

    if lineage is not None:
        lineage.add(
            spec.source_table,
            spec.target_table,
            spec.transformation_names(),
            source_layer="bronze",
            target_layer="silver",
            records_in=records_in,
            records_out=len(silver),
        )

    log.info("silver_built", **result.summary(), retention_pct=result.retention_rate)
    return result


def build_all_silver(
    context: RunContext,
    settings: Settings | None = None,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    datasets: tuple[str, ...] = SILVER_BUILD_ORDER,
) -> dict[str, SilverResult]:
    """Build every Silver table in dependency order."""
    settings = settings or get_settings()
    results: dict[str, SilverResult] = {}
    for dataset in datasets:
        try:
            results[dataset] = build_silver(
                dataset, context, settings, store=store, lineage=lineage
            )
        except Exception as exc:
            log.error(
                "silver_dataset_failed",
                dataset=dataset,
                error=str(exc)[:300],
                error_type=type(exc).__name__,
            )
            if settings.pipeline.fail_fast:
                raise
            results[dataset] = SilverResult(dataset, f"silver_{dataset}s", 0, 0, 0, 0, 0.0)
    return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_references(spec: SilverSpec, settings: Settings) -> dict[str, pd.Series]:
    """Load the key columns of the Silver dimensions this dataset references."""
    references: dict[str, pd.Series] = {}
    for table, key_column in spec.reference_tables.items():
        path = settings.dataset_path("silver", table)
        if not dataset_exists(path):
            continue
        # The referenced dimension's own key, e.g. silver_accounts.account_id.
        frame = read_parquet_or_empty(path, columns=[key_column])
        if not frame.empty:
            references[table] = frame[key_column]
    return references


def _apply_silver_audit(frame: pd.DataFrame, context: RunContext, spec: SilverSpec) -> pd.DataFrame:
    """Attach Silver audit columns, carrying Bronze provenance forward."""
    if frame.empty:
        return frame
    stamped = frame.copy()
    stamped["_silver_run_id"] = context.run_id
    stamped["_processed_timestamp"] = datetime.now(UTC)
    if "_schema_version" not in stamped.columns:
        stamped["_schema_version"] = spec.schema.schema_version
    if "_is_late_arriving" not in stamped.columns:
        stamped["_is_late_arriving"] = False
    return stamped


def _select_output_columns(frame: pd.DataFrame, spec: SilverSpec) -> pd.DataFrame:
    """Project to canonical columns + derived columns + Silver audit columns.

    Bronze-only columns (``_source_row_number``, ``ingestion_date``) are dropped
    here: they belong to the raw layer's bookkeeping and would be noise in every
    downstream query.
    """
    if frame.empty:
        return frame
    canonical = [c for c in spec.schema.column_names if c in frame.columns]
    derived = [c for c in spec.derived_columns if c in frame.columns]
    audit = [c for c in SILVER_AUDIT_COLUMNS if c in frame.columns]
    extra = [c for c in ("_duplicate_count",) if c in frame.columns]

    # A column can legitimately appear in more than one of these lists -
    # ``_is_late_arriving`` is both a derived column and a Silver audit column -
    # so de-duplicate while preserving order. Selecting a duplicated label would
    # produce a two-column frame and fail at write time.
    ordered = list(dict.fromkeys(canonical + derived + audit + extra))
    return frame[ordered]


def _add_event_derivations(frame: pd.DataFrame) -> pd.DataFrame:
    """Lateness metrics for streaming events."""
    if frame.empty:
        return frame
    enriched = frame.copy()
    if {"event_timestamp", "ingest_timestamp"} <= set(enriched.columns):
        lateness = (enriched["ingest_timestamp"] - enriched["event_timestamp"]).dt.total_seconds()
        enriched["lateness_seconds"] = lateness.astype("Float64")
        enriched["_is_late_arriving"] = (lateness > 300).fillna(False)
    else:
        enriched["lateness_seconds"] = pd.Series(pd.NA, index=enriched.index, dtype="Float64")
        enriched["_is_late_arriving"] = False
    return enriched


def _catalog(store: MetadataStore, spec: SilverSpec, path: Any, run_id: str) -> None:
    store.upsert_dataset(
        {
            "dataset": spec.target_table,
            "layer": "silver",
            "description": spec.description,
            "owner": spec.schema.owner,
            "classification": spec.schema.classification,
            "schema_version": spec.schema.schema_version,
            "row_count": dataset_row_count(path),
            "size_bytes": dataset_size_bytes(path),
            "partition_columns": [spec.partition_column] if spec.partition_column else [],
            "column_schema": dataset_schema(path),
            "last_updated": datetime.now(UTC),
            "last_run_id": run_id,
        }
    )
