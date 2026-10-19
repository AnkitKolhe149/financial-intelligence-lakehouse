"""Shared plumbing for Gold builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Settings
from src.core.io import dataset_row_count, dataset_schema, dataset_size_bytes, write_parquet
from src.core.logging import get_logger
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore

log = get_logger(__name__)


@dataclass
class GoldResult:
    """Outcome of building one Gold dataset."""

    table: str
    records: int
    path: str = ""
    source_tables: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"table": self.table, "records": self.records, **self.metrics}


def write_gold(
    frame: pd.DataFrame,
    table: str,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    source_tables: list[str] | None = None,
    transformations: list[str] | None = None,
    description: str = "",
    partition_by: list[str] | None = None,
    records_in: int = 0,
    run_id: str = "",
    **metrics: Any,
) -> GoldResult:
    """Persist a Gold dataset and register its lineage and catalogue entry."""
    path: Path = settings.dataset_path("gold", table)
    partitions = partition_by or settings.storage.partitions_for(table)
    partitions = [c for c in (partitions or []) if c in frame.columns]

    written = write_parquet(
        frame,
        path,
        partition_by=partitions,
        mode="overwrite",
        compression=settings.storage.compression,
    )

    if store is not None:
        store.upsert_dataset(
            {
                "dataset": table,
                "layer": "gold",
                "description": description,
                "owner": "financial-oversight",
                "classification": "internal",
                "schema_version": "1.0",
                "row_count": dataset_row_count(path),
                "size_bytes": dataset_size_bytes(path),
                "partition_columns": partitions,
                "column_schema": dataset_schema(path),
                "last_updated": datetime.now(UTC),
                "last_run_id": run_id,
            }
        )

    if lineage is not None:
        for source in source_tables or []:
            lineage.add(
                source,
                table,
                transformations or ["aggregation"],
                source_layer="silver" if source.startswith("silver") else "gold",
                target_layer="gold",
                records_in=records_in,
                records_out=written,
            )

    result = GoldResult(
        table=table,
        records=written,
        path=str(path),
        source_tables=source_tables or [],
        transformations=transformations or [],
        metrics=metrics,
    )
    log.info("gold_built", **result.summary())
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """Element-wise division that yields ``default`` instead of inf/NaN.

    Ratios are everywhere in this layer (utilisation, anomaly rate, concentration)
    and every one of them has a legitimate zero denominator - a department with no
    budget line, a vendor with no contracts. Returning inf would poison every
    downstream aggregate and render as "Infinity" on a dashboard.
    """
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0, pd.NA)
    return result.fillna(default).replace([float("inf"), float("-inf")], default)


def percentage(numerator: pd.Series, denominator: pd.Series, decimals: int = 2) -> pd.Series:
    return (safe_divide(numerator, denominator) * 100).round(decimals)


def risk_level_from_score(scores: pd.Series, settings: Settings) -> pd.Series:
    """Map numeric scores onto the configured level bands."""
    bands = sorted(settings.risk.levels.items(), key=lambda item: item[1].min)

    def classify(value: float) -> str:
        bounded = max(0, min(round(value), settings.risk.max_score))
        for name, band in bands:
            if band.min <= bounded <= band.max:
                return name
        return bands[0][0]

    return scores.fillna(0).map(classify).astype("string")
