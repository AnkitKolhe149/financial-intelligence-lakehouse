"""Quarantine: rejected records, preserved and queryable.

The rule is absolute - **nothing is deleted**. A record that fails validation is
written here with the raw payload it arrived as, the error code, the run that
rejected it and the source it came from. Three reasons this matters more than it
might appear:

1. *Recoverability.* When the currency reference list turns out to be missing a
   legitimate code, the affected records can be replayed from quarantine instead
   of re-requested from a source system that may no longer have them.
2. *Diagnosis.* "Which source system produces the most invalid records" is a
   question the platform can answer only if it kept them.
3. *Trust.* A pipeline that reports 48,921 of 50,000 records processed and can
   show you the other 1,079 is credible. One that reports 48,921 is not.

The raw payload is stored as a JSON string rather than as typed columns, because
the whole point is that these records did not fit the typed schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.contracts import RejectedRecord
from src.core.hashing import deterministic_uuid
from src.core.io import read_parquet_or_empty, write_parquet
from src.core.logging import get_logger

log = get_logger(__name__)

QUARANTINE_COLUMNS = [
    "record_id",
    "dataset",
    "source_system",
    "source_file",
    "source_format",
    "raw_record",
    "error_code",
    "error_message",
    "error_field",
    "detected_at",
    "pipeline_run_id",
    "ingestion_run_id",
    "quarantine_date",
]


@dataclass
class QuarantineResult:
    table: str
    records: int
    path: str
    counts_by_code: dict[str, int]


def build_quarantine_frame(
    rejected: pd.DataFrame,
    *,
    dataset: str,
    run_id: str,
    business_key: str | None = None,
) -> pd.DataFrame:
    """Convert rejected records into the quarantine schema.

    Every non-audit column of the original record is serialised into
    ``raw_record`` so the row is fully reconstructable from quarantine alone.
    """
    if rejected.empty:
        return pd.DataFrame(columns=QUARANTINE_COLUMNS)

    payload_columns = [
        column
        for column in rejected.columns
        if column not in {"error_code", "error_message", "error_field"}
    ]
    raw_records = [
        json.dumps(
            {k: _jsonable(v) for k, v in record.items()},
            default=str,
            ensure_ascii=False,
        )
        for record in rejected[payload_columns].to_dict(orient="records")
    ]

    detected_at = datetime.now(UTC)
    key_series = (
        rejected[business_key].astype("string")
        if business_key and business_key in rejected.columns
        else pd.Series([None] * len(rejected), index=rejected.index)
    )

    frame = pd.DataFrame(
        {
            # Deterministic id: re-running the same rejection produces the same
            # quarantine row rather than a new one on every attempt.
            "record_id": [
                deterministic_uuid("quarantine", dataset, key, code, message)
                for key, code, message in zip(
                    key_series.fillna("__nokey__"),
                    rejected.get("error_code", pd.Series(["UNKNOWN"] * len(rejected))),
                    raw_records,
                    strict=False,
                )
            ],
            "dataset": dataset,
            "source_system": rejected.get("_source_system", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "source_file": rejected.get("_source_file", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "source_format": rejected.get("_source_format", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "raw_record": raw_records,
            "error_code": rejected.get("error_code", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "error_message": rejected.get("error_message", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "error_field": rejected.get("error_field", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "detected_at": detected_at,
            "pipeline_run_id": run_id,
            "ingestion_run_id": rejected.get("_ingestion_run_id", pd.Series([None] * len(rejected)))
            .astype("string")
            .to_numpy(),
            "quarantine_date": detected_at.date().isoformat(),
        }
    )
    return frame[QUARANTINE_COLUMNS]


def write_quarantine(
    frame: pd.DataFrame,
    dataset: str,
    settings: Settings | None = None,
    *,
    mode: str = "append",
) -> QuarantineResult:
    """Persist quarantined records, partitioned by detection date.

    Appended rather than overwritten: quarantine is a historical record of every
    rejection, which is what makes "is our data quality getting worse?" an
    answerable question.
    """
    settings = settings or get_settings()
    table = (
        f"silver_quarantine_{dataset}s"
        if not dataset.endswith("s")
        else f"silver_quarantine_{dataset}"
    )
    path = settings.paths.quarantine / table

    if frame.empty:
        log.debug("quarantine_empty", dataset=dataset)
        return QuarantineResult(table=table, records=0, path=str(path), counts_by_code={})

    written = write_parquet(
        frame,
        path,
        partition_by=["quarantine_date"],
        mode=mode,
        compression=settings.storage.compression,
    )
    counts = frame["error_code"].value_counts().to_dict()
    log.warning(
        "records_quarantined",
        dataset=dataset,
        records=written,
        **{str(code).lower(): int(count) for code, count in counts.items()},
    )
    return QuarantineResult(
        table=table,
        records=written,
        path=str(path),
        counts_by_code={str(k): int(v) for k, v in counts.items()},
    )


def quarantine_rejected_records(
    rejections: list[RejectedRecord],
    *,
    dataset: str,
    run_id: str,
    source_system: str,
    source_file: str,
    source_format: str,
    settings: Settings | None = None,
) -> QuarantineResult:
    """Quarantine record-level rejections from the contract validator.

    This is the path taken by REST and streaming sources, where records are
    judged individually and there is no dataframe to partition.
    """
    settings = settings or get_settings()
    if not rejections:
        return QuarantineResult(
            table=f"silver_quarantine_{dataset}s", records=0, path="", counts_by_code={}
        )

    detected_at = datetime.now(UTC)
    frame = pd.DataFrame(
        {
            "record_id": [
                deterministic_uuid(
                    "quarantine", dataset, json.dumps(r.raw_record, default=str), r.error_code
                )
                for r in rejections
            ],
            "dataset": dataset,
            "source_system": source_system,
            "source_file": source_file,
            "source_format": source_format,
            "raw_record": [
                json.dumps(r.raw_record, default=str, ensure_ascii=False) for r in rejections
            ],
            "error_code": [r.error_code for r in rejections],
            "error_message": [r.error_message for r in rejections],
            "error_field": [r.field_name for r in rejections],
            "detected_at": detected_at,
            "pipeline_run_id": run_id,
            "ingestion_run_id": run_id,
            "quarantine_date": detected_at.date().isoformat(),
        }
    )
    return write_quarantine(frame[QUARANTINE_COLUMNS], dataset, settings)


def read_quarantine(
    dataset: str = "transaction",
    settings: Settings | None = None,
    *,
    error_code: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read quarantined records back - powers the API and dashboard views."""
    settings = settings or get_settings()
    table = f"silver_quarantine_{dataset}s"
    path = settings.paths.quarantine / table
    filters = [("error_code", "=", error_code)] if error_code else None
    frame = read_parquet_or_empty(path, filters=filters)
    if limit is not None and not frame.empty:
        frame = frame.head(limit)
    return frame


def quarantine_summary(settings: Settings | None = None, run_id: str | None = None) -> pd.DataFrame:
    """Aggregate quarantine counts by dataset, source system and error code.

    ``run_id`` scopes the summary to a single pipeline execution. Quarantine is
    append-only history, so an unscoped summary answers "how many records have
    ever been rejected", which is the right question for a trend and the wrong
    one for a per-run scorecard.
    """
    settings = settings or get_settings()
    base = settings.paths.quarantine
    if not base.exists():
        return pd.DataFrame(columns=["dataset", "source_system", "error_code", "records"])

    columns = ["dataset", "source_system", "error_code"]
    if run_id:
        columns.append("pipeline_run_id")

    frames = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        frame = read_parquet_or_empty(directory, columns=columns)
        if frame.empty:
            continue
        if run_id and "pipeline_run_id" in frame.columns:
            frame = frame[frame["pipeline_run_id"].astype(str) == run_id]
        if not frame.empty:
            frames.append(frame[["dataset", "source_system", "error_code"]])

    if not frames:
        return pd.DataFrame(columns=["dataset", "source_system", "error_code", "records"])

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["dataset", "source_system", "error_code"], dropna=False)
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if pd.isna(value) if not isinstance(value, list | dict | tuple) else False:
        return None
    return str(value)
