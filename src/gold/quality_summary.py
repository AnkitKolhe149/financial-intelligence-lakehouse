"""Data quality published as a Gold dataset.

Quality metrics live in the metadata database, which is the right home for them
operationally - but it makes them invisible to the analytical layer. Publishing
a Gold copy means an analyst can join *"how much did agency X spend"* to
*"how trustworthy is that number"* in a single SQL query, which is the whole
argument for treating quality as data rather than as a monitoring side-channel.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.logging import get_logger
from src.gold.base import GoldResult, percentage, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore
from src.silver.quarantine import quarantine_summary

log = get_logger(__name__)

TABLE = "gold_data_quality_summary"
DESCRIPTION = (
    "Per-dataset quality scores by dimension, joined to quarantine volumes, so "
    "analytical queries can qualify their own results."
)


def build_data_quality_summary(
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    """Publish the current quality scorecard plus quarantine counts."""
    history = store.quality_history(limit=500) if store is not None else pd.DataFrame()

    if history.empty:
        summary = pd.DataFrame(columns=_COLUMNS)
    else:
        latest = (
            history.sort_values("evaluated_at").groupby("dataset", as_index=False).tail(1).copy()
        )
        summary = latest.rename(columns={"evaluated_at": "evaluated_at"})
        summary["records_passed"] = (
            summary["records_evaluated"].fillna(0) - summary["records_failed"].fillna(0)
        ).clip(lower=0)
        summary["pass_rate_pct"] = percentage(
            summary["records_passed"], summary["records_evaluated"]
        )
        summary["check_pass_rate_pct"] = percentage(
            summary["checks_passed"], summary["checks_total"]
        )

    # Quarantine is append-only history, so a cumulative count would inflate on
    # every run and make "12% of records were rejected" mean "12% across all
    # time", which is not what a scorecard for *this* run should say. Scoping to
    # the current run keeps the rate comparable between runs.
    quarantine = quarantine_summary(settings, run_id=run_id or None)
    if not quarantine.empty:
        totals = quarantine.groupby("dataset")["records"].sum().rename("quarantined_records")
        by_code = (
            quarantine.groupby(["dataset", "error_code"])["records"]
            .sum()
            .reset_index()
            .sort_values("records", ascending=False)
            .groupby("dataset")
            .head(1)
            .set_index("dataset")
        )
        if not summary.empty:
            summary = summary.merge(totals, on="dataset", how="left")
            summary["top_error_code"] = summary["dataset"].map(by_code["error_code"])
            summary["top_error_records"] = summary["dataset"].map(by_code["records"])
        else:
            summary = totals.reset_index()
            summary["top_error_code"] = summary["dataset"].map(by_code["error_code"])
            summary["top_error_records"] = summary["dataset"].map(by_code["records"])

    if "quarantined_records" not in summary.columns:
        summary["quarantined_records"] = 0
    summary["quarantined_records"] = summary["quarantined_records"].fillna(0).astype(int)

    if "records_evaluated" in summary.columns:
        summary["quarantine_rate_pct"] = percentage(
            summary["quarantined_records"],
            summary["records_evaluated"].fillna(0) + summary["quarantined_records"],
        )

    summary["snapshot_run_id"] = run_id
    summary["snapshot_at"] = pd.Timestamp.now(tz="UTC")

    return write_gold(
        summary,
        TABLE,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["data_quality_results", "silver_quarantine_transactions"],
        transformations=["latest_scorecard_selection", "quarantine_join", "rate_computation"],
        description=DESCRIPTION,
        run_id=run_id,
        datasets=len(summary),
    )


def build_quality_trend(
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    """Quality scores over time - the trend the dashboard plots.

    A point-in-time score answers "is it good?"; the trend answers "is it getting
    worse?", which is the question that lets someone intervene before an incident
    rather than after one.
    """
    table = "gold_data_quality_trend"
    history = store.quality_history(limit=2000) if store is not None else pd.DataFrame()

    if history.empty:
        trend = pd.DataFrame(
            columns=["dataset", "evaluated_date", "overall_score", "completeness", "validity"]
        )
    else:
        trend = history.copy()
        trend["evaluated_date"] = pd.to_datetime(
            trend["evaluated_at"], errors="coerce", utc=True
        ).dt.date.astype("string")
        trend = (
            trend.groupby(["dataset", "evaluated_date"], as_index=False)
            .agg(
                overall_score=("overall_score", "mean"),
                completeness=("completeness", "mean"),
                validity=("validity", "mean"),
                uniqueness=("uniqueness", "mean"),
                integrity=("integrity", "mean"),
                freshness=("freshness", "mean"),
                volume=("volume", "mean"),
                evaluations=("overall_score", "size"),
            )
            .sort_values(["dataset", "evaluated_date"])
        )
        for column in (
            "overall_score",
            "completeness",
            "validity",
            "uniqueness",
            "integrity",
            "freshness",
            "volume",
        ):
            trend[column] = trend[column].round(3)

    return write_gold(
        trend,
        table,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["data_quality_results"],
        transformations=["daily_quality_aggregation"],
        description="Daily data-quality scores per dataset, for trend analysis.",
        run_id=run_id,
        points=len(trend),
    )


def build_pipeline_run_summary(
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    """Publish pipeline-run history as a Gold dataset.

    Run metadata lives in the operational database, which DuckDB cannot read
    without an extension and a connection string. Snapshotting it into Parquet
    means pipeline health can be joined to business data in plain SQL - "show me
    the days when the load failed *and* spending spiked" is one query rather
    than two systems and a spreadsheet.
    """
    table = "gold_pipeline_runs"
    runs = store.recent_runs(limit=500) if store is not None else pd.DataFrame()

    if runs.empty:
        # The empty frame carries the *same* columns as the populated one,
        # including the derived ones. A schema that changes shape depending on
        # whether there is data is a schema that breaks every query on a quiet
        # day - the queries in sql/06_pipeline_monitoring.sql reference
        # `succeeded` and `run_date`, and they must bind on an empty table too.
        summary = pd.DataFrame(columns=[*_RUN_COLUMNS, *_RUN_DERIVED_COLUMNS])
    else:
        summary = runs.copy()
        for column in ("parameters", "metrics"):
            if column in summary.columns:
                # Nested JSON does not belong in an analytical table; keep the
                # shape flat and let callers query the metadata store for detail.
                summary[column] = summary[column].map(
                    lambda v: str(v)[:2000] if v is not None else None
                )
        summary["run_date"] = pd.to_datetime(
            summary["start_time"], errors="coerce", utc=True
        ).dt.date.astype("string")
        summary["records_per_second"] = (
            summary["records_written"].fillna(0) / summary["duration_seconds"].replace(0, pd.NA)
        ).round(1)
        summary["quarantine_rate_pct"] = percentage(
            summary["records_quarantined"].fillna(0),
            summary["records_read"].fillna(0),
        )
        summary["succeeded"] = summary["status"].isin(["SUCCESS", "SUCCESS_WITH_WARNINGS"])

    return write_gold(
        summary,
        table,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["pipeline_runs"],
        transformations=["metadata_snapshot", "throughput_computation"],
        description=(
            "Snapshot of pipeline execution history with throughput and "
            "quarantine rates, so operational health is queryable alongside "
            "business data."
        ),
        run_id=run_id,
        runs=len(summary),
    )


_COLUMNS = [
    "dataset",
    "layer",
    "overall_score",
    "completeness",
    "validity",
    "uniqueness",
    "integrity",
    "freshness",
    "volume",
    "records_evaluated",
    "records_failed",
    "checks_total",
    "checks_passed",
    "status",
    "quarantined_records",
]

_RUN_COLUMNS = [
    "run_id",
    "pipeline_name",
    "pipeline_version",
    "engine",
    "source",
    "start_time",
    "end_time",
    "duration_seconds",
    "records_read",
    "records_written",
    "records_quarantined",
    "records_failed",
    "status",
    "error_message",
]

#: Columns computed by this builder. Listed separately so the empty-frame branch
#: can guarantee the same schema as the populated one.
_RUN_DERIVED_COLUMNS = [
    "run_date",
    "records_per_second",
    "quarantine_rate_pct",
    "succeeded",
]
