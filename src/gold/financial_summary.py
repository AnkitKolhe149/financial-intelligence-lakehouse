"""Daily financial summary - the platform's headline time series."""

from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.logging import get_logger
from src.gold.base import GoldResult, percentage, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore

log = get_logger(__name__)

TABLE = "gold_daily_financial_summary"
DESCRIPTION = (
    "One row per calendar day: volume, value, distribution statistics and "
    "anomaly counts. The primary time series behind the overview dashboard."
)


def build_daily_financial_summary(
    transactions: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    """Aggregate scored transactions to one row per day.

    The **median** is carried alongside the mean deliberately. Transaction
    amounts are heavy-tailed, so a single large legitimate payment moves the
    daily mean substantially while the median does not; showing both is what
    lets an analyst tell "we spent more" from "we made one big payment".
    """
    if transactions.empty:
        return write_gold(
            _empty_frame(),
            TABLE,
            settings,
            store=store,
            lineage=lineage,
            description=DESCRIPTION,
            run_id=run_id,
        )

    frame = transactions.copy()
    frame["transaction_date"] = pd.to_datetime(
        frame["transaction_timestamp"], errors="coerce", utc=True
    ).dt.date.astype("string")
    frame["amount_usd"] = pd.to_numeric(frame["amount_usd"], errors="coerce").fillna(0.0)

    is_anomaly = (
        frame.get("is_anomaly", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    )
    risk_level = frame.get("risk_level", pd.Series("LOW", index=frame.index)).astype("string")
    frame["_is_anomaly"] = is_anomaly
    frame["_is_high_risk"] = risk_level.isin(["HIGH", "CRITICAL"])
    frame["_is_critical"] = risk_level.eq("CRITICAL")

    grouped = frame.groupby("transaction_date", dropna=True)
    summary = grouped.agg(
        transaction_count=("transaction_id", "count"),
        total_amount_usd=("amount_usd", "sum"),
        average_transaction_usd=("amount_usd", "mean"),
        median_transaction_usd=("amount_usd", "median"),
        max_transaction_usd=("amount_usd", "max"),
        p95_transaction_usd=("amount_usd", lambda s: s.quantile(0.95)),
        unique_accounts=("account_id", "nunique"),
        unique_vendors=("vendor_id", "nunique"),
        unique_agencies=("agency_id", "nunique"),
        anomaly_count=("_is_anomaly", "sum"),
        high_risk_count=("_is_high_risk", "sum"),
        critical_risk_count=("_is_critical", "sum"),
        total_risk_score=(
            ("risk_score", "sum") if "risk_score" in frame.columns else ("amount_usd", "size")
        ),
    ).reset_index()

    summary["anomaly_rate_pct"] = percentage(summary["anomaly_count"], summary["transaction_count"])
    summary["high_risk_rate_pct"] = percentage(
        summary["high_risk_count"], summary["transaction_count"]
    )
    summary["average_risk_score"] = (
        summary["total_risk_score"] / summary["transaction_count"]
    ).round(2)
    summary = summary.drop(columns=["total_risk_score"])

    for column in (
        "total_amount_usd",
        "average_transaction_usd",
        "median_transaction_usd",
        "max_transaction_usd",
        "p95_transaction_usd",
    ):
        summary[column] = summary[column].round(2)

    summary = summary.sort_values("transaction_date").reset_index(drop=True)

    # Rolling context: a day means little in isolation. The 7-day trailing mean
    # is what turns "1,204 transactions" into "1,204 against a norm of 900".
    summary["rolling_7d_avg_volume"] = (
        summary["transaction_count"].rolling(7, min_periods=1).mean().round(1)
    )
    summary["rolling_7d_avg_amount_usd"] = (
        summary["total_amount_usd"].rolling(7, min_periods=1).mean().round(2)
    )
    summary["volume_vs_7d_avg_pct"] = percentage(
        summary["transaction_count"] - summary["rolling_7d_avg_volume"],
        summary["rolling_7d_avg_volume"],
    )

    return write_gold(
        summary,
        TABLE,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions"],
        transformations=[
            "daily_aggregation",
            "distribution_statistics",
            "anomaly_counts",
            "rolling_7d_baseline",
        ],
        description=DESCRIPTION,
        records_in=len(transactions),
        run_id=run_id,
        days=len(summary),
        total_value_usd=float(summary["total_amount_usd"].sum()),
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transaction_date",
            "transaction_count",
            "total_amount_usd",
            "average_transaction_usd",
            "median_transaction_usd",
            "max_transaction_usd",
            "p95_transaction_usd",
            "unique_accounts",
            "unique_vendors",
            "unique_agencies",
            "anomaly_count",
            "high_risk_count",
            "critical_risk_count",
            "anomaly_rate_pct",
            "high_risk_rate_pct",
            "average_risk_score",
            "rolling_7d_avg_volume",
            "rolling_7d_avg_amount_usd",
            "volume_vs_7d_avg_pct",
        ]
    )
