"""Agency and department spending, and transaction velocity.

These are the "where does the money go" datasets. Two things make them more than
a ``GROUP BY``:

* **Budget context.** Spending without its allocation is a number without a
  verdict. Both tables join the budget lines so that utilisation - and overrun -
  is a column rather than a calculation someone does in their head.
* **Concentration.** A department spending 80% of its money with one vendor is
  a different risk from one spread across forty, even at identical totals. The
  top-vendor share is computed for both levels.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.logging import get_logger
from src.gold.base import GoldResult, percentage, safe_divide, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore

log = get_logger(__name__)


def build_agency_spending(
    transactions: pd.DataFrame,
    agencies: pd.DataFrame,
    budgets: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    table = "gold_agency_spending"
    description = (
        "Spending per agency and fiscal year with budget utilisation, vendor "
        "concentration and anomaly counts."
    )
    if transactions.empty:
        return write_gold(
            pd.DataFrame(),
            table,
            settings,
            store=store,
            lineage=lineage,
            description=description,
            run_id=run_id,
        )

    frame = _prepare(transactions)
    grouped = frame.groupby(["agency_id", "fiscal_year"], dropna=True)

    spending = grouped.agg(
        transaction_count=("transaction_id", "count"),
        total_amount_usd=("amount_usd", "sum"),
        average_amount_usd=("amount_usd", "mean"),
        unique_vendors=("vendor_id", "nunique"),
        unique_departments=("department_id", "nunique"),
        unique_accounts=("account_id", "nunique"),
        anomaly_count=("_is_anomaly", "sum"),
        high_risk_count=("_is_high_risk", "sum"),
        total_risk_score=("risk_score", "sum"),
    ).reset_index()

    spending["anomaly_rate_pct"] = percentage(
        spending["anomaly_count"], spending["transaction_count"]
    )
    spending["average_risk_score"] = safe_divide(
        spending["total_risk_score"], spending["transaction_count"]
    ).round(2)
    spending = spending.drop(columns=["total_risk_score"])

    spending = spending.merge(_top_vendor_share(frame, "agency_id"), on="agency_id", how="left")
    spending = _attach_budgets(spending, budgets, "agency_id")

    if not agencies.empty:
        columns = [
            c
            for c in ("agency_id", "agency_name", "agency_abbreviation", "agency_type")
            if c in agencies.columns
        ]
        spending = spending.merge(agencies[columns], on="agency_id", how="left")

    for column in ("total_amount_usd", "average_amount_usd"):
        spending[column] = spending[column].round(2)

    spending = spending.sort_values("total_amount_usd", ascending=False).reset_index(drop=True)

    return write_gold(
        spending,
        table,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions", "silver_agencies", "silver_budgets"],
        transformations=["agency_aggregation", "budget_join", "vendor_concentration"],
        description=description,
        records_in=len(transactions),
        run_id=run_id,
        agencies=spending["agency_id"].nunique(),
    )


def build_department_spending(
    transactions: pd.DataFrame,
    departments: pd.DataFrame,
    budgets: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    table = "gold_department_spending"
    description = (
        "Spending per department and fiscal year with budget utilisation and " "overrun flags."
    )
    if transactions.empty:
        return write_gold(
            pd.DataFrame(),
            table,
            settings,
            store=store,
            lineage=lineage,
            description=description,
            run_id=run_id,
        )

    frame = _prepare(transactions)
    grouped = frame.groupby(["department_id", "fiscal_year"], dropna=True)

    spending = grouped.agg(
        transaction_count=("transaction_id", "count"),
        total_amount_usd=("amount_usd", "sum"),
        average_amount_usd=("amount_usd", "mean"),
        unique_vendors=("vendor_id", "nunique"),
        unique_accounts=("account_id", "nunique"),
        anomaly_count=("_is_anomaly", "sum"),
        high_risk_count=("_is_high_risk", "sum"),
    ).reset_index()

    spending["anomaly_rate_pct"] = percentage(
        spending["anomaly_count"], spending["transaction_count"]
    )
    spending = spending.merge(
        _top_vendor_share(frame, "department_id"), on="department_id", how="left"
    )
    spending = _attach_budgets(spending, budgets, "department_id")

    if not departments.empty:
        columns = [
            c
            for c in ("department_id", "department_name", "agency_id", "cost_center")
            if c in departments.columns
        ]
        spending = spending.merge(departments[columns], on="department_id", how="left")

    for column in ("total_amount_usd", "average_amount_usd"):
        spending[column] = spending[column].round(2)

    spending = spending.sort_values("total_amount_usd", ascending=False).reset_index(drop=True)

    return write_gold(
        spending,
        table,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions", "silver_departments", "silver_budgets"],
        transformations=["department_aggregation", "budget_join", "overrun_detection"],
        description=description,
        records_in=len(transactions),
        run_id=run_id,
        departments=spending["department_id"].nunique(),
        over_budget=int(spending.get("is_over_budget", pd.Series(dtype=bool)).sum()),
    )


def build_transaction_velocity(
    transactions: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    """Hourly activity per account, with a burst flag.

    Answers "where is activity spiking right now" without re-scanning the fact
    table, by pre-aggregating to the account-hour grain that velocity questions
    are actually asked at.
    """
    table = "gold_transaction_velocity"
    description = "Account activity per hour with peak five-minute velocity and burst flags."
    if transactions.empty:
        return write_gold(
            pd.DataFrame(),
            table,
            settings,
            store=store,
            lineage=lineage,
            description=description,
            run_id=run_id,
        )

    frame = _prepare(transactions)
    frame["transaction_hour"] = pd.to_datetime(
        frame["transaction_timestamp"], errors="coerce", utc=True
    ).dt.floor("h")

    aggregations = {
        "transaction_count": ("transaction_id", "count"),
        "total_amount_usd": ("amount_usd", "sum"),
        "max_amount_usd": ("amount_usd", "max"),
        "unique_vendors": ("vendor_id", "nunique"),
        "anomaly_count": ("_is_anomaly", "sum"),
        "high_risk_count": ("_is_high_risk", "sum"),
        "max_risk_score": ("risk_score", "max"),
    }
    if "account_velocity_5min" in frame.columns:
        aggregations["peak_velocity_5min"] = ("account_velocity_5min", "max")

    velocity = (
        frame.groupby(["account_id", "transaction_hour"], dropna=True)
        .agg(**aggregations)
        .reset_index()
    )

    if "peak_velocity_5min" not in velocity.columns:
        velocity["peak_velocity_5min"] = 0.0

    threshold = settings.risk.rule("high_velocity").get("max_transactions", 5)
    velocity["is_burst"] = velocity["peak_velocity_5min"] > threshold
    velocity["transactions_per_minute"] = (velocity["transaction_count"] / 60).round(4)
    velocity["transaction_date"] = velocity["transaction_hour"].dt.date.astype("string")
    velocity["hour_of_day"] = velocity["transaction_hour"].dt.hour

    for column in ("total_amount_usd", "max_amount_usd"):
        velocity[column] = velocity[column].round(2)

    velocity = velocity.sort_values(
        ["peak_velocity_5min", "transaction_count"], ascending=False
    ).reset_index(drop=True)

    return write_gold(
        velocity,
        table,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions"],
        transformations=["hourly_aggregation", "velocity_computation", "burst_detection"],
        description=description,
        records_in=len(transactions),
        run_id=run_id,
        account_hours=len(velocity),
        bursts=int(velocity["is_burst"].sum()),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _prepare(transactions: pd.DataFrame) -> pd.DataFrame:
    frame = transactions.copy()
    frame["amount_usd"] = pd.to_numeric(frame["amount_usd"], errors="coerce").fillna(0.0)
    frame["risk_score"] = pd.to_numeric(
        frame.get("risk_score", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    risk_level = frame.get("risk_level", pd.Series("LOW", index=frame.index)).astype("string")
    frame["_is_anomaly"] = (
        frame.get("is_anomaly", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    )
    frame["_is_high_risk"] = risk_level.isin(["HIGH", "CRITICAL"])
    if "fiscal_year" not in frame.columns:
        stamps = pd.to_datetime(frame["transaction_timestamp"], errors="coerce", utc=True)
        frame["fiscal_year"] = (stamps.dt.year + (stamps.dt.month >= 10).astype(int)).astype(
            "Int64"
        )
    return frame


def _top_vendor_share(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Share of spend going to each group's single largest vendor."""
    by_vendor = frame.groupby([level, "vendor_id"])["amount_usd"].sum().reset_index()
    totals = by_vendor.groupby(level)["amount_usd"].sum().rename("_total")
    top = by_vendor.loc[by_vendor.groupby(level)["amount_usd"].idxmax()].set_index(level)

    result = pd.concat([top[["vendor_id", "amount_usd"]], totals], axis=1).reset_index()
    result = result.rename(
        columns={"vendor_id": "top_vendor_id", "amount_usd": "top_vendor_amount_usd"}
    )
    result["top_vendor_share_pct"] = percentage(result["top_vendor_amount_usd"], result["_total"])
    return result[[level, "top_vendor_id", "top_vendor_amount_usd", "top_vendor_share_pct"]]


def _attach_budgets(spending: pd.DataFrame, budgets: pd.DataFrame, level: str) -> pd.DataFrame:
    """Join allocation totals and compute utilisation."""
    if budgets.empty or level not in budgets.columns:
        spending["allocated_amount_usd"] = 0.0
        spending["budget_utilisation_pct"] = 0.0
        spending["is_over_budget"] = False
        return spending

    working = budgets.copy()
    working["allocated_amount_usd"] = pd.to_numeric(
        working.get("allocated_amount_usd"), errors="coerce"
    ).fillna(0.0)
    working["fiscal_year"] = pd.to_numeric(working.get("fiscal_year"), errors="coerce").astype(
        "Int64"
    )

    totals = working.groupby([level, "fiscal_year"])["allocated_amount_usd"].sum().reset_index()
    merged = spending.merge(totals, on=[level, "fiscal_year"], how="left")
    merged["allocated_amount_usd"] = merged["allocated_amount_usd"].fillna(0.0).round(2)
    merged["budget_utilisation_pct"] = percentage(
        merged["total_amount_usd"], merged["allocated_amount_usd"]
    )
    merged["is_over_budget"] = (merged["total_amount_usd"] > merged["allocated_amount_usd"]) & (
        merged["allocated_amount_usd"] > 0
    )
    return merged
