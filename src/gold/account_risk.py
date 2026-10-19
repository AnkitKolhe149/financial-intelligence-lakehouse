"""Account-level risk profile.

Aggregates transaction-level risk up to the account, then computes an
**account** risk score that is not simply the mean of its transactions' scores.
Averaging would let a thousand routine payments dilute three critical ones into
invisibility - which is precisely backwards for oversight, where the rare event
is the point.

The account score therefore blends four signals: the worst transaction seen, the
share of activity that is high risk, the anomaly rate, and behavioural breadth
(how many distinct vendors and locations the account touches). Weights are
stated in one place below so the composition is auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Settings
from src.core.logging import get_logger
from src.gold.base import GoldResult, percentage, risk_level_from_score, safe_divide, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore

log = get_logger(__name__)

TABLE = "gold_account_risk"
DESCRIPTION = (
    "One row per account with volume, value, behavioural breadth and a composite "
    "risk score derived from its transactions."
)

#: Composition of the account risk score. Documented and summing to 1.0 so that
#: nobody has to reverse-engineer the number from the code.
SCORE_WEIGHTS = {
    "peak_transaction_risk": 0.40,  # the worst single transaction
    "high_risk_share": 0.30,  # sustained pattern, not one bad day
    "anomaly_rate": 0.20,  # model agreement
    "behavioural_breadth": 0.10,  # unusual vendor/location spread
}


def build_account_risk(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> GoldResult:
    if transactions.empty:
        return write_gold(
            pd.DataFrame(columns=_COLUMNS),
            TABLE,
            settings,
            store=store,
            lineage=lineage,
            description=DESCRIPTION,
            run_id=run_id,
        )

    frame = _prepare(transactions)
    grouped = frame.groupby("account_id", dropna=True)

    profile = grouped.agg(
        transaction_count=("transaction_id", "count"),
        total_amount_usd=("amount_usd", "sum"),
        average_amount_usd=("amount_usd", "mean"),
        median_amount_usd=("amount_usd", "median"),
        maximum_amount_usd=("amount_usd", "max"),
        stddev_amount_usd=("amount_usd", "std"),
        unique_vendors=("vendor_id", "nunique"),
        unique_locations=("location_id", "nunique"),
        anomaly_count=("_is_anomaly", "sum"),
        high_risk_count=("_is_high_risk", "sum"),
        critical_count=("_is_critical", "sum"),
        max_transaction_risk_score=("risk_score", "max"),
        avg_transaction_risk_score=("risk_score", "mean"),
        first_transaction_at=("transaction_timestamp", "min"),
        last_transaction_at=("transaction_timestamp", "max"),
        active_days=("transaction_date", "nunique"),
    ).reset_index()

    profile["stddev_amount_usd"] = profile["stddev_amount_usd"].fillna(0.0)
    profile["anomaly_rate_pct"] = percentage(profile["anomaly_count"], profile["transaction_count"])
    profile["high_risk_rate_pct"] = percentage(
        profile["high_risk_count"], profile["transaction_count"]
    )
    profile["transactions_per_active_day"] = safe_divide(
        profile["transaction_count"], profile["active_days"]
    ).round(2)
    profile["vendor_concentration"] = safe_divide(
        profile["unique_vendors"], profile["transaction_count"]
    ).round(4)

    # Velocity: peak transactions in any five-minute window, taken from the
    # leakage-safe feature if it survived into the scored frame.
    if "account_velocity_5min" in frame.columns:
        velocity = grouped["account_velocity_5min"].max().rename("peak_velocity_5min")
        profile = profile.merge(velocity, on="account_id", how="left")
    else:
        profile["peak_velocity_5min"] = 0.0

    profile["risk_score"] = _composite_score(profile, settings)
    profile["risk_level"] = risk_level_from_score(profile["risk_score"], settings)

    if not accounts.empty:
        columns = [
            c
            for c in (
                "account_id",
                "department_id",
                "account_type",
                "status",
                "region",
                "credit_limit_usd",
                "owner_name",
            )
            if c in accounts.columns
        ]
        profile = profile.merge(accounts[columns], on="account_id", how="left")
        if "credit_limit_usd" in profile.columns:
            profile["limit_utilisation_pct"] = percentage(
                profile["total_amount_usd"], profile["credit_limit_usd"]
            )

    for column in (
        "total_amount_usd",
        "average_amount_usd",
        "median_amount_usd",
        "maximum_amount_usd",
        "stddev_amount_usd",
    ):
        profile[column] = profile[column].round(2)

    profile = profile.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return write_gold(
        profile,
        TABLE,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions", "silver_accounts"],
        transformations=[
            "account_aggregation",
            "behavioural_metrics",
            "composite_risk_scoring",
            "risk_banding",
        ],
        description=DESCRIPTION,
        records_in=len(transactions),
        run_id=run_id,
        accounts=len(profile),
        high_risk_accounts=int((profile["risk_level"].isin(["HIGH", "CRITICAL"])).sum()),
    )


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
    frame["_is_critical"] = risk_level.eq("CRITICAL")
    if "transaction_date" not in frame.columns:
        frame["transaction_date"] = pd.to_datetime(
            frame["transaction_timestamp"], errors="coerce", utc=True
        ).dt.date.astype("string")
    if "location_id" not in frame.columns:
        frame["location_id"] = pd.NA
    return frame


def _composite_score(profile: pd.DataFrame, settings: Settings) -> pd.Series:
    """Blend the four signals into a 0-100 account score."""
    maximum = settings.risk.max_score

    peak = (profile["max_transaction_risk_score"] / maximum).clip(0, 1)
    high_risk_share = (profile["high_risk_rate_pct"] / 100).clip(0, 1)
    anomaly_rate = (profile["anomaly_rate_pct"] / 100).clip(0, 1)

    # Behavioural breadth: an account paying an unusually wide set of vendors
    # relative to its peers. Scaled by the 95th percentile so the measure is
    # relative to the population rather than to an arbitrary constant.
    vendors = profile["unique_vendors"].astype(float)
    reference = max(float(vendors.quantile(0.95)), 1.0)
    breadth = (vendors / reference).clip(0, 1)

    score = (
        peak * SCORE_WEIGHTS["peak_transaction_risk"]
        + high_risk_share * SCORE_WEIGHTS["high_risk_share"]
        + anomaly_rate * SCORE_WEIGHTS["anomaly_rate"]
        + breadth * SCORE_WEIGHTS["behavioural_breadth"]
    ) * maximum

    return np.round(score.clip(0, maximum)).astype(int)


_COLUMNS = [
    "account_id",
    "transaction_count",
    "total_amount_usd",
    "average_amount_usd",
    "maximum_amount_usd",
    "unique_vendors",
    "unique_locations",
    "anomaly_count",
    "high_risk_count",
    "risk_score",
    "risk_level",
]
