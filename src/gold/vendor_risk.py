"""Vendor-level risk profile.

Vendor risk in a procurement setting is not the same shape as account risk. The
signals that matter are relational and structural:

* **Payer breadth** - a vendor paid by many unrelated accounts across many
  agencies is either a major supplier or a problem, and the distinction is worth
  surfacing.
* **Contract coverage** - payments flowing to a vendor with no contract behind
  them are an audit finding on their own.
* **Growth** - a vendor whose volume has jumped sharply against its own recent
  baseline is the "rapidly-growing vendor" question the specification asks.
* **Duplicate registration** - vendors sharing a tax id under different ids,
  reported rather than merged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Settings
from src.core.logging import get_logger
from src.gold.base import GoldResult, percentage, risk_level_from_score, safe_divide, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore
from src.silver.deduplication import near_duplicate_vendors

log = get_logger(__name__)

TABLE = "gold_vendor_risk"
DESCRIPTION = (
    "One row per vendor with payment volume, payer breadth, contract coverage, "
    "growth against its own baseline, and a composite risk score."
)

SCORE_WEIGHTS = {
    "peak_transaction_risk": 0.30,
    "anomaly_rate": 0.25,
    "payer_breadth": 0.15,
    "missing_contract_share": 0.15,
    "growth": 0.10,
    "watchlist_flag": 0.05,
}


def build_vendor_risk(
    transactions: pd.DataFrame,
    vendors: pd.DataFrame,
    contracts: pd.DataFrame,
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
    frame["_has_contract"] = frame.get("contract_id", pd.Series(pd.NA, index=frame.index)).notna()
    frame["transaction_timestamp"] = pd.to_datetime(
        frame["transaction_timestamp"], errors="coerce", utc=True
    )

    grouped = frame.groupby("vendor_id", dropna=True)
    profile = grouped.agg(
        transaction_count=("transaction_id", "count"),
        total_amount_usd=("amount_usd", "sum"),
        average_amount_usd=("amount_usd", "mean"),
        maximum_amount_usd=("amount_usd", "max"),
        unique_accounts=("account_id", "nunique"),
        unique_agencies=("agency_id", "nunique"),
        unique_departments=("department_id", "nunique"),
        anomaly_count=("_is_anomaly", "sum"),
        high_risk_count=("_is_high_risk", "sum"),
        max_transaction_risk_score=("risk_score", "max"),
        contracted_payments=("_has_contract", "sum"),
        first_payment_at=("transaction_timestamp", "min"),
        last_payment_at=("transaction_timestamp", "max"),
    ).reset_index()

    profile["anomaly_rate_pct"] = percentage(profile["anomaly_count"], profile["transaction_count"])
    profile["high_risk_rate_pct"] = percentage(
        profile["high_risk_count"], profile["transaction_count"]
    )
    profile["uncontracted_payment_pct"] = percentage(
        profile["transaction_count"] - profile["contracted_payments"],
        profile["transaction_count"],
    )
    profile["average_payment_per_account"] = safe_divide(
        profile["total_amount_usd"], profile["unique_accounts"]
    ).round(2)

    profile = profile.merge(_growth(frame), on="vendor_id", how="left")
    profile["growth_ratio"] = profile["growth_ratio"].fillna(1.0)

    profile = _attach_contracts(profile, contracts)
    profile = _attach_vendor_master(profile, vendors)

    profile["risk_score"] = _composite_score(profile, settings)
    profile["risk_level"] = risk_level_from_score(profile["risk_score"], settings)

    for column in ("total_amount_usd", "average_amount_usd", "maximum_amount_usd"):
        profile[column] = profile[column].round(2)

    profile = profile.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return write_gold(
        profile,
        TABLE,
        settings,
        store=store,
        lineage=lineage,
        source_tables=["silver_transactions", "silver_vendors", "silver_contracts"],
        transformations=[
            "vendor_aggregation",
            "payer_breadth",
            "contract_coverage",
            "growth_vs_baseline",
            "duplicate_registration_detection",
            "composite_risk_scoring",
        ],
        description=DESCRIPTION,
        records_in=len(transactions),
        run_id=run_id,
        vendors=len(profile),
        high_risk_vendors=int((profile["risk_level"].isin(["HIGH", "CRITICAL"])).sum()),
    )


def _growth(frame: pd.DataFrame) -> pd.DataFrame:
    """Recent 30-day volume against the vendor's own prior baseline.

    Expressed as a ratio against the vendor's *own* earlier rate, not against
    other vendors: a small supplier tripling its volume is the signal, and
    comparing it to a large supplier's absolute volume would hide it.
    """
    stamps = frame["transaction_timestamp"]
    if stamps.isna().all():
        return pd.DataFrame(columns=["vendor_id", "recent_30d_amount_usd", "growth_ratio"])

    latest = stamps.max()
    recent_window = latest - pd.Timedelta(days=30)
    prior_window = latest - pd.Timedelta(days=120)

    recent = frame.loc[stamps > recent_window]
    prior = frame.loc[(stamps <= recent_window) & (stamps > prior_window)]

    recent_totals = recent.groupby("vendor_id")["amount_usd"].sum().rename("recent_30d_amount_usd")
    prior_totals = prior.groupby("vendor_id")["amount_usd"].sum().rename("_prior_90d")

    merged = pd.concat([recent_totals, prior_totals], axis=1).fillna(0.0).reset_index()
    # Prior window is 3x longer, so normalise it to a comparable 30-day rate.
    prior_rate = merged["_prior_90d"] / 3.0
    merged["growth_ratio"] = np.where(
        prior_rate > 0,
        (merged["recent_30d_amount_usd"] / prior_rate).round(3),
        np.where(merged["recent_30d_amount_usd"] > 0, 3.0, 1.0),  # brand-new vendor
    )
    return merged[["vendor_id", "recent_30d_amount_usd", "growth_ratio"]]


def _attach_contracts(profile: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    if contracts.empty or "vendor_id" not in contracts.columns:
        profile["contract_count"] = 0
        profile["contract_value_usd"] = 0.0
        profile["sole_source_contracts"] = 0
        return profile

    working = contracts.copy()
    working["contract_value_usd"] = pd.to_numeric(
        working.get("contract_value_usd"), errors="coerce"
    ).fillna(0.0)
    working["_sole_source"] = (
        working.get("competition_type", pd.Series("", index=working.index))
        .astype("string")
        .eq("SOLE_SOURCE")
    )

    summary = (
        working.groupby("vendor_id")
        .agg(
            contract_count=("contract_id", "nunique"),
            contract_value_usd=("contract_value_usd", "sum"),
            sole_source_contracts=("_sole_source", "sum"),
        )
        .reset_index()
    )

    merged = profile.merge(summary, on="vendor_id", how="left")
    merged["contract_count"] = merged["contract_count"].fillna(0).astype(int)
    merged["contract_value_usd"] = merged["contract_value_usd"].fillna(0.0).round(2)
    merged["sole_source_contracts"] = merged["sole_source_contracts"].fillna(0).astype(int)
    # Payments exceeding the contracted ceiling are a direct audit finding.
    merged["payments_over_contract_value"] = (
        merged["total_amount_usd"] > merged["contract_value_usd"]
    ) & (merged["contract_value_usd"] > 0)
    return merged


def _attach_vendor_master(profile: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    if vendors.empty:
        profile["vendor_name"] = profile["vendor_id"]
        profile["risk_flag"] = "NONE"
        profile["shares_tax_id_with"] = 0
        return profile

    columns = [
        c
        for c in (
            "vendor_id",
            "vendor_name",
            "vendor_type",
            "country",
            "state",
            "risk_flag",
            "registration_date",
            "is_active",
            "tax_id",
        )
        if c in vendors.columns
    ]
    merged = profile.merge(vendors[columns], on="vendor_id", how="left")
    merged["vendor_name"] = merged.get("vendor_name", merged["vendor_id"]).fillna(
        merged["vendor_id"]
    )
    merged["risk_flag"] = merged.get("risk_flag", pd.Series("NONE", index=merged.index)).fillna(
        "NONE"
    )

    # Duplicate registrations: reported, never auto-merged. Merging master data
    # on a fuzzy signal destroys the audit trail; the platform's job is to hand
    # a steward the candidates.
    duplicates = near_duplicate_vendors(vendors)
    if not duplicates.empty and "tax_id" in merged.columns:
        counts = duplicates.set_index("tax_id")["occurrences"]
        merged["shares_tax_id_with"] = (
            merged["tax_id"].map(counts).fillna(1).astype(int) - 1
        ).clip(lower=0)
    else:
        merged["shares_tax_id_with"] = 0
    return merged


def _composite_score(profile: pd.DataFrame, settings: Settings) -> pd.Series:
    maximum = settings.risk.max_score

    peak = (profile["max_transaction_risk_score"] / maximum).clip(0, 1)
    anomaly_rate = (profile["anomaly_rate_pct"] / 100).clip(0, 1)

    accounts = profile["unique_accounts"].astype(float)
    reference = max(float(accounts.quantile(0.95)), 1.0)
    breadth = (accounts / reference).clip(0, 1)

    uncontracted = (profile["uncontracted_payment_pct"] / 100).clip(0, 1)
    # Growth contributes only above 2x; ordinary fluctuation should not score.
    growth = ((profile["growth_ratio"] - 2.0) / 3.0).clip(0, 1)
    watchlist = (
        profile.get("risk_flag", pd.Series("NONE", index=profile.index))
        .astype("string")
        .isin(["WATCH", "SANCTIONED"])
        .astype(float)
    )

    score = (
        peak * SCORE_WEIGHTS["peak_transaction_risk"]
        + anomaly_rate * SCORE_WEIGHTS["anomaly_rate"]
        + breadth * SCORE_WEIGHTS["payer_breadth"]
        + uncontracted * SCORE_WEIGHTS["missing_contract_share"]
        + growth * SCORE_WEIGHTS["growth"]
        + watchlist * SCORE_WEIGHTS["watchlist_flag"]
    ) * maximum

    return np.round(score.clip(0, maximum)).astype(int)


_COLUMNS = [
    "vendor_id",
    "vendor_name",
    "transaction_count",
    "total_amount_usd",
    "unique_accounts",
    "unique_agencies",
    "anomaly_count",
    "contract_count",
    "risk_score",
    "risk_level",
]
