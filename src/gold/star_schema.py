"""Conformed star schema for BI consumption.

The aggregate Gold tables answer specific questions fast. A star schema answers
*unanticipated* questions - which is what an analyst with a BI tool actually
needs. Both are published, because they serve different consumers, and the
duplication is deliberate rather than accidental.

    fact_transactions
        |-- dim_date
        |-- dim_account
        |-- dim_vendor      (slowly changing, type 2)
        |-- dim_agency
        |-- dim_department
        |-- dim_location

**Surrogate keys** are deterministic hashes of the natural key, not sequences.
That choice matters here: a sequence would assign different keys on every full
rebuild, breaking any saved BI report that cached them, and would be genuinely
painful to generate in a distributed engine. A hash is stable across rebuilds
and computable in parallel without coordination.

**dim_vendor is SCD type 2.** Vendor attributes change - a supplier gets added
to a watch list, changes country, is deactivated - and a report of last year's
spending should reflect what was true then. Each version carries
``effective_from`` / ``effective_to`` / ``is_current``, so a fact row joins to
the version that was current when the payment happened.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.core.config import Settings
from src.core.hashing import short_hash
from src.core.io import dataset_exists, read_parquet
from src.core.logging import get_logger
from src.gold.base import GoldResult, write_gold
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore

log = get_logger(__name__)

#: Sentinel end date for open SCD-2 records. A far-future date rather than NULL
#: so that ``BETWEEN effective_from AND effective_to`` works without special
#: cases in every query.
END_OF_TIME = pd.Timestamp("2999-12-31", tz="UTC")


def surrogate_key(prefix: str, *parts: object) -> str:
    """Deterministic surrogate key: stable across rebuilds and machines."""
    return f"{prefix}_{short_hash(*parts, length=16)}"


def build_dim_date(transactions: pd.DataFrame, settings: Settings, **kwargs) -> GoldResult:
    """Calendar dimension covering the transaction date range.

    Includes the US federal fiscal calendar, because every oversight question is
    asked in fiscal terms and computing fiscal year in each query is how two
    reports end up disagreeing.
    """
    if transactions.empty:
        return write_gold(pd.DataFrame(), "dim_date", settings, **kwargs)

    stamps = pd.to_datetime(transactions["transaction_timestamp"], errors="coerce", utc=True)
    start, end = stamps.min().normalize(), stamps.max().normalize()
    dates = pd.date_range(start, end, freq="D", tz="UTC")

    frame = pd.DataFrame({"date": dates})
    frame["date_key"] = frame["date"].dt.strftime("%Y%m%d").astype(int)
    frame["day"] = frame["date"].dt.day
    frame["month"] = frame["date"].dt.month
    frame["month_name"] = frame["date"].dt.month_name()
    frame["quarter"] = frame["date"].dt.quarter
    frame["year"] = frame["date"].dt.year
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_name"] = frame["date"].dt.day_name()
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_weekend"] = frame["date"].dt.dayofweek >= 5
    frame["is_month_end"] = frame["date"].dt.is_month_end
    frame["is_quarter_end"] = frame["date"].dt.is_quarter_end

    # US federal fiscal year runs 1 October - 30 September.
    frame["fiscal_year"] = frame["year"] + (frame["month"] >= 10).astype(int)
    frame["fiscal_quarter"] = ((frame["month"] - 10) % 12) // 3 + 1
    frame["is_fiscal_year_end"] = (frame["month"] == 9) & frame["date"].dt.is_month_end
    frame["date"] = frame["date"].dt.date.astype("string")

    return write_gold(
        frame,
        "dim_date",
        settings,
        description="Calendar dimension including the US federal fiscal calendar.",
        transformations=["calendar_generation", "fiscal_calendar"],
        **kwargs,
    )


def build_dim_vendor(
    vendors: pd.DataFrame,
    settings: Settings,
    *,
    existing: pd.DataFrame | None = None,
    **kwargs,
) -> GoldResult:
    """Vendor dimension with slowly-changing (type 2) history.

    On first build every vendor gets one open version. On later builds, a vendor
    whose tracked attributes changed gets its current row closed and a new
    version opened; unchanged vendors are left alone.
    """
    tracked = ["vendor_name", "vendor_type", "country", "state", "risk_flag", "is_active"]

    if vendors.empty:
        return write_gold(pd.DataFrame(), "dim_vendor", settings, **kwargs)

    incoming = vendors.copy()
    available = [c for c in tracked if c in incoming.columns]
    now = pd.Timestamp(datetime.now(UTC))

    # A hash of the tracked attributes is what detects a change. Comparing
    # column by column would work too, but would need editing every time an
    # attribute is added to the dimension.
    incoming["_attribute_hash"] = [
        short_hash(*row, length=16)
        for row in incoming[available].astype("string").fillna("").to_numpy()
    ]

    if existing is None or existing.empty:
        dimension = incoming.copy()
        dimension["vendor_key"] = [surrogate_key("VEN", vid, 1) for vid in dimension["vendor_id"]]
        dimension["version"] = 1
        dimension["effective_from"] = now
        dimension["effective_to"] = END_OF_TIME
        dimension["is_current"] = True
        changed = 0
    else:
        current = existing.loc[existing["is_current"].fillna(False).astype(bool)].copy()
        merged = incoming.merge(
            current[["vendor_id", "_attribute_hash", "version"]],
            on="vendor_id",
            how="left",
            suffixes=("", "_existing"),
        )

        is_new = merged["_attribute_hash_existing"].isna()
        is_changed = ~is_new & (merged["_attribute_hash"] != merged["_attribute_hash_existing"])

        new_versions = merged.loc[is_new | is_changed].copy()
        new_versions["version"] = new_versions["version"].fillna(0).astype(int) + 1
        new_versions["vendor_key"] = [
            surrogate_key("VEN", vid, ver)
            for vid, ver in zip(new_versions["vendor_id"], new_versions["version"], strict=True)
        ]
        new_versions["effective_from"] = now
        new_versions["effective_to"] = END_OF_TIME
        new_versions["is_current"] = True

        # Close the superseded versions rather than deleting them - the whole
        # point of type 2 is that history remains queryable.
        superseded_ids = set(
            new_versions.loc[is_changed.reindex(new_versions.index, fill_value=False), "vendor_id"]
        )
        history = existing.copy()
        closing = history["vendor_id"].isin(superseded_ids) & history["is_current"].fillna(False)
        history.loc[closing, "effective_to"] = now
        history.loc[closing, "is_current"] = False

        dimension = pd.concat([history, new_versions], ignore_index=True)
        changed = int(is_changed.sum())

    dimension = dimension.drop(columns=["_attribute_hash_existing"], errors="ignore")
    log.info(
        "dim_vendor_built",
        rows=len(dimension),
        current=int(dimension["is_current"].sum()),
        new_versions=changed,
    )
    return write_gold(
        dimension,
        "dim_vendor",
        settings,
        description=(
            "Vendor dimension with type-2 history: each version carries "
            "effective_from/effective_to so facts join to the version that was "
            "current when the payment occurred."
        ),
        transformations=["scd_type_2", "attribute_hash_change_detection"],
        source_tables=["silver_vendors"],
        **kwargs,
    )


def build_simple_dimension(
    frame: pd.DataFrame,
    table: str,
    natural_key: str,
    prefix: str,
    settings: Settings,
    description: str,
    **kwargs,
) -> GoldResult:
    """Type-1 dimension: current state only, overwritten each run."""
    if frame.empty:
        return write_gold(pd.DataFrame(), table, settings, **kwargs)

    dimension = frame.copy()
    dimension[f"{prefix.lower()}_key"] = [
        surrogate_key(prefix, value) for value in dimension[natural_key]
    ]
    dimension["_loaded_at"] = pd.Timestamp.now(tz="UTC")
    return write_gold(
        dimension,
        table,
        settings,
        description=description,
        transformations=["type_1_dimension", "surrogate_key_generation"],
        **kwargs,
    )


def build_fact_transactions(
    scored: pd.DataFrame,
    settings: Settings,
    **kwargs,
) -> GoldResult:
    """The transaction fact table, keyed to its dimensions.

    Carries foreign keys plus the measures, and deliberately **not** the
    descriptive attributes - those belong in the dimensions. Keeping the fact
    narrow is what keeps it fast, and it is the difference between a star schema
    and a wide table with extra steps.
    """
    if scored.empty:
        return write_gold(pd.DataFrame(), "fact_transactions", settings, **kwargs)

    frame = scored.copy()
    stamps = pd.to_datetime(frame["transaction_timestamp"], errors="coerce", utc=True)

    fact = pd.DataFrame(
        {
            "transaction_key": [surrogate_key("TXN", t) for t in frame["transaction_id"]],
            "transaction_id": frame["transaction_id"].astype("string"),
            "date_key": stamps.dt.strftime("%Y%m%d").astype("Int64"),
            "account_key": [surrogate_key("ACC", a) for a in frame.get("account_id", "")],
            "vendor_key": [surrogate_key("VEN", v) for v in frame.get("vendor_id", "")],
            "agency_key": [surrogate_key("AGY", a) for a in frame.get("agency_id", "")],
            "department_key": [surrogate_key("DEP", d) for d in frame.get("department_id", "")],
            "location_key": [surrogate_key("LOC", loc) for loc in frame.get("location_id", "")],
            "account_id": frame.get("account_id", pd.NA),
            "vendor_id": frame.get("vendor_id", pd.NA),
            "agency_id": frame.get("agency_id", pd.NA),
            "department_id": frame.get("department_id", pd.NA),
            "contract_id": frame.get("contract_id", pd.NA),
            # --- measures ---
            "amount": pd.to_numeric(frame.get("amount"), errors="coerce"),
            "amount_usd": pd.to_numeric(frame.get("amount_usd"), errors="coerce"),
            "currency": frame.get("currency", pd.NA),
            "risk_score": pd.to_numeric(frame.get("risk_score"), errors="coerce")
            .fillna(0)
            .astype("Int64"),
            "anomaly_score": pd.to_numeric(frame.get("anomaly_score"), errors="coerce"),
            "is_anomaly": frame.get("is_anomaly", False),
            "risk_level": frame.get("risk_level", "LOW"),
            "transaction_timestamp": stamps,
            "transaction_date": stamps.dt.date.astype("string"),
            "transaction_month": stamps.dt.strftime("%Y-%m"),
            "fiscal_year": frame.get("fiscal_year", pd.NA),
        }
    )

    return write_gold(
        fact,
        "fact_transactions",
        settings,
        description=(
            "Transaction fact table with surrogate keys to every dimension and "
            "the additive measures (amount, amount_usd, risk_score)."
        ),
        transformations=["surrogate_key_generation", "fact_narrowing", "measure_selection"],
        source_tables=["silver_transactions"],
        partition_by=["transaction_month"],
        **kwargs,
    )


def build_star_schema(
    scored: pd.DataFrame,
    dimensions: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    lineage: LineageCollector | None = None,
    run_id: str = "",
) -> dict[str, GoldResult]:
    """Build the fact table and every dimension."""
    common = {"store": store, "lineage": lineage, "run_id": run_id}
    results: dict[str, GoldResult] = {}

    results["dim_date"] = build_dim_date(scored, settings, **common)

    existing_vendor_dim = None
    vendor_dim_path = settings.dataset_path("gold", "dim_vendor")
    if dataset_exists(vendor_dim_path):
        existing_vendor_dim = read_parquet(vendor_dim_path)
    results["dim_vendor"] = build_dim_vendor(
        dimensions.get("vendors", pd.DataFrame()),
        settings,
        existing=existing_vendor_dim,
        **common,
    )

    simple = [
        ("dim_account", "accounts", "account_id", "ACC", "Account dimension (type 1)."),
        ("dim_agency", "agencies", "agency_id", "AGY", "Agency dimension (type 1)."),
        ("dim_department", "departments", "department_id", "DEP", "Department dimension (type 1)."),
        ("dim_location", "locations", "location_id", "LOC", "Location dimension (type 1)."),
    ]
    for table, source, key, prefix, description in simple:
        results[table] = build_simple_dimension(
            dimensions.get(source, pd.DataFrame()),
            table,
            key,
            prefix,
            settings,
            description,
            **common,
        )

    results["fact_transactions"] = build_fact_transactions(scored, settings, **common)
    return results
