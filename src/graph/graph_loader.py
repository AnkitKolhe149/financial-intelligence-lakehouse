"""Load the financial graph into Neo4j.

Graph model::

    (Agency)-[:OWNS]->(Department)-[:USES]->(Account)-[:MADE]->(Transaction)
    (Vendor)-[:HAS_CONTRACT]->(Contract)-[:BELONGS_TO]->(Agency)
    (Transaction)-[:PAID_TO]->(Vendor)
    (Transaction)-[:OCCURRED_AT]->(Location)

Two loading decisions shape everything:

**Not every transaction is loaded.** The graph exists for investigation, not
storage - it duplicates data the lake already holds authoritatively. Loading
every row would make it a slow second copy of the fact table. Instead the loader
takes all *risk-relevant* transactions plus a sample of ordinary ones, capped by
``graph.max_transactions``. Without the ordinary sample every account would look
100% suspicious, which is worse than useless for comparison.

**Writes are idempotent.** Every statement uses ``MERGE``, so re-running the
loader updates rather than duplicates. A graph load that doubles the edges on
re-run makes every degree-based metric silently wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.exceptions import GraphError
from src.core.logging import get_logger
from src.graph.neo4j_client import Neo4jClient

log = get_logger(__name__)


@dataclass
class GraphLoadResult:
    """What a graph load produced."""

    loaded: bool
    nodes: dict[str, int] = field(default_factory=dict)
    relationships: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    reason: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "nodes": sum(self.nodes.values()),
            "relationships": sum(self.relationships.values()),
            "duration_seconds": round(self.duration_seconds, 2),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

MERGE_AGENCIES = """
UNWIND $rows AS row
MERGE (a:Agency {agency_id: row.agency_id})
SET a.name = row.agency_name,
    a.abbreviation = row.agency_abbreviation,
    a.agency_type = row.agency_type,
    a.annual_budget_usd = row.annual_budget_usd
"""

MERGE_DEPARTMENTS = """
UNWIND $rows AS row
MERGE (d:Department {department_id: row.department_id})
SET d.name = row.department_name,
    d.cost_center = row.cost_center
WITH d, row
MATCH (a:Agency {agency_id: row.agency_id})
MERGE (a)-[:OWNS]->(d)
"""

MERGE_ACCOUNTS = """
UNWIND $rows AS row
MERGE (acc:Account {account_id: row.account_id})
SET acc.account_type = row.account_type,
    acc.status = row.status,
    acc.region = row.region,
    acc.credit_limit_usd = row.credit_limit_usd,
    acc.risk_score = row.risk_score,
    acc.risk_level = row.risk_level
WITH acc, row
MATCH (d:Department {department_id: row.department_id})
MERGE (d)-[:USES]->(acc)
"""

MERGE_VENDORS = """
UNWIND $rows AS row
MERGE (v:Vendor {vendor_id: row.vendor_id})
SET v.name = row.vendor_name,
    v.vendor_type = row.vendor_type,
    v.country = row.country,
    v.tax_id = row.tax_id,
    v.risk_flag = row.risk_flag,
    v.risk_score = row.risk_score,
    v.risk_level = row.risk_level,
    v.total_paid_usd = row.total_amount_usd
"""

MERGE_CONTRACTS = """
UNWIND $rows AS row
MERGE (c:Contract {contract_id: row.contract_id})
SET c.contract_type = row.contract_type,
    c.value_usd = row.contract_value_usd,
    c.status = row.status,
    c.competition_type = row.competition_type
WITH c, row
MATCH (v:Vendor {vendor_id: row.vendor_id})
MERGE (v)-[:HAS_CONTRACT]->(c)
WITH c, row
MATCH (a:Agency {agency_id: row.agency_id})
MERGE (c)-[:BELONGS_TO]->(a)
"""

MERGE_LOCATIONS = """
UNWIND $rows AS row
MERGE (l:Location {location_id: row.location_id})
SET l.city = row.city,
    l.state = row.state,
    l.country = row.country,
    l.region = row.region
"""

MERGE_TRANSACTIONS = """
UNWIND $rows AS row
MERGE (t:Transaction {transaction_id: row.transaction_id})
SET t.amount_usd = row.amount_usd,
    t.currency = row.currency,
    t.timestamp = row.transaction_timestamp,
    t.risk_score = row.risk_score,
    t.risk_level = row.risk_level,
    t.is_anomaly = row.is_anomaly,
    t.anomaly_score = row.anomaly_score
WITH t, row
MATCH (acc:Account {account_id: row.account_id})
MERGE (acc)-[:MADE]->(t)
WITH t, row
MATCH (v:Vendor {vendor_id: row.vendor_id})
MERGE (t)-[:PAID_TO]->(v)
WITH t, row
WHERE row.location_id IS NOT NULL
MATCH (l:Location {location_id: row.location_id})
MERGE (t)-[:OCCURRED_AT]->(l)
"""


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def load_graph(
    frames: dict[str, pd.DataFrame],
    settings: Settings | None = None,
    *,
    scored_transactions: pd.DataFrame | None = None,
    account_risk: pd.DataFrame | None = None,
    vendor_risk: pd.DataFrame | None = None,
    clear_first: bool = True,
) -> GraphLoadResult:
    """Load dimensions and a bounded transaction set into Neo4j.

    Returns a result rather than raising when the graph is unavailable - callers
    (the pipeline, the demo) treat that as a skipped optional stage.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    if not settings.graph.enabled:
        return GraphLoadResult(False, reason="graph disabled in configuration")

    client = Neo4jClient(settings)
    try:
        client.connect()
    except GraphError as exc:
        log.warning("graph_load_skipped", reason=exc.message)
        return GraphLoadResult(
            False,
            reason=f"Neo4j unavailable: {exc.message}",
            duration_seconds=time.perf_counter() - started,
        )

    try:
        client.create_constraints()
        if clear_first:
            client.clear()

        nodes: dict[str, int] = {}
        relationships: dict[str, int] = {}
        batch = settings.graph.batch_size

        def load(label: str, cypher: str, frame: pd.DataFrame, columns: list[str]) -> None:
            if frame is None or frame.empty:
                log.debug("graph_skip_empty", label=label)
                return
            rows = _rows(frame, columns)
            counters = client.write_batch(cypher, rows, batch_size=batch)
            nodes[label] = counters["nodes_created"]
            relationships[label] = counters["relationships_created"]
            log.info("graph_loaded", label=label, rows=len(rows), **counters)

        agencies = frames.get("agencies", pd.DataFrame())
        load(
            "Agency",
            MERGE_AGENCIES,
            agencies,
            ["agency_id", "agency_name", "agency_abbreviation", "agency_type", "annual_budget_usd"],
        )

        load(
            "Department",
            MERGE_DEPARTMENTS,
            frames.get("departments", pd.DataFrame()),
            ["department_id", "department_name", "agency_id", "cost_center"],
        )

        accounts = _enrich(frames.get("accounts", pd.DataFrame()), account_risk, "account_id")
        load(
            "Account",
            MERGE_ACCOUNTS,
            accounts,
            [
                "account_id",
                "department_id",
                "account_type",
                "status",
                "region",
                "credit_limit_usd",
                "risk_score",
                "risk_level",
            ],
        )

        vendors = _enrich(
            frames.get("vendors", pd.DataFrame()),
            vendor_risk,
            "vendor_id",
            extra=["total_amount_usd"],
        )
        load(
            "Vendor",
            MERGE_VENDORS,
            vendors,
            [
                "vendor_id",
                "vendor_name",
                "vendor_type",
                "country",
                "tax_id",
                "risk_flag",
                "risk_score",
                "risk_level",
                "total_amount_usd",
            ],
        )

        load(
            "Contract",
            MERGE_CONTRACTS,
            frames.get("contracts", pd.DataFrame()),
            [
                "contract_id",
                "vendor_id",
                "agency_id",
                "contract_type",
                "contract_value_usd",
                "status",
                "competition_type",
            ],
        )

        load(
            "Location",
            MERGE_LOCATIONS,
            frames.get("locations", pd.DataFrame()),
            ["location_id", "city", "state", "country", "region"],
        )

        transactions = select_transactions_for_graph(
            (
                scored_transactions
                if scored_transactions is not None
                else frames.get("transactions", pd.DataFrame())
            ),
            settings,
        )
        load(
            "Transaction",
            MERGE_TRANSACTIONS,
            transactions,
            [
                "transaction_id",
                "account_id",
                "vendor_id",
                "location_id",
                "amount_usd",
                "currency",
                "transaction_timestamp",
                "risk_score",
                "risk_level",
                "is_anomaly",
                "anomaly_score",
            ],
        )

        duration = time.perf_counter() - started
        result = GraphLoadResult(True, nodes, relationships, duration)
        log.info("graph_load_complete", **result.summary())
        return result

    except Exception as exc:
        log.error("graph_load_failed", error=str(exc)[:300], error_type=type(exc).__name__)
        return GraphLoadResult(
            False, reason=str(exc)[:200], duration_seconds=time.perf_counter() - started
        )
    finally:
        client.close()


def select_transactions_for_graph(transactions: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Choose which transactions to materialise as nodes.

    All flagged transactions, plus a random sample of ordinary ones for
    comparison, capped at ``graph.max_transactions``. Loading only the flagged
    ones would make every account in the graph look entirely suspicious.
    """
    if transactions.empty:
        return transactions

    cap = settings.graph.max_transactions
    frame = transactions.copy()

    if "risk_level" in frame.columns:
        flagged = frame[
            frame["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])
            | frame.get("is_anomaly", False).fillna(False).astype(bool)
        ]
    else:
        flagged = frame.iloc[0:0]

    if settings.graph.high_risk_only:
        return flagged.head(cap)

    remaining = cap - len(flagged)
    if remaining <= 0:
        return (
            flagged.nlargest(cap, "risk_score")
            if "risk_score" in flagged.columns
            else flagged.head(cap)
        )

    ordinary = frame.drop(index=flagged.index, errors="ignore")
    sample = (
        ordinary.sample(n=min(remaining, len(ordinary)), random_state=settings.generation.seed)
        if not ordinary.empty
        else ordinary
    )

    selected = pd.concat([flagged, sample], ignore_index=True)
    log.info(
        "graph_transaction_selection",
        total_available=len(frame),
        flagged=len(flagged),
        sampled=len(sample),
        selected=len(selected),
        cap=cap,
    )
    return selected


def _enrich(
    dimension: pd.DataFrame,
    risk: pd.DataFrame | None,
    key: str,
    extra: list[str] | None = None,
) -> pd.DataFrame:
    """Attach risk scores to a dimension before loading it as nodes.

    Risk lives on the node itself so that Cypher can filter on it
    (``MATCH (v:Vendor) WHERE v.risk_level = 'HIGH'``) rather than traversing to
    the transactions and aggregating on every query.
    """
    if dimension is None or dimension.empty:
        return pd.DataFrame()
    if risk is None or risk.empty:
        enriched = dimension.copy()
        enriched["risk_score"] = 0
        enriched["risk_level"] = "LOW"
        for column in extra or []:
            enriched[column] = 0.0
        return enriched

    columns = [key, "risk_score", "risk_level", *(extra or [])]
    available = [c for c in columns if c in risk.columns]
    merged = dimension.merge(risk[available], on=key, how="left")
    merged["risk_score"] = merged["risk_score"].fillna(0).astype(int)
    merged["risk_level"] = merged["risk_level"].fillna("LOW")
    for column in extra or []:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0.0)
    return merged


def _rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Convert a frame to driver-safe rows.

    The Neo4j driver rejects numpy scalars, pandas NA and Timestamps, so
    everything is coerced to Python primitives here rather than failing mid-load.
    """
    present = [c for c in columns if c in frame.columns]
    working = frame[present].copy()

    for column in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column]):
            working[column] = working[column].astype("string")
        elif pd.api.types.is_bool_dtype(working[column]):
            working[column] = working[column].astype(bool)
        elif pd.api.types.is_numeric_dtype(working[column]):
            working[column] = pd.to_numeric(working[column], errors="coerce").astype(float)
        else:
            working[column] = working[column].astype("string")

    working = working.where(pd.notna(working), None)
    records = working.to_dict(orient="records")

    for column in columns:
        if column not in present:
            for record in records:
                record[column] = None
    return records
