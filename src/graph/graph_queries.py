"""Fraud-investigation queries, with an in-process fallback.

Each question is implemented **twice**: once in Cypher against Neo4j, once with
NetworkX over the Parquet lake. That is not redundancy for its own sake - it is
what lets the dashboard's investigation page work on a laptop with no containers
running, while still demonstrating the real graph implementation.

The fallback is honest about what it is. It builds a bounded in-memory graph
from Gold and Silver, so it answers the same questions on the same data, but it
would not scale to a graph that does not fit in memory - which is exactly why
Neo4j is the primary path.

Queries implemented:

1. ``shared_vendor_accounts`` - accounts connected through a common vendor.
2. ``high_risk_vendor_networks`` - vendors linked to multiple risky accounts.
3. ``vendor_neighbourhood`` - everything within N hops of one vendor.
4. ``circular_payment_paths`` - agency -> contract -> vendor <- account loops.
5. ``shared_tax_id_vendors`` - duplicate registrations, via the graph.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.exceptions import GraphError
from src.core.logging import get_logger
from src.graph.neo4j_client import Neo4jClient, graph_available

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cypher library
# ---------------------------------------------------------------------------

CYPHER_SHARED_VENDOR_ACCOUNTS = """
// Two different accounts paying the same vendor.
MATCH (a1:Account)-[:MADE]->(t1:Transaction)-[:PAID_TO]->(v:Vendor),
      (a2:Account)-[:MADE]->(t2:Transaction)-[:PAID_TO]->(v)
WHERE a1.account_id < a2.account_id
WITH a1, a2, v,
     count(DISTINCT t1) AS a1_payments,
     count(DISTINCT t2) AS a2_payments,
     sum(t1.amount_usd) AS a1_usd
RETURN a1.account_id      AS account_a,
       a2.account_id      AS account_b,
       v.vendor_id        AS vendor_id,
       v.name             AS vendor_name,
       v.risk_level       AS vendor_risk_level,
       a1_payments, a2_payments,
       round(a1_usd, 2)   AS account_a_paid_usd
ORDER BY a1_payments + a2_payments DESC
LIMIT $limit
"""

CYPHER_HIGH_RISK_VENDOR_NETWORKS = """
// Vendors receiving money from several independently high-risk accounts.
MATCH (a:Account)-[:MADE]->(t:Transaction)-[:PAID_TO]->(v:Vendor)
WHERE a.risk_level IN ['HIGH', 'CRITICAL']
WITH v, collect(DISTINCT a.account_id) AS accounts,
     count(t) AS transactions, sum(t.amount_usd) AS total_usd,
     sum(CASE WHEN t.is_anomaly THEN 1 ELSE 0 END) AS anomalies
WHERE size(accounts) >= $min_accounts
RETURN v.vendor_id            AS vendor_id,
       v.name                 AS vendor_name,
       v.risk_level           AS vendor_risk_level,
       size(accounts)         AS high_risk_accounts,
       accounts               AS account_ids,
       transactions,
       anomalies,
       round(total_usd, 2)    AS total_usd
ORDER BY high_risk_accounts DESC, total_usd DESC
LIMIT $limit
"""

CYPHER_VENDOR_NEIGHBOURHOOD = """
// Everything within $depth hops of a vendor. Variable-length traversal is the
// query shape that is natural here and painful in SQL.
MATCH path = (v:Vendor {vendor_id: $vendor_id})-[*1..%(depth)d]-(connected)
WITH v, connected, path
RETURN DISTINCT
       labels(connected)[0]                          AS node_type,
       coalesce(connected.vendor_id, connected.account_id, connected.agency_id,
                connected.contract_id, connected.transaction_id,
                connected.department_id, connected.location_id)  AS node_id,
       coalesce(connected.name, connected.transaction_id, '')    AS label,
       connected.risk_level                          AS risk_level,
       length(path)                                  AS hops
ORDER BY hops, node_type
LIMIT $limit
"""

CYPHER_CIRCULAR_PATHS = """
// Agency awards a contract to a vendor, and one of its own accounts pays that
// vendor outside the contract. Not proof of anything - a starting point.
MATCH (ag:Agency)-[:OWNS]->(d:Department)-[:USES]->(acc:Account)
      -[:MADE]->(t:Transaction)-[:PAID_TO]->(v:Vendor)
      -[:HAS_CONTRACT]->(c:Contract)-[:BELONGS_TO]->(ag)
WHERE t.risk_level IN ['HIGH', 'CRITICAL']
RETURN ag.agency_id       AS agency_id,
       ag.name            AS agency_name,
       c.contract_id      AS contract_id,
       c.competition_type AS competition_type,
       v.vendor_id        AS vendor_id,
       v.name             AS vendor_name,
       acc.account_id     AS account_id,
       count(t)           AS flagged_transactions,
       round(sum(t.amount_usd), 2) AS flagged_usd
ORDER BY flagged_usd DESC
LIMIT $limit
"""

CYPHER_SHARED_TAX_ID = """
// The same tax id registered under several vendor ids.
MATCH (v:Vendor)
WHERE v.tax_id IS NOT NULL
WITH v.tax_id AS tax_id, collect(v) AS vendors
WHERE size(vendors) > 1
RETURN tax_id,
       size(vendors)                                       AS registrations,
       [x IN vendors | x.vendor_id]                        AS vendor_ids,
       [x IN vendors | x.name]                             AS vendor_names,
       round(reduce(s = 0.0, x IN vendors | s + coalesce(x.total_paid_usd, 0)), 2) AS combined_paid_usd
ORDER BY combined_paid_usd DESC
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# public API - Neo4j first, fallback second
# ---------------------------------------------------------------------------


def shared_vendor_accounts(limit: int = 25, settings: Settings | None = None) -> dict[str, Any]:
    """Accounts linked by a shared vendor."""
    settings = settings or get_settings()
    if graph_available(settings):
        try:
            with Neo4jClient(settings) as client:
                rows = client.run(CYPHER_SHARED_VENDOR_ACCOUNTS, limit=limit)
            return {"backend": "neo4j", "query": "shared_vendor_accounts", "results": rows}
        except GraphError as exc:
            log.warning("cypher_failed", query="shared_vendor_accounts", error=exc.message)
    return _fallback_shared_vendor_accounts(limit, settings)


def high_risk_vendor_networks(
    limit: int = 25, min_accounts: int = 2, settings: Settings | None = None
) -> dict[str, Any]:
    """Vendors connected to multiple high-risk accounts."""
    settings = settings or get_settings()
    if graph_available(settings):
        try:
            with Neo4jClient(settings) as client:
                rows = client.run(
                    CYPHER_HIGH_RISK_VENDOR_NETWORKS, limit=limit, min_accounts=min_accounts
                )
            return {"backend": "neo4j", "query": "high_risk_vendor_networks", "results": rows}
        except GraphError as exc:
            log.warning("cypher_failed", query="high_risk_vendor_networks", error=exc.message)
    return _fallback_high_risk_vendor_networks(limit, min_accounts, settings)


def vendor_neighbourhood(
    vendor_id: str, depth: int = 2, limit: int = 100, settings: Settings | None = None
) -> dict[str, Any]:
    """Everything within ``depth`` hops of a vendor."""
    settings = settings or get_settings()
    depth = max(1, min(depth, 3))

    if graph_available(settings):
        try:
            with Neo4jClient(settings) as client:
                rows = client.run(
                    CYPHER_VENDOR_NEIGHBOURHOOD % {"depth": depth},
                    vendor_id=vendor_id,
                    limit=limit,
                )
            return {
                "backend": "neo4j",
                "vendor_id": vendor_id,
                "depth": depth,
                "available": True,
                "nodes": rows,
                "node_count": len(rows),
            }
        except GraphError as exc:
            log.warning("cypher_failed", query="vendor_neighbourhood", error=exc.message)
    return _fallback_vendor_neighbourhood(vendor_id, depth, limit, settings)


def circular_payment_paths(limit: int = 25, settings: Settings | None = None) -> dict[str, Any]:
    """Agency -> contract -> vendor <- account loops with flagged payments."""
    settings = settings or get_settings()
    if graph_available(settings):
        try:
            with Neo4jClient(settings) as client:
                rows = client.run(CYPHER_CIRCULAR_PATHS, limit=limit)
            return {"backend": "neo4j", "query": "circular_payment_paths", "results": rows}
        except GraphError as exc:
            log.warning("cypher_failed", query="circular_payment_paths", error=exc.message)
    return _fallback_circular_paths(limit, settings)


def shared_tax_id_vendors(limit: int = 25, settings: Settings | None = None) -> dict[str, Any]:
    """Vendors sharing a tax id - duplicate registrations."""
    settings = settings or get_settings()
    if graph_available(settings):
        try:
            with Neo4jClient(settings) as client:
                rows = client.run(CYPHER_SHARED_TAX_ID, limit=limit)
            return {"backend": "neo4j", "query": "shared_tax_id_vendors", "results": rows}
        except GraphError as exc:
            log.warning("cypher_failed", query="shared_tax_id_vendors", error=exc.message)
    return _fallback_shared_tax_id(limit, settings)


def graph_statistics(settings: Settings | None = None) -> dict[str, Any]:
    """Node and relationship counts, or the reason they are unavailable."""
    settings = settings or get_settings()
    if not graph_available(settings):
        return {
            "available": False,
            "reason": "Neo4j is not reachable",
            "hint": "docker compose up -d neo4j",
        }
    try:
        with Neo4jClient(settings) as client:
            return {
                "available": True,
                "nodes": client.node_counts(),
                "relationships": client.relationship_counts(),
            }
    except GraphError as exc:
        return {"available": False, "reason": exc.message}


# ---------------------------------------------------------------------------
# NetworkX fallback over the lake
# ---------------------------------------------------------------------------


def _load_lake(settings: Settings) -> dict[str, pd.DataFrame]:
    from src.core.io import read_parquet_or_empty

    return {
        "transactions": read_parquet_or_empty(
            settings.dataset_path("silver", "silver_transactions"),
            columns=[
                "transaction_id",
                "account_id",
                "vendor_id",
                "amount_usd",
                "transaction_timestamp",
                "agency_id",
                "contract_id",
            ],
        ),
        "vendors": read_parquet_or_empty(
            settings.dataset_path("silver", "silver_vendors"),
            columns=["vendor_id", "vendor_name", "tax_id"],
        ),
        "vendor_risk": read_parquet_or_empty(
            settings.dataset_path("gold", "gold_vendor_risk"),
            columns=["vendor_id", "vendor_name", "risk_level", "risk_score", "total_amount_usd"],
        ),
        "account_risk": read_parquet_or_empty(
            settings.dataset_path("gold", "gold_account_risk"),
            columns=["account_id", "risk_level", "risk_score"],
        ),
        "contracts": read_parquet_or_empty(
            settings.dataset_path("silver", "silver_contracts"),
            columns=["contract_id", "vendor_id", "agency_id", "competition_type"],
        ),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "backend": "unavailable",
        "results": [],
        "reason": reason,
        "hint": "Build the lake (`make pipeline`) or start Neo4j (`docker compose up -d neo4j`).",
    }


def _fallback_shared_vendor_accounts(limit: int, settings: Settings) -> dict[str, Any]:
    data = _load_lake(settings)
    transactions = data["transactions"]
    if transactions.empty:
        return _unavailable("no Silver transactions available")

    pairs = transactions[["account_id", "vendor_id"]].dropna().drop_duplicates()
    merged = pairs.merge(pairs, on="vendor_id", suffixes=("_a", "_b"))
    merged = merged[merged["account_id_a"] < merged["account_id_b"]]
    if merged.empty:
        return {"backend": "networkx_fallback", "results": []}

    grouped = (
        merged.groupby(["account_id_a", "account_id_b"])
        .agg(shared_vendors=("vendor_id", "nunique"))
        .reset_index()
        .sort_values("shared_vendors", ascending=False)
        .head(limit)
    )
    return {
        "backend": "networkx_fallback",
        "query": "shared_vendor_accounts",
        "results": grouped.rename(
            columns={"account_id_a": "account_a", "account_id_b": "account_b"}
        ).to_dict(orient="records"),
    }


def _fallback_high_risk_vendor_networks(
    limit: int, min_accounts: int, settings: Settings
) -> dict[str, Any]:
    data = _load_lake(settings)
    transactions, account_risk = data["transactions"], data["account_risk"]
    if transactions.empty or account_risk.empty:
        return _unavailable("gold_account_risk has not been built")

    risky = account_risk[account_risk["risk_level"].isin(["HIGH", "CRITICAL"])]
    if risky.empty:
        return {"backend": "networkx_fallback", "results": []}

    joined = transactions.merge(risky, on="account_id", how="inner")
    grouped = (
        joined.groupby("vendor_id")
        .agg(
            high_risk_accounts=("account_id", "nunique"),
            transactions=("transaction_id", "count"),
            total_usd=("amount_usd", "sum"),
            mean_account_risk=("risk_score", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["high_risk_accounts"] >= min_accounts]
    if not data["vendor_risk"].empty:
        grouped = grouped.merge(
            data["vendor_risk"][["vendor_id", "vendor_name", "risk_level"]],
            on="vendor_id",
            how="left",
        )
    grouped = grouped.sort_values(["high_risk_accounts", "total_usd"], ascending=False).head(limit)
    grouped["total_usd"] = grouped["total_usd"].round(2)

    return {
        "backend": "networkx_fallback",
        "query": "high_risk_vendor_networks",
        "results": grouped.to_dict(orient="records"),
    }


def _fallback_vendor_neighbourhood(
    vendor_id: str, depth: int, limit: int, settings: Settings
) -> dict[str, Any]:
    """Build a bounded NetworkX graph and do a breadth-first expansion."""
    import networkx as nx

    data = _load_lake(settings)
    transactions = data["transactions"]
    if transactions.empty:
        return {
            "vendor_id": vendor_id,
            "available": False,
            "reason": "no Silver transactions available",
            "nodes": [],
        }

    relevant = transactions[transactions["vendor_id"] == vendor_id]
    if relevant.empty:
        return {
            "vendor_id": vendor_id,
            "available": True,
            "nodes": [],
            "node_count": 0,
            "backend": "networkx_fallback",
            "reason": f"No transactions found for vendor {vendor_id}",
        }

    # Expand only around the accounts that touched this vendor, so the in-memory
    # graph stays proportional to the question rather than to the dataset.
    accounts = set(relevant["account_id"].dropna())
    scope = transactions[transactions["account_id"].isin(accounts)]

    graph = nx.Graph()
    graph.add_node(f"Vendor:{vendor_id}", node_type="Vendor", node_id=vendor_id)

    for row in scope.itertuples(index=False):
        account_node = f"Account:{row.account_id}"
        vendor_node = f"Vendor:{row.vendor_id}"
        transaction_node = f"Transaction:{row.transaction_id}"
        graph.add_node(account_node, node_type="Account", node_id=row.account_id)
        graph.add_node(vendor_node, node_type="Vendor", node_id=row.vendor_id)
        graph.add_node(
            transaction_node,
            node_type="Transaction",
            node_id=row.transaction_id,
            amount_usd=float(row.amount_usd or 0),
        )
        graph.add_edge(account_node, transaction_node, relationship="MADE")
        graph.add_edge(transaction_node, vendor_node, relationship="PAID_TO")
        if getattr(row, "agency_id", None):
            agency_node = f"Agency:{row.agency_id}"
            graph.add_node(agency_node, node_type="Agency", node_id=row.agency_id)
            graph.add_edge(account_node, agency_node, relationship="BELONGS_TO")

    start = f"Vendor:{vendor_id}"
    hops = nx.single_source_shortest_path_length(graph, start, cutoff=depth)

    vendor_risk = data["vendor_risk"]
    risk_lookup = (
        dict(zip(vendor_risk["vendor_id"], vendor_risk["risk_level"], strict=False))
        if not vendor_risk.empty
        else {}
    )
    account_lookup = (
        dict(
            zip(
                data["account_risk"]["account_id"], data["account_risk"]["risk_level"], strict=False
            )
        )
        if not data["account_risk"].empty
        else {}
    )

    nodes = []
    for node, distance in sorted(hops.items(), key=lambda kv: kv[1]):
        if node == start:
            continue
        attributes = graph.nodes[node]
        node_id = str(attributes.get("node_id", ""))
        node_type = attributes.get("node_type", "")
        nodes.append(
            {
                "node_type": node_type,
                "node_id": node_id,
                "label": node_id,
                "risk_level": (
                    risk_lookup.get(node_id)
                    if node_type == "Vendor"
                    else account_lookup.get(node_id)
                ),
                "hops": distance,
            }
        )
        if len(nodes) >= limit:
            break

    return {
        "backend": "networkx_fallback",
        "vendor_id": vendor_id,
        "depth": depth,
        "available": True,
        "nodes": nodes,
        "node_count": len(nodes),
        "note": (
            "Computed in-process from Parquet because Neo4j is not reachable. "
            "The Cypher implementation is used when it is."
        ),
    }


def _fallback_circular_paths(limit: int, settings: Settings) -> dict[str, Any]:
    data = _load_lake(settings)
    transactions, contracts = data["transactions"], data["contracts"]
    if transactions.empty or contracts.empty:
        return _unavailable("silver_contracts or silver_transactions unavailable")

    joined = transactions.merge(contracts, on="vendor_id", how="inner", suffixes=("", "_contract"))
    # The agency that awarded the contract is also the agency that paid.
    loops = joined[joined["agency_id"] == joined["agency_id_contract"]]
    if loops.empty:
        return {"backend": "networkx_fallback", "results": []}

    grouped = (
        loops.groupby(["agency_id", "contract_id", "vendor_id", "competition_type"])
        .agg(transactions=("transaction_id", "count"), total_usd=("amount_usd", "sum"))
        .reset_index()
        .sort_values("total_usd", ascending=False)
        .head(limit)
    )
    grouped["total_usd"] = grouped["total_usd"].round(2)
    return {
        "backend": "networkx_fallback",
        "query": "circular_payment_paths",
        "results": grouped.to_dict(orient="records"),
    }


def _fallback_shared_tax_id(limit: int, settings: Settings) -> dict[str, Any]:
    data = _load_lake(settings)
    vendors = data["vendors"]
    if vendors.empty or "tax_id" not in vendors.columns:
        return _unavailable("silver_vendors unavailable")

    grouped = (
        vendors.dropna(subset=["tax_id"])
        .groupby("tax_id")
        .agg(
            registrations=("vendor_id", "nunique"),
            vendor_ids=("vendor_id", lambda s: sorted(map(str, s))),
            vendor_names=("vendor_name", lambda s: sorted({str(v) for v in s})),
        )
        .reset_index()
    )
    grouped = grouped[grouped["registrations"] > 1].head(limit)
    return {
        "backend": "networkx_fallback",
        "query": "shared_tax_id_vendors",
        "results": grouped.to_dict(orient="records"),
    }
