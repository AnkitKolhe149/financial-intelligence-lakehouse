"""Graph layer: query correctness and graceful degradation.

These run without Neo4j. That is the point: the fallback path is a shipped
feature, not a test convenience, so it is what gets exercised by default. The
Cypher path is covered by tests marked ``neo4j``, which skip when no instance is
reachable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import GraphError
from src.core.io import write_parquet
from src.graph.graph_loader import select_transactions_for_graph
from src.graph.graph_queries import (
    graph_statistics,
    high_risk_vendor_networks,
    shared_tax_id_vendors,
    shared_vendor_accounts,
    vendor_neighbourhood,
)
from src.graph.neo4j_client import Neo4jClient, graph_available, reset_probe_cache
from tests.conftest import neo4j_required


@pytest.fixture
def graph_lake(settings, store, run_context):
    """A lake with the Silver and Gold datasets the graph queries read."""
    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i}" for i in range(12)],
            "account_id": ["ACC-1"] * 4 + ["ACC-2"] * 4 + ["ACC-3"] * 4,
            "vendor_id": ["VEN-1", "VEN-2", "VEN-1", "VEN-3"] * 3,
            "amount_usd": [100.0 * (i + 1) for i in range(12)],
            "transaction_timestamp": pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC"),
            "agency_id": ["AGY-1"] * 12,
            "contract_id": ["CON-1"] * 6 + [None] * 6,
        }
    )
    vendors = pd.DataFrame(
        {
            "vendor_id": ["VEN-1", "VEN-2", "VEN-3"],
            "vendor_name": ["Acme Inc", "Acme Inc.", "Other Corp"],
            "tax_id": ["TX-SHARED", "TX-SHARED", "TX-UNIQUE"],
        }
    )
    account_risk = pd.DataFrame(
        {
            "account_id": ["ACC-1", "ACC-2", "ACC-3"],
            "risk_level": ["HIGH", "CRITICAL", "LOW"],
            "risk_score": [60, 80, 10],
        }
    )
    vendor_risk = pd.DataFrame(
        {
            "vendor_id": ["VEN-1", "VEN-2", "VEN-3"],
            "vendor_name": ["Acme Inc", "Acme Inc.", "Other Corp"],
            "risk_level": ["HIGH", "LOW", "LOW"],
            "risk_score": [70, 20, 15],
            "total_amount_usd": [1000.0, 500.0, 300.0],
        }
    )
    contracts = pd.DataFrame(
        {
            "contract_id": ["CON-1"],
            "vendor_id": ["VEN-1"],
            "agency_id": ["AGY-1"],
            "competition_type": ["SOLE_SOURCE"],
        }
    )

    write_parquet(transactions, settings.dataset_path("silver", "silver_transactions"))
    write_parquet(vendors, settings.dataset_path("silver", "silver_vendors"))
    write_parquet(contracts, settings.dataset_path("silver", "silver_contracts"))
    write_parquet(account_risk, settings.dataset_path("gold", "gold_account_risk"))
    write_parquet(vendor_risk, settings.dataset_path("gold", "gold_vendor_risk"))
    return settings


class TestAvailability:
    def test_probe_never_raises(self, settings):
        reset_probe_cache()
        assert isinstance(graph_available(settings, use_cache=False), bool)

    def test_disabled_configuration_reports_unavailable(self, settings):
        reset_probe_cache()
        disabled = settings.model_copy(
            update={"graph": settings.graph.model_copy(update={"enabled": False})}
        )
        assert graph_available(disabled, use_cache=False) is False

    def test_connect_failure_raises_a_typed_error(self, settings):
        unreachable = settings.model_copy(
            update={"neo4j": settings.neo4j.model_copy(update={"uri": "bolt://127.0.0.1:1"})}
        )
        with pytest.raises(GraphError):
            Neo4jClient(unreachable).connect()

    def test_statistics_explain_why_they_are_unavailable(self, settings):
        reset_probe_cache()
        stats = graph_statistics(settings)
        if not stats.get("available"):
            assert "reason" in stats
            assert "hint" in stats


class TestFallbackQueries:
    def test_shared_vendor_accounts(self, graph_lake):
        result = shared_vendor_accounts(limit=10, settings=graph_lake)
        assert result["backend"] in {"neo4j", "networkx_fallback"}
        assert result["results"], "ACC-1, ACC-2 and ACC-3 all share vendors"

    def test_high_risk_vendor_networks_finds_the_shared_vendor(self, graph_lake):
        result = high_risk_vendor_networks(limit=10, min_accounts=2, settings=graph_lake)
        vendors = {row["vendor_id"] for row in result["results"]}
        # VEN-1 is paid by ACC-1 (HIGH) and ACC-2 (CRITICAL).
        assert "VEN-1" in vendors

    def test_low_risk_accounts_are_excluded(self, graph_lake):
        result = high_risk_vendor_networks(limit=10, min_accounts=3, settings=graph_lake)
        # Only two of the three accounts are high risk, so nothing reaches three.
        assert not result["results"]

    def test_vendor_neighbourhood_expands_from_the_vendor(self, graph_lake):
        result = vendor_neighbourhood("VEN-1", depth=2, settings=graph_lake)
        assert result["available"] is True
        assert result["node_count"] > 0
        types = {node["node_type"] for node in result["nodes"]}
        assert "Account" in types

    def test_neighbourhood_of_an_unknown_vendor_is_empty_not_an_error(self, graph_lake):
        result = vendor_neighbourhood("VEN-NOPE", settings=graph_lake)
        assert result["node_count"] == 0
        assert "reason" in result

    def test_duplicate_registrations_are_found(self, graph_lake):
        result = shared_tax_id_vendors(limit=10, settings=graph_lake)
        assert result["results"]
        assert result["results"][0]["registrations"] == 2

    def test_queries_degrade_without_a_lake(self, settings):
        result = shared_vendor_accounts(limit=10, settings=settings)
        assert result["results"] == []
        assert "hint" in result or "reason" in result


class TestGraphLoading:
    def test_selection_includes_ordinary_transactions_for_contrast(self, settings):
        """Loading only flagged rows would make every account look 100% risky."""
        frame = pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i}" for i in range(100)],
                "risk_level": ["CRITICAL"] * 5 + ["LOW"] * 95,
                "risk_score": [90] * 5 + [0] * 95,
                "is_anomaly": [True] * 5 + [False] * 95,
            }
        )
        selected = select_transactions_for_graph(frame, settings)
        assert len(selected) > 5
        assert (selected["risk_level"] == "LOW").any()

    def test_selection_respects_the_cap(self, settings):
        capped = settings.model_copy(
            update={"graph": settings.graph.model_copy(update={"max_transactions": 20})}
        )
        frame = pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i}" for i in range(500)],
                "risk_level": ["LOW"] * 500,
                "risk_score": [0] * 500,
                "is_anomaly": [False] * 500,
            }
        )
        assert len(select_transactions_for_graph(frame, capped)) <= 20

    def test_high_risk_only_mode(self, settings):
        strict = settings.model_copy(
            update={"graph": settings.graph.model_copy(update={"high_risk_only": True})}
        )
        frame = pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i}" for i in range(50)],
                "risk_level": ["CRITICAL"] * 5 + ["LOW"] * 45,
                "risk_score": [90] * 5 + [0] * 45,
                "is_anomaly": [True] * 5 + [False] * 45,
            }
        )
        assert len(select_transactions_for_graph(frame, strict)) == 5

    def test_load_is_skipped_cleanly_when_neo4j_is_absent(self, settings, graph_lake):
        from src.graph.graph_loader import load_graph

        unreachable = settings.model_copy(
            update={"neo4j": settings.neo4j.model_copy(update={"uri": "bolt://127.0.0.1:1"})}
        )
        result = load_graph({}, unreachable)
        assert result.loaded is False
        assert result.reason
        # Never raises: the pipeline treats the graph as an optional stage.


@neo4j_required
class TestNeo4jIntegration:
    """Runs only when a Neo4j instance is reachable."""

    def test_load_and_query(self, graph_lake, settings):
        from src.core.io import read_parquet
        from src.graph.graph_loader import load_graph

        frames = {
            "agencies": pd.DataFrame(
                {
                    "agency_id": ["AGY-1"],
                    "agency_name": ["Test Agency"],
                    "agency_abbreviation": ["TA"],
                    "agency_type": ["EXECUTIVE"],
                    "annual_budget_usd": [1000.0],
                }
            ),
            "departments": pd.DataFrame(
                {
                    "department_id": ["DEP-1"],
                    "department_name": ["Test Dept"],
                    "agency_id": ["AGY-1"],
                    "cost_center": ["CC-1"],
                }
            ),
            "accounts": pd.DataFrame(
                {
                    "account_id": ["ACC-1", "ACC-2", "ACC-3"],
                    "department_id": ["DEP-1"] * 3,
                    "account_type": ["OPERATING"] * 3,
                    "status": ["ACTIVE"] * 3,
                    "region": ["WEST"] * 3,
                    "credit_limit_usd": [1000.0] * 3,
                }
            ),
            "vendors": read_parquet(settings.dataset_path("silver", "silver_vendors")),
            "contracts": read_parquet(settings.dataset_path("silver", "silver_contracts")),
            "locations": pd.DataFrame(
                {
                    "location_id": ["LOC-1"],
                    "city": ["Seattle"],
                    "state": ["WA"],
                    "country": ["US"],
                    "region": ["WEST"],
                }
            ),
            "transactions": read_parquet(settings.dataset_path("silver", "silver_transactions")),
        }
        result = load_graph(
            frames,
            settings,
            account_risk=read_parquet(settings.dataset_path("gold", "gold_account_risk")),
            vendor_risk=read_parquet(settings.dataset_path("gold", "gold_vendor_risk")),
        )
        assert result.loaded is True
        assert sum(result.nodes.values()) > 0

        stats = graph_statistics(settings)
        assert stats["available"] is True
        assert "Vendor" in stats["nodes"]

        cypher_result = high_risk_vendor_networks(settings=settings)
        assert cypher_result["backend"] == "neo4j"

    def test_load_is_idempotent(self, graph_lake, settings):
        """MERGE, not CREATE - re-running must not double the edges."""
        from src.core.io import read_parquet
        from src.graph.graph_loader import load_graph

        frames = {
            "vendors": read_parquet(settings.dataset_path("silver", "silver_vendors")),
        }
        load_graph(frames, settings, clear_first=True)
        first = graph_statistics(settings)["nodes"].get("Vendor", 0)

        load_graph(frames, settings, clear_first=False)
        second = graph_statistics(settings)["nodes"].get("Vendor", 0)

        assert first == second
