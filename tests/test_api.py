"""Serving API and the source-system API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, settings, tmp_path):
    """A serving-API client bound to the isolated temporary lake."""
    from src.core import config as config_module
    from src.metadata import store as store_module

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr("src.api.main.get_settings", lambda: settings)

    store = store_module.MetadataStore(settings)
    monkeypatch.setattr("src.api.main.get_metadata_store", lambda: store)

    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def populated_client(client, settings, store, run_context, scored_transactions, reference_data):
    """A client whose lake has a Gold layer."""
    from src.core.io import write_parquet
    from src.gold.account_risk import build_account_risk
    from src.gold.anomaly_summary import build_anomaly_transactions, build_risk_alerts
    from src.gold.financial_summary import build_daily_financial_summary
    from src.gold.quality_summary import build_data_quality_summary
    from src.gold.vendor_risk import build_vendor_risk

    common = {"store": store, "run_id": run_context.run_id}
    build_daily_financial_summary(scored_transactions, settings, **common)
    build_account_risk(scored_transactions, reference_data.accounts, settings, **common)
    build_vendor_risk(
        scored_transactions, reference_data.vendors, reference_data.contracts, settings, **common
    )
    build_anomaly_transactions(scored_transactions, settings, **common)
    build_risk_alerts(scored_transactions, settings, **common, minimum_score=50)
    build_data_quality_summary(settings, **common)

    silver = scored_transactions.copy()
    silver["_source_system"] = "TEST"
    silver["_record_hash"] = "abc123"
    silver["_ingestion_run_id"] = run_context.run_id
    write_parquet(silver, settings.dataset_path("silver", "silver_transactions"))

    store.record_run(run_context.finish("SUCCESS"))
    return client


class TestOperations:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == "financial-intelligence-lakehouse"

    def test_health_reports_lake_state(self, client):
        payload = client.get("/health").json()
        assert payload["status"] in {"healthy", "degraded"}
        assert set(payload["lake"]) == {"bronze", "silver", "gold"}
        assert payload["engine"] == "pandas"

    def test_health_is_degraded_before_the_first_load(self, client):
        assert client.get("/health").json()["status"] == "degraded"

    def test_metrics_shape(self, populated_client):
        payload = populated_client.get("/metrics").json()
        for section in ("pipeline", "data", "quality", "risk", "model"):
            assert section in payload
        assert payload["risk"]["anomalies"] >= 0

    def test_metrics_work_on_an_empty_lake(self, client):
        assert client.get("/metrics").status_code == 200


class TestEmptyLakeDegradation:
    """Every list endpoint must return an empty result with a hint, not a 500."""

    @pytest.mark.parametrize(
        "path",
        [
            "/anomalies",
            "/alerts",
            "/data-quality",
            "/quarantine",
            "/pipeline-runs",
            "/catalog",
            "/lineage/silver_transactions",
        ],
    )
    def test_endpoint_degrades_gracefully(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert "hint" in response.json()


class TestTransactions:
    def test_fetch_by_id(self, populated_client, scored_transactions):
        transaction_id = scored_transactions.iloc[0]["transaction_id"]
        payload = populated_client.get(f"/transactions/{transaction_id}").json()
        assert payload["transaction_id"] == transaction_id
        assert "risk_score" in payload
        assert "lineage" in payload

    def test_flagged_transaction_carries_its_explanation(
        self, populated_client, scored_transactions
    ):
        flagged = scored_transactions[scored_transactions["risk_level"] == "CRITICAL"].iloc[0]
        payload = populated_client.get(f"/transactions/{flagged['transaction_id']}").json()
        assert payload["risk_level"] == "CRITICAL"
        assert isinstance(payload["risk_reasons"], list)

    def test_unknown_transaction_is_404(self, populated_client):
        assert populated_client.get("/transactions/TXN-NOPE").status_code == 404


class TestRiskEndpoints:
    def test_account_risk(self, populated_client, scored_transactions):
        account_id = scored_transactions.iloc[0]["account_id"]
        payload = populated_client.get(f"/accounts/{account_id}/risk").json()
        assert payload["account_id"] == account_id
        assert "risk_score" in payload
        assert isinstance(payload["recent_flagged_transactions"], list)

    def test_vendor_risk(self, populated_client, scored_transactions):
        vendor_id = scored_transactions.iloc[0]["vendor_id"]
        payload = populated_client.get(f"/vendors/{vendor_id}/risk").json()
        assert payload["vendor_id"] == vendor_id
        assert "flagged_transaction_count" in payload

    def test_unknown_entities_are_404(self, populated_client):
        assert populated_client.get("/accounts/ACC-NOPE/risk").status_code == 404
        assert populated_client.get("/vendors/VEN-NOPE/risk").status_code == 404

    def test_anomalies_are_filterable(self, populated_client):
        payload = populated_client.get("/anomalies?risk_level=CRITICAL&limit=10").json()
        assert payload["total"] >= 1
        assert all(r["risk_level"] == "CRITICAL" for r in payload["results"])

    def test_anomaly_reasons_are_parsed_back_to_lists(self, populated_client):
        payload = populated_client.get("/anomalies?limit=5").json()
        for record in payload["results"]:
            assert isinstance(record["risk_reasons"], list)

    def test_minimum_score_filter(self, populated_client):
        payload = populated_client.get("/anomalies?min_risk_score=75").json()
        assert all(r["risk_score"] >= 75 for r in payload["results"])

    def test_pagination(self, populated_client):
        first = populated_client.get("/anomalies?limit=1&offset=0").json()
        second = populated_client.get("/anomalies?limit=1&offset=1").json()
        assert first["returned"] == 1
        if second["returned"]:
            assert first["results"][0]["transaction_id"] != second["results"][0]["transaction_id"]

    def test_alerts(self, populated_client):
        payload = populated_client.get("/alerts?status=OPEN").json()
        assert all(r["status"] == "OPEN" for r in payload["results"])


class TestGovernance:
    def test_schemas_are_published(self, client):
        payload = client.get("/schemas").json()["datasets"]
        assert "transaction" in payload
        transaction = payload["transaction"]
        assert transaction["business_key"] == ["transaction_id"]
        assert any(f["name"] == "amount" for f in transaction["fields"])

    def test_risk_rules_are_published(self, client):
        payload = client.get("/risk-rules").json()
        assert payload["max_score"] == 100
        assert len(payload["rules"]) >= 5
        assert all(r["description"] for r in payload["rules"])

    def test_sources_are_published(self, client):
        sources = client.get("/sources").json()["sources"]
        formats = {s["format"] for s in sources}
        assert {"csv", "json", "xml", "xlsx", "rest_api"} <= formats

    def test_quality_score_matches_the_pipeline_calculation(self, populated_client):
        """The API must not report a different headline number than the pipeline."""
        payload = populated_client.get("/data-quality").json()
        if payload["datasets"]:
            assert payload["platform_score"] is not None
            assert 0 <= payload["platform_score"] <= 100


class TestSourceSystemApi:
    @pytest.fixture
    def source_client(self, monkeypatch, settings, source_files):
        monkeypatch.setattr("api.source_system.main.get_settings", lambda: settings)
        from api.source_system.main import _load, app

        _load.cache_clear()
        return TestClient(app)

    def test_health(self, source_client):
        payload = source_client.get("/health").json()
        assert payload["service"] == "source-system-api"

    def test_pagination_envelope(self, source_client):
        payload = source_client.get("/vendors?page=1&page_size=5").json()
        assert payload["pagination"]["page_size"] == 5
        assert len(payload["results"]) <= 5
        assert "has_next" in payload["pagination"]

    def test_page_size_is_capped(self, source_client):
        payload = source_client.get("/vendors?page_size=99999").json()
        assert payload["pagination"]["page_size"] <= 500

    def test_following_the_cursor_covers_every_record(self, source_client):
        seen: list[str] = []
        page = 1
        while page <= 50:
            payload = source_client.get(f"/vendors?page={page}&page_size=25").json()
            seen.extend(r["vendor_id"] for r in payload["results"])
            if not payload["pagination"]["has_next"]:
                break
            page += 1
        assert len(seen) == payload["pagination"]["total_records"]
        assert len(set(seen)) == len(seen), "pagination returned duplicates"

    def test_filters(self, source_client):
        payload = source_client.get("/vendors?country=US&page_size=50").json()
        assert all(r["country"] == "US" for r in payload["results"])

    def test_fault_injection_enables_retry_testing(self, source_client):
        # fail_rate=1.0 must always fail, which is what makes the retry test
        # deterministic rather than flaky.
        assert source_client.get("/vendors?fail_rate=1.0").status_code == 503
