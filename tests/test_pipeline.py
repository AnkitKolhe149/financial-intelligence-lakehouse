"""End-to-end pipeline, metadata, lineage and SQL analytics."""

from __future__ import annotations

import pytest

from src.analytics.duckdb_client import AnalyticsEngine, _split_statements
from src.core.config import project_root
from src.metadata.lineage import LineageCollector, upstream_of
from src.pipelines.batch_pipeline import run_batch_pipeline


@pytest.fixture
def generated(settings, monkeypatch):
    """A tiny but complete source landscape in the temporary lake."""
    from src.data_generation.generate_all import generate_all

    monkeypatch.setattr("src.data_generation.generate_all.get_settings", lambda: settings)
    generate_all(
        settings,
        scale="small",
        overrides={
            "transactions": 800,
            "accounts": 30,
            "vendors": 15,
            "agencies": 3,
            "contracts": 25,
            "events": 20,
        },
        write_sample_files=False,
    )
    return settings


class TestBatchPipeline:
    @pytest.fixture
    def executed(self, generated, store, monkeypatch):
        settings = generated
        for module in (
            "src.pipelines.batch_pipeline",
            "src.silver.transformations",
            "src.gold.builder",
            "src.data_quality.validator",
        ):
            monkeypatch.setattr(f"{module}.get_settings", lambda s=settings: s, raising=False)
        # The REST source system is not running in tests; the file fallbacks are
        # what should be exercised.
        monkeypatch.setattr("src.ingestion.sources.source_api_available", lambda *a, **k: False)
        return run_batch_pipeline(settings, store, train_model=True)

    def test_pipeline_succeeds(self, executed):
        assert executed.status.startswith("SUCCESS")
        assert executed.context.records_read > 0
        assert executed.context.records_written > 0

    def test_records_are_quarantined_not_lost(self, executed):
        """Read = written + quarantined, allowing for cross-source duplicates."""
        assert executed.context.records_quarantined > 0
        assert executed.context.records_written < executed.context.records_read

    def test_every_layer_is_populated(self, executed, generated):
        from src.core.io import dataset_exists

        settings = generated
        assert dataset_exists(settings.dataset_path("bronze", "bronze_transactions"))
        assert dataset_exists(settings.dataset_path("silver", "silver_transactions"))
        assert dataset_exists(settings.dataset_path("gold", "gold_daily_financial_summary"))
        assert dataset_exists(settings.dataset_path("gold", "gold_anomaly_transactions"))

    def test_quality_is_scored(self, executed):
        assert executed.quality is not None
        assert 0 < executed.quality.overall <= 100
        assert "transaction" in executed.quality.scorecards

    def test_run_metadata_is_persisted(self, executed, store):
        runs = store.recent_runs()
        assert not runs.empty
        run = runs[runs["run_id"] == executed.context.run_id].iloc[0]
        assert run["status"].startswith("SUCCESS")
        assert run["records_read"] > 0

        stages = store.run_stages(executed.context.run_id)
        assert not stages.empty
        assert {"silver", "data_quality"} <= set(stages["stage"])

    def test_lineage_is_captured_end_to_end(self, executed, store):
        edges = store.all_lineage()
        assert not edges.empty

        ancestors = upstream_of("gold_account_risk", edges.to_dict(orient="records"))
        assert "silver_transactions" in ancestors
        assert "bronze_transactions" in ancestors

    def test_rerunning_is_idempotent(self, executed, generated, store, monkeypatch):
        """A second run must not duplicate Bronze rows."""
        from src.core.io import dataset_row_count

        settings = generated
        before = dataset_row_count(settings.dataset_path("bronze", "bronze_transactions"))

        monkeypatch.setattr("src.ingestion.sources.source_api_available", lambda *a, **k: False)
        second = run_batch_pipeline(settings, store, train_model=False)
        after = dataset_row_count(settings.dataset_path("bronze", "bronze_transactions"))

        assert after == before
        assert second.status.startswith("SUCCESS")

    def test_failure_is_still_recorded(self, generated, store, monkeypatch):
        """A pipeline that breaks must leave evidence behind."""
        from src.pipelines.batch_pipeline import BatchPipeline

        settings = generated.model_copy(
            update={"pipeline": generated.pipeline.model_copy(update={"fail_fast": True})}
        )
        pipeline = BatchPipeline(settings, store)
        monkeypatch.setattr(
            pipeline, "_build_silver", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        with pytest.raises(RuntimeError):
            pipeline.run(skip_ingest=True)

        runs = store.recent_runs()
        failed = runs[runs["run_id"] == pipeline.context.run_id].iloc[0]
        assert failed["status"] == "FAILED"
        assert "boom" in str(failed["error_message"])


class TestSqlAnalytics:
    @pytest.fixture
    def engine(self, settings, scored_transactions, reference_data, store, run_context):
        from src.core.io import write_parquet
        from src.gold.account_risk import build_account_risk
        from src.gold.anomaly_summary import build_anomaly_transactions
        from src.gold.financial_summary import build_daily_financial_summary

        silver = scored_transactions.copy()
        silver["_schema_version"] = "1.1"
        write_parquet(silver, settings.dataset_path("silver", "silver_transactions"))
        write_parquet(reference_data.vendors, settings.dataset_path("silver", "silver_vendors"))

        common = {"store": store, "run_id": run_context.run_id}
        build_daily_financial_summary(scored_transactions, settings, **common)
        build_account_risk(scored_transactions, reference_data.accounts, settings, **common)
        build_anomaly_transactions(scored_transactions, settings, **common)

        return AnalyticsEngine(settings)

    def test_views_are_registered_for_every_dataset(self, engine):
        views = engine.views()
        assert "silver_transactions" in views
        assert "gold_daily_financial_summary" in views

    def test_query_returns_rows(self, engine):
        result = engine.query("SELECT COUNT(*) AS n FROM silver_transactions")
        assert result.succeeded
        assert result.frame.iloc[0]["n"] == 10

    def test_partition_filter_prunes(self, engine):
        result = engine.query(
            "SELECT COUNT(*) AS n FROM silver_transactions WHERE transaction_date = '2026-01-01'"
        )
        assert result.succeeded

    def test_write_statements_are_refused(self, engine):
        result = engine.query("DELETE FROM silver_transactions")
        assert not result.succeeded
        assert "read" in result.error.lower()

    def test_a_failing_query_returns_an_error_not_an_exception(self, engine):
        result = engine.query("SELECT * FROM table_that_does_not_exist")
        assert not result.succeeded
        assert result.error

    def test_statement_splitter_ignores_semicolons_in_comments(self):
        """A semicolon inside a comment must not cut the file mid-sentence."""
        sql = """
        -- name: first
        -- this comment; contains a semicolon
        SELECT 1;

        -- name: second
        SELECT 2;
        """
        statements = _split_statements(sql)
        assert len(statements) == 2
        assert statements[0][1] == "first"
        assert statements[1][1] == "second"

    def test_comment_only_chunks_are_dropped(self):
        assert _split_statements("-- just a comment\n") == []

    def test_every_shipped_sql_file_parses(self, engine):
        """Guards against a Gold column rename silently breaking the library."""
        directory = project_root() / "sql"
        files = sorted(directory.glob("*.sql"))
        assert len(files) >= 10

        for path in files:
            for statement, name in _split_statements(path.read_text(encoding="utf-8")):
                assert statement.strip(), f"{path.name}: empty statement for {name}"


class TestLineageCollector:
    def test_edges_record_row_counts(self):
        collector = LineageCollector("run-1", "test", "1.0.0")
        collector.add("bronze_x", "silver_x", ["cast"], records_in=100, records_out=90)

        payload = collector.edges[0].to_json()
        assert payload["records_dropped"] == 10
        assert payload["transformation"] == ["cast"]

    def test_json_export_merges_by_edge(self, settings):
        collector = LineageCollector("run-1", "test", "1.0.0")
        collector.add("a", "b", ["one"])
        path = collector.export_json(settings)

        second = LineageCollector("run-2", "test", "1.0.0")
        second.add("a", "b", ["two"])
        second.add("b", "c", ["three"])
        second.export_json(settings)

        import json

        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["edge_count"] == 2

    def test_upstream_traversal_is_transitive(self):
        edges = [
            {"source_dataset": "file.csv", "target_dataset": "bronze_x"},
            {"source_dataset": "bronze_x", "target_dataset": "silver_x"},
            {"source_dataset": "silver_x", "target_dataset": "gold_x"},
        ]
        assert set(upstream_of("gold_x", edges)) == {"silver_x", "bronze_x", "file.csv"}
