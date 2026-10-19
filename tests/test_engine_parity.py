"""Engine parity: pandas and Spark must produce the same Silver layer.

The claim that both engines "interpret the same specification" is only worth
making if something checks it. These tests run the identical Bronze input
through both engines and compare the results row for row.

They are marked ``spark`` and skip automatically where PySpark is absent (the
host runs Python 3.14, which PySpark does not support). Inside the container:

    docker compose run --rm spark pytest tests/test_engine_parity.py -v
"""

from __future__ import annotations

import pytest

from src.core.io import read_parquet, write_parquet
from src.engines.base import get_engine
from src.silver.transformations import build_silver
from tests.conftest import spark_required


@pytest.fixture
def bronze_fixture(settings, run_context, transactions, reference_data):
    """Write a defect-carrying Bronze layer both engines will read."""
    from src.ingestion.base import stamp_audit_columns

    datasets = {
        "transaction": transactions,
        "account": reference_data.accounts,
        "vendor": reference_data.vendors,
        "agency": reference_data.agencies,
        "department": reference_data.departments,
        "contract": reference_data.contracts,
        "location": reference_data.locations,
    }
    from src.core.schema import table_name

    for dataset, frame in datasets.items():
        stamped = stamp_audit_columns(
            frame,
            dataset=dataset,
            source_system="TEST",
            source_format="csv",
            source_file=f"{dataset}.csv",
            run_id=run_context.run_id,
        )
        write_parquet(
            stamped,
            settings.dataset_path("bronze", table_name("bronze", dataset)),
            partition_by=["ingestion_date"],
        )
    return settings


class TestEngineSelection:
    def test_pandas_engine_is_the_default(self, settings):
        engine = get_engine("pandas", settings)
        assert engine.capabilities.name == "pandas"
        assert engine.capabilities.distributed is False
        assert engine.capabilities.supports_predicate_pushdown is True

    def test_unknown_engine_is_rejected(self, settings):
        from src.core.exceptions import EngineError

        with pytest.raises(EngineError, match="Unknown engine"):
            get_engine("hadoop", settings)

    def test_spark_requested_without_pyspark_gives_a_useful_message(self, settings, monkeypatch):
        import builtins

        from src.core.exceptions import EngineError

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("src.engines.spark_engine"):
                raise ImportError("No module named 'pyspark'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(EngineError) as info:
            get_engine("spark", settings)
        assert "container" in str(info.value)


class TestPandasEngineIO:
    def test_round_trip(self, settings, clean_transactions):
        engine = get_engine("pandas", settings)
        engine.write(clean_transactions, "silver", "silver_transactions")

        assert engine.exists("silver", "silver_transactions")
        assert engine.row_count("silver", "silver_transactions") == 10

        loaded = engine.read("silver", "silver_transactions")
        assert len(loaded) == 10

    def test_column_pruning_and_predicate_pushdown(self, settings, clean_transactions):
        engine = get_engine("pandas", settings)
        engine.write(clean_transactions, "silver", "silver_transactions")

        pruned = engine.read("silver", "silver_transactions", columns=["transaction_id", "amount"])
        assert list(pruned.columns) == ["transaction_id", "amount"]

        filtered = engine.read("silver", "silver_transactions", filters=[("currency", "=", "USD")])
        assert set(filtered["currency"]) == {"USD"}

    def test_missing_dataset_reports_zero_not_an_error(self, settings):
        engine = get_engine("pandas", settings)
        assert engine.exists("gold", "not_built_yet") is False
        assert engine.row_count("gold", "not_built_yet") == 0


@spark_required
@pytest.mark.spark
class TestSparkParity:
    """Both engines, same Bronze input, same Silver output."""

    @pytest.fixture
    def spark_engine(self, settings):
        from src.engines.spark_engine import SparkEngine

        engine = SparkEngine(settings)
        yield engine
        engine.stop()

    def _build_with_spark(self, dataset, spark_engine, run_context, settings):
        from src.pipelines.spark_pipeline import build_silver_spark

        return build_silver_spark(dataset, spark_engine, run_context, settings)

    def test_row_counts_match(self, bronze_fixture, spark_engine, run_context, store):
        settings = bronze_fixture

        for dataset in ("agency", "department", "location", "vendor", "contract", "account"):
            build_silver(dataset, run_context, settings, store=store)
        pandas_result = build_silver("transaction", run_context, settings, store=store)

        # Rebuild the same table with Spark, into a separate location.
        spark_result = self._build_with_spark("transaction", spark_engine, run_context, settings)

        assert spark_result["records_in"] == pandas_result.records_in
        assert (
            spark_result["records_out"] == pandas_result.records_out
        ), "engines disagree on how many records survive validation and dedup"

    def test_the_same_transactions_survive(self, bronze_fixture, spark_engine, run_context, store):
        settings = bronze_fixture
        for dataset in ("agency", "department", "location", "vendor", "contract", "account"):
            build_silver(dataset, run_context, settings, store=store)
        build_silver("transaction", run_context, settings, store=store)
        pandas_ids = set(
            read_parquet(settings.dataset_path("silver", "silver_transactions"))["transaction_id"]
        )

        self._build_with_spark("transaction", spark_engine, run_context, settings)
        spark_ids = set(
            spark_engine.read("silver", "silver_transactions")
            .select("transaction_id")
            .toPandas()["transaction_id"]
        )

        assert pandas_ids == spark_ids, (
            f"only in pandas: {sorted(pandas_ids - spark_ids)[:5]}; "
            f"only in spark: {sorted(spark_ids - pandas_ids)[:5]}"
        )

    def test_usd_conversion_agrees(self, bronze_fixture, spark_engine, run_context, store):
        settings = bronze_fixture
        build_silver("transaction", run_context, settings, store=store)
        pandas_total = (
            read_parquet(settings.dataset_path("silver", "silver_transactions"))["amount_usd"]
            .astype(float)
            .sum()
        )

        self._build_with_spark("transaction", spark_engine, run_context, settings)
        spark_total = float(
            spark_engine.read("silver", "silver_transactions")
            .selectExpr("sum(amount_usd) AS total")
            .toPandas()["total"]
            .iloc[0]
        )

        assert pandas_total == pytest.approx(spark_total, rel=1e-6)

    def test_quarantine_counts_agree(self, bronze_fixture, spark_engine, run_context, store):
        settings = bronze_fixture
        pandas_result = build_silver("transaction", run_context, settings, store=store)
        spark_result = self._build_with_spark("transaction", spark_engine, run_context, settings)
        assert spark_result["quarantined"] == pytest.approx(
            pandas_result.records_quarantined, rel=0.02
        )


@spark_required
@pytest.mark.spark
class TestSparkEngine:
    def test_session_is_tuned(self, settings):
        from src.engines.spark_engine import SparkEngine

        engine = SparkEngine(settings)
        try:
            conf = engine.session.conf
            assert conf.get("spark.sql.adaptive.enabled") == "true"
            assert conf.get("spark.sql.sources.partitionOverwriteMode") == "dynamic"
            assert engine.capabilities.distributed is True
        finally:
            engine.stop()

    def test_delta_or_parquet_is_reported_honestly(self, settings):
        from src.engines.spark_engine import SparkEngine, delta_available

        engine = SparkEngine(settings)
        try:
            assert engine.storage_format == ("delta" if delta_available() else "parquet")
            assert engine.capabilities.supports_delta == delta_available()
        finally:
            engine.stop()
