"""Shared pytest fixtures.

Every test that touches the lake gets an **isolated temporary lake**. Tests that
share a directory pass individually and fail together, and a test suite that
writes into ``data/`` destroys the developer's working state - both of which are
worse than slow tests.

The fixtures build small but *realistic* data: the same generators the platform
ships with, at a tiny scale, including the injected defects. Testing quality
logic against clean synthetic data would prove nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from src.core.config import Paths, Settings, load_settings
from src.core.run_context import RunContext
from src.data_generation.base import GeneratorContext
from src.metadata.store import MetadataStore


@pytest.fixture(scope="session")
def base_settings() -> Settings:
    """The real configuration, loaded once."""
    return load_settings()


@pytest.fixture
def settings(base_settings: Settings, tmp_path: Path) -> Settings:
    """Settings pointing at an isolated temporary lake."""
    paths = Paths(
        raw=tmp_path / "raw",
        bronze=tmp_path / "lake" / "bronze",
        silver=tmp_path / "lake" / "silver",
        gold=tmp_path / "lake" / "gold",
        quarantine=tmp_path / "quarantine",
        sample=tmp_path / "sample",
        models=tmp_path / "models",
        metadata=tmp_path / "metadata",
        streaming_inbox=tmp_path / "raw" / "streaming",
        streaming_checkpoint=tmp_path / "lake" / "_checkpoints",
        duckdb=tmp_path / "lake" / "analytics.duckdb",
    )
    paths.ensure()
    # postgres_dsn is cleared so tests always use the SQLite fallback: a test
    # suite that silently writes into a developer's PostgreSQL is a nasty
    # surprise, and CI has no PostgreSQL at all.
    return base_settings.model_copy(update={"paths": paths, "postgres_dsn": None})


@pytest.fixture
def store(settings: Settings) -> Iterator[MetadataStore]:
    metadata_store = MetadataStore(settings)
    yield metadata_store
    metadata_store.close()


@pytest.fixture
def run_context() -> RunContext:
    return RunContext(pipeline_name="test_pipeline", pipeline_version="1.0.0", seed=42)


@pytest.fixture
def generator(settings: Settings) -> GeneratorContext:
    return GeneratorContext(settings=settings, seed=42)


# ---------------------------------------------------------------------------
# data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_data(generator: GeneratorContext):
    """A tiny but complete set of dimensions."""
    from src.data_generation.generate_all import build_reference_data

    return build_reference_data(generator, agencies=3, vendors=12, contracts=20, accounts=15)


@pytest.fixture
def transactions(generator: GeneratorContext, reference_data):
    """500 transactions including the configured defects."""
    from src.data_generation.transactions import generate_transactions

    result = generate_transactions(
        generator,
        500,
        accounts=reference_data.accounts,
        vendors=reference_data.vendors,
        contracts=reference_data.contracts,
        departments=reference_data.departments,
        locations=reference_data.locations,
    )
    return result.transactions


@pytest.fixture
def clean_transactions() -> pd.DataFrame:
    """A small, hand-written, entirely valid frame.

    Used where a test needs to assert that valid data survives untouched -
    generated data always contains defects by design, which makes "nothing was
    rejected" impossible to assert against it.
    """
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:06d}" for i in range(1, 11)],
            "account_id": ["ACC-0000001"] * 5 + ["ACC-0000002"] * 5,
            "vendor_id": ["VEN-000001"] * 3 + ["VEN-000002"] * 7,
            "contract_id": [None] * 10,
            "department_id": ["DEP-00001"] * 10,
            "agency_id": ["AGY-0001"] * 10,
            "amount": [100.0, 250.5, 75.25, 1000.0, 42.0, 333.33, 89.99, 1500.0, 12.5, 660.0],
            "currency": ["USD"] * 8 + ["EUR", "GBP"],
            "transaction_timestamp": pd.date_range(
                "2026-01-01 09:00:00", periods=10, freq="h", tz="UTC"
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "posted_timestamp": pd.date_range(
                "2026-01-01 11:00:00", periods=10, freq="h", tz="UTC"
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "transaction_type": ["PAYMENT"] * 10,
            "payment_method": ["ACH"] * 10,
            "merchant_category": ["IT_SOFTWARE"] * 10,
            "description": ["Test payment"] * 10,
            "location_id": ["LOC-0001"] * 10,
            "channel": ["BATCH"] * 10,
            "status": ["SETTLED"] * 10,
        }
    )


@pytest.fixture
def bronze_transactions(clean_transactions: pd.DataFrame, run_context: RunContext) -> pd.DataFrame:
    """Clean transactions with Bronze audit columns attached."""
    from src.ingestion.base import stamp_audit_columns

    return stamp_audit_columns(
        clean_transactions,
        dataset="transaction",
        source_system="TEST_SOURCE",
        source_format="csv",
        source_file="test_transactions.csv",
        run_id=run_context.run_id,
    )


@pytest.fixture
def scored_transactions(clean_transactions: pd.DataFrame) -> pd.DataFrame:
    """Transactions carrying model and risk columns, as Gold builders expect."""
    frame = clean_transactions.copy()
    frame["transaction_timestamp"] = pd.to_datetime(frame["transaction_timestamp"], utc=True)
    frame["amount_usd"] = frame["amount"]
    frame["transaction_date"] = frame["transaction_timestamp"].dt.date.astype("string")
    frame["fiscal_year"] = 2026
    frame["anomaly_score"] = [0.1, 0.2, 0.9, 0.15, 0.3, 0.05, 0.95, 0.4, 0.2, 0.1]
    frame["is_anomaly"] = [False, False, True, False, False, False, True, False, False, False]
    frame["risk_score"] = [0, 10, 75, 5, 25, 0, 80, 30, 10, 0]
    frame["risk_level"] = [
        "LOW",
        "LOW",
        "CRITICAL",
        "LOW",
        "MEDIUM",
        "LOW",
        "CRITICAL",
        "MEDIUM",
        "LOW",
        "LOW",
    ]
    frame["risk_reasons"] = [[] for _ in range(10)]
    frame["risk_rule_hits"] = [[] for _ in range(10)]
    frame["model_version"] = "test_model_v1"
    frame["scored_at"] = pd.Timestamp.now(tz="UTC")
    frame["account_velocity_5min"] = 1.0
    return frame


@pytest.fixture
def source_files(settings: Settings, generator: GeneratorContext, reference_data, transactions):
    """Write a full multi-format source tree into the temporary lake."""
    from src.data_generation.generate_all import write_sources

    labels = pd.DataFrame(columns=["transaction_id", "is_fraud", "fraud_type"])
    write_sources(settings, reference_data, transactions, labels, events=[])
    return settings.paths.raw


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch: pytest.MonkeyPatch):
    """Keep test output readable."""
    monkeypatch.setenv("FIL_LOG_LEVEL", "ERROR")
    yield


@pytest.fixture
def cleanup_cached_singletons():
    """Drop cached global state between tests that patch the environment."""
    from src.core.config import reset_settings_cache
    from src.metadata.store import reset_metadata_store_cache

    reset_settings_cache()
    reset_metadata_store_cache()
    yield
    reset_settings_cache()
    reset_metadata_store_cache()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes more than a couple of seconds")


def _spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def _neo4j_available() -> bool:
    try:
        from src.graph.neo4j_client import graph_available

        return graph_available(use_cache=False)
    except Exception:
        return False


spark_required = pytest.mark.skipif(
    not _spark_available(), reason="PySpark is not installed (it runs in the spark container)"
)
neo4j_required = pytest.mark.skipif(not _neo4j_available(), reason="Neo4j is not reachable")
