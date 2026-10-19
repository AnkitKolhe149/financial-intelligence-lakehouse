"""Relational schema for the operational metadata store.

Defined with SQLAlchemy Core (not the ORM) because these tables are written by
pipelines and read by dashboards - there are no domain objects with behaviour to
map, and Core keeps the generated SQL obvious.

Portability rules applied throughout:

* ``JSON`` rather than ``JSONB`` - PostgreSQL upgrades it internally, SQLite
  stores it as text, and neither needs a dialect branch in our code.
* ``DateTime(timezone=True)`` everywhere. Storing naive timestamps in a system
  whose whole job is temporal analysis is how you get an unreproducible bug.
* Explicit indexes on the columns the dashboard filters by, because "it was fast
  on 50 rows" is not a performance argument.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata_obj = MetaData()

# ---------------------------------------------------------------------------
# pipeline execution
# ---------------------------------------------------------------------------

pipeline_runs = Table(
    "pipeline_runs",
    metadata_obj,
    Column("run_id", String(64), primary_key=True),
    Column("pipeline_name", String(128), nullable=False),
    Column("pipeline_version", String(32), nullable=False),
    Column("config_version", String(32)),
    Column("model_version", String(64)),
    Column("engine", String(16)),
    Column("source", String(64)),
    Column("seed", Integer),
    Column("start_time", DateTime(timezone=True), nullable=False),
    Column("end_time", DateTime(timezone=True)),
    Column("duration_seconds", Float),
    Column("records_read", Integer, default=0),
    Column("records_written", Integer, default=0),
    Column("records_quarantined", Integer, default=0),
    Column("records_failed", Integer, default=0),
    Column("status", String(32), nullable=False),
    Column("error_message", Text),
    Column("host", String(128)),
    Column("python_version", String(32)),
    Column("platform_version", String(32)),
    Column("parameters", JSON),
    Column("metrics", JSON),
    Index("ix_pipeline_runs_name_start", "pipeline_name", "start_time"),
    Index("ix_pipeline_runs_status", "status"),
)

pipeline_stages = Table(
    "pipeline_stages",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("stage", String(64), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("duration_seconds", Float),
    Column("records_in", Integer),
    Column("records_out", Integer),
    Column("status", String(32)),
    Column("detail", JSON),
    Index("ix_pipeline_stages_run", "run_id"),
)

# ---------------------------------------------------------------------------
# data quality
# ---------------------------------------------------------------------------

data_quality_results = Table(
    "data_quality_results",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("dataset", String(128), nullable=False),
    Column("layer", String(16), nullable=False),
    Column("evaluated_at", DateTime(timezone=True), nullable=False),
    Column("overall_score", Float, nullable=False),
    Column("completeness", Float),
    Column("validity", Float),
    Column("uniqueness", Float),
    Column("integrity", Float),
    Column("freshness", Float),
    Column("volume", Float),
    Column("records_evaluated", Integer),
    Column("records_failed", Integer),
    Column("checks_total", Integer),
    Column("checks_passed", Integer),
    Column("status", String(16)),  # PASS | WARN | FAIL
    Index("ix_dq_results_dataset_time", "dataset", "evaluated_at"),
    Index("ix_dq_results_run", "run_id"),
)

data_quality_checks = Table(
    "data_quality_checks",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("dataset", String(128), nullable=False),
    Column("check_name", String(128), nullable=False),
    Column("dimension", String(32), nullable=False),
    Column("column_name", String(128)),
    Column("severity", String(16)),  # blocking | warning | info
    Column("passed", Boolean, nullable=False),
    Column("score", Float),
    Column("records_evaluated", Integer),
    Column("records_failed", Integer),
    Column("threshold", Float),
    Column("observed_value", Float),
    Column("message", Text),
    Column("evaluated_at", DateTime(timezone=True), nullable=False),
    Index("ix_dq_checks_run_dataset", "run_id", "dataset"),
    Index("ix_dq_checks_passed", "passed"),
)

# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------

lineage_edges = Table(
    "lineage_edges",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("source_dataset", String(200), nullable=False),
    Column("source_layer", String(32)),
    Column("target_dataset", String(200), nullable=False),
    Column("target_layer", String(32)),
    Column("transformations", JSON),
    Column("pipeline_name", String(128)),
    Column("pipeline_version", String(32)),
    Column("records_in", Integer),
    Column("records_out", Integer),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_lineage_target", "target_dataset"),
    Index("ix_lineage_source", "source_dataset"),
)

# ---------------------------------------------------------------------------
# ingestion ledger - the idempotency mechanism
# ---------------------------------------------------------------------------

ingestion_ledger = Table(
    "ingestion_ledger",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(128), nullable=False),
    Column("source_system", String(128), nullable=False),
    Column("source_file", String(400), nullable=False),
    # SHA-256 of the file's bytes (or of the payload for API sources). Together
    # with source_file this is what makes re-ingestion a no-op instead of a
    # duplicate load.
    Column("content_hash", String(64), nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("records", Integer),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Column("schema_version", String(16)),
    UniqueConstraint("source_file", "content_hash", name="uq_ingestion_ledger_file_hash"),
    Index("ix_ingestion_ledger_dataset", "dataset"),
)

# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------

model_registry = Table(
    "model_registry",
    metadata_obj,
    Column("model_version", String(64), primary_key=True),
    Column("model_name", String(64), nullable=False),
    Column("algorithm", String(64), nullable=False),
    Column("trained_at", DateTime(timezone=True), nullable=False),
    Column("training_rows", Integer),
    Column("feature_list", JSON),
    Column("hyperparameters", JSON),
    Column("training_window_start", DateTime(timezone=True)),
    Column("training_window_end", DateTime(timezone=True)),
    Column("evaluation", JSON),
    Column("artifact_path", String(400)),
    Column("run_id", String(64)),
    Column("is_active", Boolean, default=True),
    Index("ix_model_registry_name_active", "model_name", "is_active"),
)

# ---------------------------------------------------------------------------
# dataset catalogue - the local stand-in for Unity Catalog
# ---------------------------------------------------------------------------

dataset_catalog = Table(
    "dataset_catalog",
    metadata_obj,
    Column("dataset", String(200), primary_key=True),
    Column("layer", String(32), nullable=False),
    Column("description", Text),
    Column("owner", String(128)),
    Column("classification", String(32)),
    Column("schema_version", String(16)),
    Column("row_count", Integer),
    Column("size_bytes", Integer),
    Column("partition_columns", JSON),
    Column("column_schema", JSON),
    Column("last_updated", DateTime(timezone=True)),
    Column("last_run_id", String(64)),
    Index("ix_dataset_catalog_layer", "layer"),
)

ALL_TABLES = (
    pipeline_runs,
    pipeline_stages,
    data_quality_results,
    data_quality_checks,
    lineage_edges,
    ingestion_ledger,
    model_registry,
    dataset_catalog,
)
