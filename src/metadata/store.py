"""Metadata store access layer.

One class fronts both backends. Callers never learn whether they are talking to
PostgreSQL or SQLite - they call :meth:`MetadataStore.record_run` and the store
figures out the rest. That matters because the same pipeline code runs on a
laptop with no containers and in an environment with a real database, and
neither should require a code change.

Two behaviours are worth calling out:

* **The store never raises into a pipeline.** If the metadata database is down,
  the load of financial data should not fail; the failure is logged and the run
  continues. Observability is important, but it is not more important than the
  data. Errors are surfaced through :attr:`MetadataStore.degraded`.
* **Writes are upserts keyed by ``run_id``**, so a pipeline that records its run
  at the start and updates it at the end produces one row, not two - and a
  re-run with the same id corrects the record rather than duplicating it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, delete, func, select
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.core.run_context import RunContext
from src.metadata.schema import (
    ALL_TABLES,
    data_quality_checks,
    data_quality_results,
    dataset_catalog,
    ingestion_ledger,
    lineage_edges,
    metadata_obj,
    model_registry,
    pipeline_runs,
    pipeline_stages,
)

log = get_logger(__name__)


class MetadataStore:
    """Read/write access to the operational metadata tables."""

    def __init__(self, settings: Settings | None = None, *, dsn: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.dsn = dsn or self.settings.postgres_dsn or self._sqlite_dsn()
        self.backend = "postgres" if self.dsn.startswith("postgresql") else "sqlite"
        self.degraded = False
        self._engine: Engine | None = None
        self._initialise()

    # -- connection ---------------------------------------------------------

    def _sqlite_dsn(self) -> str:
        path = Path(self.settings.paths.metadata) / "local" / "metadata.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"

    def _initialise(self) -> None:
        try:
            self._engine = create_engine(self.dsn, future=True, pool_pre_ping=True)
            metadata_obj.create_all(self._engine)
            log.info("metadata_store_ready", backend=self.backend)
        except SQLAlchemyError as exc:
            # A missing PostgreSQL is a normal local condition, not a crash.
            log.warning(
                "metadata_store_unavailable",
                backend=self.backend,
                error=str(exc)[:200],
                detail="pipeline metadata will not be persisted for this run",
            )
            self.degraded = True
            self._engine = None

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @property
    def available(self) -> bool:
        return self._engine is not None

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    # -- pipeline runs ------------------------------------------------------

    def record_run(self, context: RunContext) -> None:
        """Upsert a pipeline run and replace its stage rows."""
        if not self.available:
            return
        payload = context.to_dict()
        stages = payload.pop("stages", [])
        payload["parameters"] = _jsonable(payload.get("parameters"))
        payload["metrics"] = _jsonable(payload.get("metrics"))

        try:
            assert self._engine is not None
            with self._engine.begin() as connection:
                connection.execute(
                    delete(pipeline_runs).where(pipeline_runs.c.run_id == context.run_id)
                )
                connection.execute(pipeline_runs.insert().values(**payload))
                connection.execute(
                    delete(pipeline_stages).where(pipeline_stages.c.run_id == context.run_id)
                )
                if stages:
                    connection.execute(
                        pipeline_stages.insert(),
                        [
                            {
                                "run_id": context.run_id,
                                "sequence": index,
                                "stage": stage["stage"],
                                "duration_seconds": stage["duration_seconds"],
                                "records_in": stage["records_in"],
                                "records_out": stage["records_out"],
                                "status": stage["status"],
                                "detail": _jsonable(stage.get("detail")),
                            }
                            for index, stage in enumerate(stages, start=1)
                        ],
                    )
        except SQLAlchemyError as exc:
            self._handle_error("record_run", exc)

    def recent_runs(self, limit: int = 25, pipeline_name: str | None = None) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        query = select(pipeline_runs).order_by(pipeline_runs.c.start_time.desc()).limit(limit)
        if pipeline_name:
            query = query.where(pipeline_runs.c.pipeline_name == pipeline_name)
        return self._read(query)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        frame = self._read(select(pipeline_runs).where(pipeline_runs.c.run_id == run_id))
        if frame.empty:
            return None
        return frame.iloc[0].to_dict()

    def run_stages(self, run_id: str) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        return self._read(
            select(pipeline_stages)
            .where(pipeline_stages.c.run_id == run_id)
            .order_by(pipeline_stages.c.sequence)
        )

    # -- data quality -------------------------------------------------------

    def record_quality_result(self, result: dict[str, Any], checks: list[dict[str, Any]]) -> None:
        """Persist one dataset's quality scorecard plus its individual checks."""
        if not self.available:
            return
        try:
            assert self._engine is not None
            with self._engine.begin() as connection:
                connection.execute(data_quality_results.insert().values(**result))
                if checks:
                    connection.execute(data_quality_checks.insert(), checks)
        except SQLAlchemyError as exc:
            self._handle_error("record_quality_result", exc)

    def quality_history(self, dataset: str | None = None, limit: int = 200) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        query = (
            select(data_quality_results)
            .order_by(data_quality_results.c.evaluated_at.desc())
            .limit(limit)
        )
        if dataset:
            query = query.where(data_quality_results.c.dataset == dataset)
        return self._read(query)

    def failed_checks(self, run_id: str | None = None, limit: int = 200) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        query = (
            select(data_quality_checks)
            .where(data_quality_checks.c.passed.is_(False))
            .order_by(data_quality_checks.c.evaluated_at.desc())
            .limit(limit)
        )
        if run_id:
            query = query.where(data_quality_checks.c.run_id == run_id)
        return self._read(query)

    def checks_for_run(self, run_id: str) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        return self._read(select(data_quality_checks).where(data_quality_checks.c.run_id == run_id))

    # -- lineage ------------------------------------------------------------

    def record_lineage(self, edges: list[dict[str, Any]]) -> None:
        if not self.available or not edges:
            return
        try:
            assert self._engine is not None
            with self._engine.begin() as connection:
                connection.execute(lineage_edges.insert(), edges)
        except SQLAlchemyError as exc:
            self._handle_error("record_lineage", exc)

    def lineage_for(self, dataset: str) -> pd.DataFrame:
        """Every edge touching ``dataset``, upstream or downstream."""
        if not self.available:
            return pd.DataFrame()
        query = select(lineage_edges).where(
            (lineage_edges.c.target_dataset == dataset)
            | (lineage_edges.c.source_dataset == dataset)
        )
        return self._read(query)

    def all_lineage(self) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        return self._read(select(lineage_edges))

    # -- ingestion ledger (idempotency) -------------------------------------

    def already_ingested(self, source_file: str, content_hash: str) -> bool:
        """True when this exact file content has been loaded before."""
        if not self.available:
            return False
        try:
            assert self._engine is not None
            with self._engine.connect() as connection:
                found = connection.execute(
                    select(func.count())
                    .select_from(ingestion_ledger)
                    .where(
                        ingestion_ledger.c.source_file == source_file,
                        ingestion_ledger.c.content_hash == content_hash,
                    )
                ).scalar_one()
            return bool(found)
        except SQLAlchemyError as exc:
            self._handle_error("already_ingested", exc)
            return False

    def record_ingestion(self, entries: list[dict[str, Any]]) -> None:
        if not self.available or not entries:
            return
        try:
            assert self._engine is not None
            with self._engine.begin() as connection:
                for entry in entries:
                    # Delete-then-insert keeps the unique constraint happy when a
                    # file is re-ingested with --force.
                    connection.execute(
                        delete(ingestion_ledger).where(
                            ingestion_ledger.c.source_file == entry["source_file"],
                            ingestion_ledger.c.content_hash == entry["content_hash"],
                        )
                    )
                    connection.execute(ingestion_ledger.insert().values(**entry))
        except SQLAlchemyError as exc:
            self._handle_error("record_ingestion", exc)

    def ingestion_history(self, limit: int = 100) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        return self._read(
            select(ingestion_ledger).order_by(ingestion_ledger.c.ingested_at.desc()).limit(limit)
        )

    # -- model registry -----------------------------------------------------

    def register_model(self, entry: dict[str, Any]) -> None:
        """Register a model version and demote the previous active one."""
        if not self.available:
            return
        try:
            assert self._engine is not None
            with self._engine.begin() as connection:
                connection.execute(
                    model_registry.update()
                    .where(model_registry.c.model_name == entry["model_name"])
                    .values(is_active=False)
                )
                connection.execute(
                    delete(model_registry).where(
                        model_registry.c.model_version == entry["model_version"]
                    )
                )
                connection.execute(model_registry.insert().values(**entry))
        except SQLAlchemyError as exc:
            self._handle_error("register_model", exc)

    def active_model(self, model_name: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        frame = self._read(
            select(model_registry).where(
                model_registry.c.model_name == model_name,
                model_registry.c.is_active.is_(True),
            )
        )
        return None if frame.empty else frame.iloc[0].to_dict()

    def model_history(self, limit: int = 50) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        return self._read(
            select(model_registry).order_by(model_registry.c.trained_at.desc()).limit(limit)
        )

    # -- dataset catalogue --------------------------------------------------

    def upsert_dataset(self, entry: dict[str, Any]) -> None:
        if not self.available:
            return
        try:
            assert self._engine is not None
            with self._engine.begin() as connection:
                connection.execute(
                    delete(dataset_catalog).where(dataset_catalog.c.dataset == entry["dataset"])
                )
                connection.execute(dataset_catalog.insert().values(**entry))
        except SQLAlchemyError as exc:
            self._handle_error("upsert_dataset", exc)

    def catalog(self) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        return self._read(select(dataset_catalog).order_by(dataset_catalog.c.dataset))

    # -- maintenance --------------------------------------------------------

    def reset(self) -> None:
        """Drop and recreate every table. Used by tests and ``make clean``."""
        if not self.available:
            return
        assert self._engine is not None
        metadata_obj.drop_all(self._engine)
        metadata_obj.create_all(self._engine)

    def table_counts(self) -> dict[str, int]:
        """Row counts per table - the cheapest possible health check."""
        if not self.available:
            return {}
        counts: dict[str, int] = {}
        assert self._engine is not None
        try:
            with self._engine.connect() as connection:
                for table in ALL_TABLES:
                    counts[table.name] = int(
                        connection.execute(select(func.count()).select_from(table)).scalar_one()
                    )
        except SQLAlchemyError as exc:
            self._handle_error("table_counts", exc)
        return counts

    # -- internals ----------------------------------------------------------

    def _read(self, query: Any) -> pd.DataFrame:
        try:
            assert self._engine is not None
            with self._engine.connect() as connection:
                rows = connection.execute(query)
                return pd.DataFrame(rows.mappings().all())
        except SQLAlchemyError as exc:
            self._handle_error("read", exc)
            return pd.DataFrame()

    def _handle_error(self, operation: str, exc: Exception) -> None:
        self.degraded = True
        log.error("metadata_store_error", operation=operation, error=str(exc)[:300])


def _jsonable(value: Any) -> Any:
    """Coerce a value into something the JSON column can store."""
    if value is None:
        return None
    return json.loads(json.dumps(value, default=str))


@lru_cache(maxsize=1)
def get_metadata_store() -> MetadataStore:
    """Process-wide metadata store singleton."""
    return MetadataStore()


def reset_metadata_store_cache() -> None:
    get_metadata_store.cache_clear()


def utcnow() -> datetime:
    return datetime.now(UTC)
