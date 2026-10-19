"""The end-to-end batch pipeline.

    python -m src.pipelines.batch_pipeline
    python -m src.pipelines.batch_pipeline --source csv
    python -m src.pipelines.batch_pipeline --date 2026-08-15
    python -m src.pipelines.batch_pipeline --skip-ingest --no-train

Stages::

    extract -> bronze -> silver -> quality -> ml + risk -> gold -> metadata

Three properties are engineered deliberately:

**Metadata is always written.** The whole run is wrapped so that a failure still
produces a ``pipeline_runs`` row with the error and the stage timings up to the
failure point. A pipeline that records nothing when it breaks is a pipeline you
debug by guesswork.

**Stages fail soft by default.** ``pipeline.fail_fast`` is false, so a broken
source or a failed Gold dataset degrades the run rather than aborting it. The
run is then marked ``SUCCESS_WITH_WARNINGS``, which is honest - it neither
pretends everything worked nor discards the parts that did.

**Every stage is re-runnable.** Bronze skips already-ingested content, Silver
and Gold overwrite deterministically. Running the pipeline twice produces the
same lake, not double the rows.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.bronze.loader import BronzeLoader
from src.core.config import Settings, get_settings
from src.core.exceptions import LakehouseError
from src.core.logging import bind_run_context, clear_run_context, configure_logging, get_logger
from src.core.run_context import RunContext
from src.data_quality.validator import DataQualityValidator, QualityRunResult
from src.gold.builder import GoldBuildResult, build_all_gold, load_silver
from src.ingestion.sources import read_source, select_sources
from src.metadata.lineage import LineageCollector
from src.metadata.store import MetadataStore, get_metadata_store
from src.silver.spec import SILVER_BUILD_ORDER
from src.silver.transformations import SilverResult, build_all_silver

log = get_logger(__name__)

PIPELINE_NAME = "financial_intelligence_batch"


@dataclass
class BatchPipelineResult:
    """Everything one batch run produced."""

    context: RunContext
    bronze: dict[str, Any] = field(default_factory=dict)
    silver: dict[str, SilverResult] = field(default_factory=dict)
    quality: QualityRunResult | None = None
    gold: GoldBuildResult | None = None
    lineage_edges: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.context.status

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.context.run_id,
            "status": self.context.status,
            "duration_seconds": self.context.duration_seconds,
            "records_read": self.context.records_read,
            "records_written": self.context.records_written,
            "records_quarantined": self.context.records_quarantined,
            "quality_score": self.quality.overall if self.quality else None,
            "gold_datasets": len(self.gold.datasets) if self.gold else 0,
            "warnings": len(self.warnings),
        }


class BatchPipeline:
    """Sequences the batch stages and owns the run's metadata."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: MetadataStore | None = None,
        *,
        source: str = "all",
        target_date: date | None = None,
        train_model: bool = True,
        force_ingest: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store if store is not None else get_metadata_store()
        self.source = source
        self.target_date = target_date
        self.train_model = train_model
        self.force_ingest = force_ingest

        self.context = RunContext(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=self.settings.pipeline.version,
            source=source,
            engine=self.settings.engine,
            config_version=self.settings.pipeline.version,
            seed=self.settings.generation.seed,
            parameters={
                "source": source,
                "date": target_date.isoformat() if target_date else None,
                "train_model": train_model,
                "force_ingest": force_ingest,
                "engine": self.settings.engine,
            },
        )
        self.lineage = LineageCollector(
            self.context.run_id, PIPELINE_NAME, self.settings.pipeline.version
        )
        self.warnings: list[str] = []

    # -- orchestration ------------------------------------------------------

    def run(self, *, skip_ingest: bool = False, skip_gold: bool = False) -> BatchPipelineResult:
        """Execute the pipeline. Records metadata whatever the outcome."""
        self.settings.paths.ensure()
        bind_run_context(pipeline=PIPELINE_NAME, run_id=self.context.run_id)
        log.info(
            "pipeline_started",
            source=self.source,
            engine=self.settings.engine,
            target_date=str(self.target_date) if self.target_date else None,
        )

        result = BatchPipelineResult(context=self.context)

        # Record the run as RUNNING before any work starts. Two reasons: an
        # in-flight run is visible to monitoring rather than appearing only once
        # it finishes, and a process killed mid-run leaves a RUNNING row that
        # says so, instead of leaving no trace at all. record_run() is keyed by
        # run_id and replaces, so the final write updates this row.
        self._record_metadata()

        try:
            if not skip_ingest:
                result.bronze = self._stage("ingest_and_bronze", self._ingest_to_bronze)
            else:
                log.info("stage_skipped", stage="ingest_and_bronze")

            result.silver = self._stage("silver", self._build_silver)
            result.quality = self._stage("data_quality", self._run_quality)

            if not skip_gold:
                result.gold = self._stage("ml_risk_and_gold", self._build_gold)

            result.lineage_edges = self._stage("lineage", self._persist_lineage)

            if self.warnings:
                self.context.succeed_with_warnings("; ".join(self.warnings[:5]))
            else:
                self.context.finish("SUCCESS")

        except Exception as exc:
            log.error(
                "pipeline_failed",
                error=str(exc)[:400],
                error_type=type(exc).__name__,
            )
            self.context.fail(exc)
            result.warnings = self.warnings
            self._record_metadata()
            clear_run_context()
            raise

        result.warnings = self.warnings
        self._record_metadata()
        log.info("pipeline_completed", **result.summary())
        clear_run_context()
        return result

    def _stage(self, name: str, function: Any) -> Any:
        """Run one stage, timing it and recording it on the context."""
        started = time.perf_counter()
        try:
            output = function()
        except Exception as exc:
            duration = time.perf_counter() - started
            self.context.add_stage(name, duration, status="FAILED", error=str(exc)[:300])
            if self.settings.pipeline.fail_fast:
                raise
            # Fail soft: record, warn, and let the remaining stages run.
            self.warnings.append(f"{name}: {type(exc).__name__}: {exc}"[:200])
            log.error("stage_error", stage=name, error=str(exc)[:300])
            return None
        duration = time.perf_counter() - started
        self.context.add_stage(name, duration, status="SUCCESS")
        return output

    # -- stages -------------------------------------------------------------

    def _ingest_to_bronze(self) -> dict[str, Any]:
        """Read every selected source and load it into Bronze."""
        definitions = select_sources(self.settings, requested=self.source)
        loader = BronzeLoader(
            self.context.run_id, self.settings, self.store, force=self.force_ingest
        )

        by_dataset: dict[str, list] = {}
        source_errors = 0
        for definition in definitions:
            results = read_source(definition, self.settings, self.context.run_id)
            for item in results:
                source_errors += len(item.errors)
            by_dataset.setdefault(definition.dataset, []).extend(results)

        outcomes: dict[str, Any] = {}
        for dataset, results in by_dataset.items():
            load = loader.load(dataset, results)
            outcomes[load.table] = load.summary()
            self.context.count(read=load.records_written)

            source_files = {r.source_file for r in results if r.record_count}
            for source_file in source_files:
                self.lineage.add(
                    source_file,
                    load.table,
                    ["schema_mapping", "audit_stamping", "content_hash", "idempotency_check"],
                    source_layer="source",
                    target_layer="bronze",
                    records_in=sum(r.record_count for r in results if r.source_file == source_file),
                    records_out=load.records_written,
                )

        if source_errors:
            self.warnings.append(f"{source_errors} source-level read errors")
        log.info("bronze_stage_complete", tables=len(outcomes), source_errors=source_errors)
        return outcomes

    def _build_silver(self) -> dict[str, SilverResult]:
        results = build_all_silver(
            self.context,
            self.settings,
            store=self.store,
            lineage=self.lineage,
            datasets=SILVER_BUILD_ORDER,
        )
        for result in results.values():
            self.context.count(written=result.records_out, quarantined=result.records_quarantined)
        return results

    def _run_quality(self) -> QualityRunResult:
        validator = DataQualityValidator(self.context.run_id, self.settings, self.store)
        outcome = validator.evaluate_all()
        self.context.add_metric("data_quality_score", outcome.overall)
        self.context.add_metric(
            "data_quality_by_dataset",
            {name: card.overall_score for name, card in outcome.scorecards.items()},
        )

        if outcome.blocking_failures:
            names = sorted({c.check_name for c in outcome.blocking_failures})
            self.warnings.append(f"blocking quality checks failed: {', '.join(names[:4])}")
        if outcome.overall < self.settings.quality.thresholds.overall_fail:
            self.warnings.append(
                f"overall quality {outcome.overall:.2f}% below the "
                f"{self.settings.quality.thresholds.overall_fail}% floor"
            )
        return outcome

    def _build_gold(self) -> GoldBuildResult:
        frames = load_silver(self.settings)
        if frames.get("transactions", pd.DataFrame()).empty:
            self.warnings.append("no Silver transactions available; Gold layer skipped")
            return GoldBuildResult()
        return build_all_gold(
            self.context,
            self.settings,
            store=self.store,
            lineage=self.lineage,
            frames=frames,
            train_model=self.train_model,
        )

    def _persist_lineage(self) -> int:
        written = self.lineage.persist(self.store)
        self.lineage.export_json(self.settings)
        return written

    def _record_metadata(self) -> None:
        """Persist the run. Never raises - metadata must not break the pipeline."""
        try:
            self.store.record_run(self.context)
        except Exception as exc:
            log.error("run_metadata_not_recorded", error=str(exc)[:200])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_batch_pipeline(
    settings: Settings | None = None,
    store: MetadataStore | None = None,
    *,
    source: str = "all",
    target_date: date | None = None,
    train_model: bool = True,
    skip_ingest: bool = False,
    skip_gold: bool = False,
    force_ingest: bool = False,
) -> BatchPipelineResult:
    """Programmatic entry point, used by the demo and the tests.

    ``store`` is injectable rather than always resolved from the global
    singleton. Without it, a caller that passes custom ``settings`` still gets a
    metadata store built from the *process-wide* configuration - so a test with
    an isolated lake would quietly write its run metadata into the developer's
    real database, and then fail to find it.
    """
    pipeline = BatchPipeline(
        settings,
        store,
        source=source,
        target_date=target_date,
        train_model=train_model,
        force_ingest=force_ingest,
    )
    return pipeline.run(skip_ingest=skip_ingest, skip_gold=skip_gold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipelines.batch_pipeline",
        description="Run the end-to-end batch pipeline (Bronze -> Silver -> Gold).",
    )
    parser.add_argument(
        "--source", default="all", help="all | csv | json | xml | excel | api | a feed name"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD), recorded in the run parameters.",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Rebuild Silver and Gold from existing Bronze."
    )
    parser.add_argument("--skip-gold", action="store_true", help="Stop after data quality.")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Score with the registered model instead of retraining.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-ingest sources even if their content was already loaded.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first stage error instead of degrading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    if args.fail_fast:
        # Settings are frozen, so a copy is made rather than mutated in place.
        settings = settings.model_copy(
            update={"pipeline": settings.pipeline.model_copy(update={"fail_fast": True})}
        )

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None

    try:
        result = run_batch_pipeline(
            settings,
            source=args.source,
            target_date=target_date,
            train_model=not args.no_train,
            skip_ingest=args.skip_ingest,
            skip_gold=args.skip_gold,
            force_ingest=args.force_ingest,
        )
    except LakehouseError as exc:
        print(f"\nPipeline failed: {exc}")
        return 1
    except Exception as exc:
        print(f"\nPipeline failed: {type(exc).__name__}: {exc}")
        return 1

    _print_summary(result)
    return 0 if result.status.startswith("SUCCESS") else 1


def _print_summary(result: BatchPipelineResult) -> None:
    context = result.context
    print("\n" + "=" * 56)
    print("FINANCIAL INTELLIGENCE PIPELINE")
    print("=" * 56)
    print(f"Run ID:              {context.run_id}")
    print(f"Engine:              {context.engine}")
    print(f"Records read:        {context.records_read:>12,}")
    print(f"Silver records:      {context.records_written:>12,}")
    print(f"Quarantined:         {context.records_quarantined:>12,}")

    if result.gold and result.gold.scored_transactions is not None:
        scored = result.gold.scored_transactions
        if not scored.empty:
            anomalies = int(scored["is_anomaly"].sum())
            levels = scored["risk_level"].value_counts()
            print(f"Anomalies:           {anomalies:>12,}")
            print(f"High risk:           {int(levels.get('HIGH', 0)):>12,}")
            print(f"Critical risk:       {int(levels.get('CRITICAL', 0)):>12,}")

    if result.quality:
        print(f"Data quality:        {result.quality.overall:>11.2f}%")
    print(f"Duration:            {context.duration_seconds:>11.1f}s")
    print(f"Status:              {context.status}")
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:5]:
            print(f"  - {warning}")
    print("=" * 56)


if __name__ == "__main__":
    sys.exit(main())
