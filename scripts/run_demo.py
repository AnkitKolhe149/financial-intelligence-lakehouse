"""One-command demonstration of the whole platform.

    python scripts/run_demo.py
    python scripts/run_demo.py --scale small --skip-streaming
    python scripts/run_demo.py --scale large --external

Runs every stage end to end and prints a summary built entirely from **measured
values** - no figure in the output is hard-coded. If a stage produces nothing,
the summary says zero rather than a plausible-looking number.

Optional components (the REST source system, USAspending, Neo4j) are probed and
skipped with a stated reason when unavailable. The demo is expected to complete
on a laptop with no containers running.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow `python scripts/run_demo.py` from a clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import Settings, get_settings
from src.core.io import dataset_row_count
from src.core.logging import configure_logging, get_logger
from src.metadata.store import MetadataStore

log = get_logger(__name__)

BANNER = "=" * 60


@dataclass
class DemoStep:
    name: str
    status: str = "pending"
    duration_seconds: float = 0.0
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class Demo:
    """Sequences the demonstration and collects measured results."""

    def __init__(self, settings: Settings, args: argparse.Namespace) -> None:
        self.settings = settings
        self.args = args
        self.store = MetadataStore(settings)
        self.steps: list[DemoStep] = []
        self.results: dict[str, Any] = {}

    # -- helpers ------------------------------------------------------------

    def step(self, name: str, function, *, optional: bool = False) -> Any:
        record = DemoStep(name=name)
        self.steps.append(record)
        print(f"\n>> {name}")
        started = time.perf_counter()
        try:
            outcome = function()
        except Exception as exc:
            record.duration_seconds = time.perf_counter() - started
            record.status = "skipped" if optional else "failed"
            record.detail = f"{type(exc).__name__}: {exc}"[:200]
            symbol = "-" if optional else "x"
            print(f"   {symbol} {record.detail}")
            if not optional:
                log.error("demo_step_failed", step=name, error=str(exc)[:300])
            return None
        record.duration_seconds = time.perf_counter() - started
        record.status = "ok"
        print(f"   done in {record.duration_seconds:.1f}s")
        return outcome

    # -- stages -------------------------------------------------------------

    def clean(self) -> None:
        """Remove the previous run's artefacts for a genuinely fresh demo.

        The metadata store is disposed first: it holds an open handle to the
        SQLite file, and on Windows an open handle makes the file undeletable.
        The store is reopened afterwards so later stages still record metadata.
        """
        self.store.close()

        targets = [
            self.settings.paths.bronze.parent,  # the whole lake
            self.settings.paths.quarantine,
            self.settings.paths.models,
            self.settings.paths.raw,  # regenerated in the next step
            Path(self.settings.paths.metadata) / "local",
        ]
        removed = []
        for directory in targets:
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(directory.name)

        self.settings.paths.ensure()
        self.store = MetadataStore(self.settings)
        print(f"   removed {len(removed)} directories: {', '.join(removed)}")

    def generate(self) -> dict[str, int]:
        from src.data_generation.generate_all import generate_all

        summary = generate_all(
            self.settings, scale=self.args.scale, write_sample_files=self.args.samples
        )
        self.results["generation"] = summary.counts
        self.results["fraud_labels"] = summary.fraud_labels
        for name, value in summary.counts.items():
            print(f"   {name:<16}{value:>10,}")
        return summary.counts

    def check_source_api(self) -> bool:
        from src.ingestion.api_ingestion import source_api_available

        available = source_api_available(self.settings, timeout=2.0)
        if available:
            print(f"   REST source system is up at {self.settings.services.source_api_base_url}")
        else:
            print(
                "   REST source system is not running; the file-based fallbacks "
                "will be used instead (start it with `make source-api`)"
            )
        self.results["source_api_available"] = available
        return available

    def fetch_external(self) -> dict[str, Any]:
        from src.ingestion.usaspending_client import fetch_or_fallback

        result = fetch_or_fallback(self.settings, days=90, max_pages=2)
        self.results["usaspending"] = {
            "source": result.source,
            "records": len(result.frame),
        }
        print(f"   source={result.source} records={len(result.frame):,}")
        if result.is_live:
            top = result.frame.nlargest(1, "award_amount_usd")
            if not top.empty:
                row = top.iloc[0]
                print(
                    f"   largest award: {row['recipient_name']} "
                    f"${float(row['award_amount_usd']):,.0f} ({row['awarding_agency_name']})"
                )
        return self.results["usaspending"]

    def run_pipeline(self) -> Any:
        from src.pipelines.batch_pipeline import run_batch_pipeline

        result = run_batch_pipeline(
            self.settings, self.store, source="all", train_model=True, force_ingest=True
        )
        self.results["pipeline"] = result
        print(
            f"   read={result.context.records_read:,} "
            f"silver={result.context.records_written:,} "
            f"quarantined={result.context.records_quarantined:,}"
        )
        return result

    def load_graph(self) -> dict[str, Any]:
        from src.core.io import read_parquet_or_empty
        from src.gold.builder import load_silver
        from src.graph.graph_loader import load_graph

        frames = load_silver(self.settings)
        result = load_graph(
            frames,
            self.settings,
            scored_transactions=(
                self.results["pipeline"].gold.scored_transactions
                if self.results.get("pipeline") and self.results["pipeline"].gold
                else None
            ),
            account_risk=read_parquet_or_empty(
                self.settings.dataset_path("gold", "gold_account_risk")
            ),
            vendor_risk=read_parquet_or_empty(
                self.settings.dataset_path("gold", "gold_vendor_risk")
            ),
        )
        self.results["graph"] = result
        if result.loaded:
            print(
                f"   nodes={sum(result.nodes.values()):,} "
                f"relationships={sum(result.relationships.values()):,}"
            )
        else:
            print(f"   skipped: {result.reason}")
        return result

    def run_streaming(self) -> dict[str, Any]:
        from src.streaming.event_generator import EventSimulator
        from src.streaming.streaming_pipeline import StreamingPipeline

        simulator = EventSimulator(self.settings, anomaly_rate=0.12, late_rate=0.06, batch_size=25)
        simulator.run(total_events=self.args.events, interval=0)
        print(f"   emitted {simulator.stats.events_emitted:,} events")

        pipeline = StreamingPipeline(self.settings, self.store, reset_checkpoint=True)
        stats = pipeline.run(max_batches=10, idle_exit_after=1.0)
        pipeline.context.finish("SUCCESS")
        self.store.record_run(pipeline.context)

        self.results["streaming"] = stats
        print(
            f"   processed {stats.events_read:,} events, "
            f"{stats.events_late:,} late, {stats.alerts:,} alerts"
        )
        return stats

    def run_sql(self) -> dict[str, Any]:
        from src.analytics.duckdb_client import AnalyticsEngine
        from src.core.config import project_root

        engine = AnalyticsEngine(self.settings)
        total = failed = 0
        for path in sorted((project_root() / "sql").glob("*.sql")):
            for result in engine.run_sql_file(path):
                total += 1
                failed += 0 if result.succeeded else 1
        engine.close()

        self.results["sql"] = {"queries": total, "failed": failed}
        print(f"   {total - failed}/{total} analytical queries succeeded")
        return self.results["sql"]

    # -- summary ------------------------------------------------------------

    def summary(self) -> None:
        from src.core.io import read_parquet_or_empty

        pipeline = self.results.get("pipeline")
        context = pipeline.context if pipeline else None

        anomalies = read_parquet_or_empty(
            self.settings.dataset_path("gold", "gold_anomaly_transactions")
        )
        alerts = read_parquet_or_empty(self.settings.dataset_path("gold", "gold_risk_alerts"))
        accounts = read_parquet_or_empty(self.settings.dataset_path("gold", "gold_account_risk"))
        vendors = read_parquet_or_empty(self.settings.dataset_path("gold", "gold_vendor_risk"))

        high = int((anomalies["risk_level"] == "HIGH").sum()) if not anomalies.empty else 0
        critical = int((anomalies["risk_level"] == "CRITICAL").sum()) if not anomalies.empty else 0
        ml_flagged = int(anomalies["is_anomaly"].fillna(False).sum()) if not anomalies.empty else 0

        quality = pipeline.quality.overall if pipeline and pipeline.quality else 0.0

        print("\n" + BANNER)
        print("FINANCIAL INTELLIGENCE PIPELINE")
        print(BANNER)

        if context:
            print(f"Transactions ingested: {context.records_read:>12,}")
            print(f"Silver records:        {context.records_written:>12,}")
            print(f"Quarantined:           {context.records_quarantined:>12,}")
            print()
        print(f"Anomalies (model):     {ml_flagged:>12,}")
        print(f"High risk:             {high:>12,}")
        print(f"Critical risk:         {critical:>12,}")
        print(f"Open alerts:           {len(alerts):>12,}")
        print()
        print(
            f"High-risk accounts:    "
            f"{int(accounts['risk_level'].isin(['HIGH', 'CRITICAL']).sum()) if not accounts.empty else 0:>12,}"
        )
        print(
            f"High-risk vendors:     "
            f"{int(vendors['risk_level'].isin(['HIGH', 'CRITICAL']).sum()) if not vendors.empty else 0:>12,}"
        )
        print()
        print(f"Data quality:          {quality:>11.2f}%")

        streaming = self.results.get("streaming")
        if streaming:
            print(f"Streaming events:      {streaming.events_read:>12,}")
            print(f"Realtime alerts:       {streaming.alerts:>12,}")

        graph = self.results.get("graph")
        if graph:
            print(
                f"Graph:                 "
                f"{'loaded ' + str(sum(graph.nodes.values())) + ' nodes' if graph.loaded else 'not loaded (fallback in use)':>12}"
            )

        total_duration = sum(step.duration_seconds for step in self.steps)
        print(f"Total duration:        {total_duration:>11.1f}s")
        print(f"Pipeline status:       {context.status if context else 'NOT RUN':>12}")
        print(BANNER)

        print("\nLayer row counts")
        print("-" * 60)
        for layer, tables in (
            ("bronze", ["bronze_transactions", "bronze_accounts", "bronze_vendors"]),
            ("silver", ["silver_transactions", "silver_accounts", "silver_vendors"]),
            (
                "gold",
                [
                    "gold_daily_financial_summary",
                    "gold_account_risk",
                    "gold_vendor_risk",
                    "gold_anomaly_transactions",
                    "fact_transactions",
                ],
            ),
        ):
            for table in tables:
                count = dataset_row_count(self.settings.dataset_path(layer, table))
                print(f"  {layer:<8}{table:<32}{count:>10,}")

        print("\nSteps")
        print("-" * 60)
        for step in self.steps:
            symbol = {"ok": "ok", "skipped": "--", "failed": "FAILED"}.get(step.status, "?")
            print(f"  [{symbol:^6}] {step.name:<38}{step.duration_seconds:>7.1f}s")
            if step.detail:
                print(f"           {step.detail}")

        print("\nNext steps")
        print("-" * 60)
        print("  streamlit run dashboard/app.py           # the dashboard")
        print("  uvicorn src.api.main:app --port 8000     # the serving API")
        print("  python -m src.analytics.run_sql          # the SQL library")
        print("  docker compose up -d postgres neo4j      # optional services")
        print()

    def exit_code(self) -> int:
        return 1 if any(step.status == "failed" for step in self.steps) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_demo.py",
        description="Run the complete Financial Intelligence Lakehouse demonstration.",
    )
    parser.add_argument("--scale", default="default", choices=("small", "default", "large"))
    parser.add_argument("--events", type=int, default=400, help="Streaming events to emit.")
    parser.add_argument("--keep", action="store_true", help="Keep the existing lake.")
    parser.add_argument("--samples", action="store_true", help="Rewrite data/sample/ files.")
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-sql", action="store_true")
    parser.add_argument(
        "--external",
        action="store_true",
        help="Call the live USAspending API (skipped by default to keep the demo offline).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    print(BANNER)
    print("FINANCIAL INTELLIGENCE LAKEHOUSE - DEMONSTRATION")
    print(BANNER)
    print(f"Scale:   {args.scale}")
    print(f"Engine:  {settings.engine}")
    print(f"Lake:    {settings.paths.bronze.parent}")
    print(f"Metadata backend: {settings.metadata_backend}")

    demo = Demo(settings, args)

    if not args.keep:
        demo.step("Clean previous run", demo.clean)
    demo.step("Generate multi-format source data", demo.generate)
    demo.step("Probe the REST source system", demo.check_source_api, optional=True)
    if args.external:
        demo.step("Fetch federal awards from USAspending", demo.fetch_external, optional=True)
    demo.step("Run the batch pipeline (Bronze -> Silver -> Gold)", demo.run_pipeline)
    if not args.skip_graph:
        demo.step("Load the investigation graph", demo.load_graph, optional=True)
    if not args.skip_streaming:
        demo.step("Simulate and process the event stream", demo.run_streaming, optional=True)
    if not args.skip_sql:
        demo.step("Execute the SQL analytics library", demo.run_sql, optional=True)

    demo.summary()
    return demo.exit_code()


if __name__ == "__main__":
    sys.exit(main())
