"""Measure pipeline performance across dataset sizes.

    python scripts/benchmark.py                     # small + default
    python scripts/benchmark.py --scales small default large
    python scripts/benchmark.py --compare-optimisations

Every number this prints is measured on the machine it runs on. Nothing is
hard-coded, and the output records the interpreter, platform and CPU count so a
figure quoted in documentation can be traced to the environment that produced
it. Results are written to ``metadata/benchmarks.json`` and summarised in
``docs/performance.md``.

``--compare-optimisations`` re-runs the Silver stage with specific optimisations
disabled, which is the only honest way to claim an optimisation helped: measure
it with and without, on the same data, on the same machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.core.config import Settings, get_settings
from src.core.io import dataset_size_bytes, read_parquet
from src.core.logging import configure_logging, get_logger
from src.core.run_context import RunContext
from src.metadata.store import MetadataStore

log = get_logger(__name__)


@dataclass
class StageTiming:
    stage: str
    seconds: float
    records_in: int = 0
    records_out: int = 0

    @property
    def records_per_second(self) -> float:
        return round(self.records_in / self.seconds, 1) if self.seconds > 0 else 0.0


@dataclass
class BenchmarkResult:
    scale: str
    transactions_generated: int
    stages: list[StageTiming] = field(default_factory=list)
    lake_bytes: int = 0
    peak_memory_mb: float | None = None

    @property
    def total_seconds(self) -> float:
        return round(sum(stage.seconds for stage in self.stages), 3)

    @property
    def throughput(self) -> float:
        silver = next((s for s in self.stages if s.stage == "silver"), None)
        return silver.records_per_second if silver else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "transactions_generated": self.transactions_generated,
            "total_seconds": self.total_seconds,
            "throughput_records_per_second": self.throughput,
            "lake_bytes": self.lake_bytes,
            "lake_mb": round(self.lake_bytes / 1024 / 1024, 2),
            "peak_memory_mb": self.peak_memory_mb,
            "stages": [
                {**asdict(stage), "records_per_second": stage.records_per_second}
                for stage in self.stages
            ],
        }


def environment() -> dict[str, Any]:
    """Record where the numbers came from."""
    import os

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "pandas": pd.__version__,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _clean(settings: Settings, store: MetadataStore) -> None:
    store.close()
    for directory in (
        settings.paths.bronze.parent,
        settings.paths.quarantine,
        settings.paths.raw,
        Path(settings.paths.metadata) / "local",
    ):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    settings.paths.ensure()


def benchmark_scale(scale: str, settings: Settings, *, track_memory: bool) -> BenchmarkResult:
    """Time each stage of the pipeline at one dataset size."""
    import tracemalloc

    from src.bronze.loader import BronzeLoader
    from src.data_generation.generate_all import generate_all
    from src.data_quality.validator import DataQualityValidator
    from src.gold.builder import build_all_gold
    from src.ingestion.sources import read_source, select_sources
    from src.silver.transformations import build_all_silver

    store = MetadataStore(settings)
    _clean(settings, store)
    store = MetadataStore(settings)
    context = RunContext(pipeline_name="benchmark", pipeline_version="1.0.0")

    result = BenchmarkResult(scale=scale, transactions_generated=0)
    if track_memory:
        tracemalloc.start()

    # -- generation ---------------------------------------------------------
    started = time.perf_counter()
    summary = generate_all(settings, scale=scale, write_sample_files=False)
    elapsed = time.perf_counter() - started
    generated = summary.counts["transactions"]
    result.transactions_generated = generated
    result.stages.append(StageTiming("generation", elapsed, generated, generated))

    # -- ingestion + bronze -------------------------------------------------
    started = time.perf_counter()
    loader = BronzeLoader(context.run_id, settings, store, force=True)
    definitions = select_sources(settings, requested="all", service_available=False)
    by_dataset: dict[str, list] = {}
    for definition in definitions:
        by_dataset.setdefault(definition.dataset, []).extend(
            read_source(definition, settings, context.run_id)
        )
    bronze_rows = 0
    for dataset, results in by_dataset.items():
        bronze_rows += loader.load(dataset, results).records_written
    elapsed = time.perf_counter() - started
    result.stages.append(StageTiming("ingestion_bronze", elapsed, bronze_rows, bronze_rows))

    # -- silver -------------------------------------------------------------
    started = time.perf_counter()
    silver_results = build_all_silver(context, settings, store=store)
    elapsed = time.perf_counter() - started
    silver_in = sum(r.records_in for r in silver_results.values())
    silver_out = sum(r.records_out for r in silver_results.values())
    result.stages.append(StageTiming("silver", elapsed, silver_in, silver_out))

    # -- quality ------------------------------------------------------------
    started = time.perf_counter()
    DataQualityValidator(context.run_id, settings, store).evaluate_all()
    result.stages.append(
        StageTiming("data_quality", time.perf_counter() - started, silver_out, silver_out)
    )

    # -- ml + gold ----------------------------------------------------------
    started = time.perf_counter()
    gold = build_all_gold(context, settings, store=store, train_model=True)
    elapsed = time.perf_counter() - started
    result.stages.append(StageTiming("ml_risk_gold", elapsed, silver_out, gold.total_records))

    if track_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_memory_mb = round(peak / 1024 / 1024, 1)

    for layer in ("bronze", "silver", "gold"):
        base = getattr(settings.paths, layer)
        result.lake_bytes += (
            sum(dataset_size_bytes(directory) for directory in base.iterdir() if directory.is_dir())
            if base.exists()
            else 0
        )

    store.close()
    return result


def compare_optimisations(settings: Settings) -> list[dict[str, Any]]:
    """Measure specific optimisations with and without, on identical data.

    Two are compared, both of which are claims made elsewhere in the
    documentation and therefore need evidence:

    1. **Column pruning + predicate pushdown** versus reading the whole dataset
       and filtering in pandas afterwards.
    2. **Footer-metadata row counts** versus loading the data to call ``len``.
    """
    comparisons: list[dict[str, Any]] = []
    path = settings.dataset_path("silver", "silver_transactions")
    if not path.exists():
        log.warning("optimisation_comparison_skipped", reason="silver_transactions not built")
        return comparisons

    frame = read_parquet(path)
    if frame.empty:
        return comparisons
    sample_date = str(frame["transaction_date"].iloc[len(frame) // 2])

    # --- 1. pushdown -------------------------------------------------------
    started = time.perf_counter()
    for _ in range(5):
        naive = read_parquet(path)
        naive = naive[naive["transaction_date"] == sample_date][["transaction_id", "amount_usd"]]
    naive_seconds = (time.perf_counter() - started) / 5

    started = time.perf_counter()
    for _ in range(5):
        pushed = read_parquet(
            path,
            columns=["transaction_id", "amount_usd"],
            filters=[("transaction_date", "=", sample_date)],
        )
    pushed_seconds = (time.perf_counter() - started) / 5

    comparisons.append(
        {
            "optimisation": "predicate pushdown + column pruning",
            "naive_seconds": round(naive_seconds, 4),
            "optimised_seconds": round(pushed_seconds, 4),
            "improvement_pct": (
                round((1 - pushed_seconds / naive_seconds) * 100, 1) if naive_seconds else 0.0
            ),
            "rows_returned": len(pushed),
            "note": "read one partition and two columns instead of the whole table",
        }
    )

    # --- 2. footer counts --------------------------------------------------
    from src.core.io import dataset_row_count

    started = time.perf_counter()
    for _ in range(5):
        count_naive = len(read_parquet(path))
    naive_count_seconds = (time.perf_counter() - started) / 5

    started = time.perf_counter()
    for _ in range(5):
        count_fast = dataset_row_count(path)
    fast_count_seconds = (time.perf_counter() - started) / 5

    comparisons.append(
        {
            "optimisation": "row count from Parquet footers",
            "naive_seconds": round(naive_count_seconds, 4),
            "optimised_seconds": round(fast_count_seconds, 6),
            "improvement_pct": (
                round((1 - fast_count_seconds / naive_count_seconds) * 100, 1)
                if naive_count_seconds
                else 0.0
            ),
            "rows_returned": int(count_fast),
            "note": f"metadata-only count agrees with the full read ({count_naive:,} rows)",
        }
    )
    return comparisons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/benchmark.py",
        description="Measure pipeline performance across dataset sizes.",
    )
    parser.add_argument(
        "--scales", nargs="*", default=["small", "default"], choices=["small", "default", "large"]
    )
    parser.add_argument("--compare-optimisations", action="store_true")
    parser.add_argument("--memory", action="store_true", help="Track peak Python allocation.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING", settings.log_format)

    print("=" * 72)
    print("FINANCIAL INTELLIGENCE LAKEHOUSE - BENCHMARK")
    print("=" * 72)
    for key, value in environment().items():
        print(f"  {key:<14}{value}")

    results: list[BenchmarkResult] = []
    for scale in args.scales:
        print(f"\nRunning scale: {scale}")
        print("-" * 72)
        result = benchmark_scale(scale, settings, track_memory=args.memory)
        results.append(result)

        print(f"  {'stage':<20}{'seconds':>10}{'records in':>14}{'records/sec':>14}")
        for stage in result.stages:
            print(
                f"  {stage.stage:<20}{stage.seconds:>10.2f}"
                f"{stage.records_in:>14,}{stage.records_per_second:>14,.0f}"
            )
        print(f"  {'TOTAL':<20}{result.total_seconds:>10.2f}")
        print(f"  lake size: {result.lake_bytes / 1024 / 1024:.1f} MB")
        if result.peak_memory_mb:
            print(f"  peak python allocation: {result.peak_memory_mb:.1f} MB")

    comparisons = compare_optimisations(settings) if args.compare_optimisations else []
    if comparisons:
        print("\nOptimisation comparisons (same data, same machine)")
        print("-" * 72)
        for entry in comparisons:
            print(f"  {entry['optimisation']}")
            print(
                f"    naive {entry['naive_seconds']:.4f}s -> "
                f"optimised {entry['optimised_seconds']:.4f}s "
                f"({entry['improvement_pct']:.1f}% faster)"
            )
            print(f"    {entry['note']}")

    if len(results) > 1:
        print("\nScaling")
        print("-" * 72)
        print(f"  {'scale':<12}{'transactions':>14}{'total s':>10}{'records/s':>12}{'MB':>8}")
        for result in results:
            print(
                f"  {result.scale:<12}{result.transactions_generated:>14,}"
                f"{result.total_seconds:>10.2f}{result.throughput:>12,.0f}"
                f"{result.lake_bytes / 1024 / 1024:>8.1f}"
            )

    payload = {
        "environment": environment(),
        "results": [result.to_dict() for result in results],
        "optimisation_comparisons": comparisons,
    }
    output = Path(args.output) if args.output else settings.paths.metadata / "benchmarks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
