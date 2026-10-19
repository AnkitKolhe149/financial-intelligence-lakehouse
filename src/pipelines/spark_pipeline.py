"""Bronze -> Silver -> Gold on PySpark + Delta.

    docker compose run --rm spark python -m src.pipelines.spark_pipeline
    docker compose run --rm spark python -m src.pipelines.spark_pipeline --datasets transaction

This is a genuine second implementation of the same pipeline, driven by the same
declarative specifications in :mod:`src.silver.spec` and the same
``config/settings.yaml``. It is not a wrapper around the pandas code.

The Spark-idiomatic choices, and why:

**Window functions instead of grouped Python.** Deduplication uses
``row_number() OVER (PARTITION BY key ORDER BY ...)``, which is the same
semantics as the pandas engine's "sort then take the first per group" but
executes as a single distributed shuffle.

**Broadcast joins for referential integrity.** Dimensions are small; broadcasting
them turns each foreign-key check from a shuffle join into a map-side lookup.
The hint is explicit rather than left to the optimiser, because the optimiser
only knows sizes it has statistics for.

**One pass, many outputs.** Validation computes a single ``error_code`` column
with a chained ``when`` expression, then splits valid from rejected. Computing
each rule as a separate filter would re-scan the data once per rule.

**Cache exactly once, deliberately.** The validated frame is used by both the
Silver write and the Gold aggregations, so it is cached - and unpersisted when
finished. Caching everything is a common way to make a Spark job slower.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING, Any

from src.core.config import Settings, get_settings
from src.core.contracts import ALLOWED_CURRENCIES, USD_REFERENCE_RATES, ErrorCode
from src.core.logging import configure_logging, get_logger
from src.core.run_context import RunContext
from src.engines.spark_engine import SparkEngine
from src.metadata.lineage import LineageCollector
from src.metadata.store import get_metadata_store
from src.silver.spec import SILVER_BUILD_ORDER, SilverSpec, get_silver_spec

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

log = get_logger(__name__)

PIPELINE_NAME = "financial_intelligence_spark"


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------


def normalise_strings(frame: DataFrame, spec: SilverSpec) -> DataFrame:
    """Trim, collapse whitespace, apply case rules and null-out null-ish tokens."""
    from pyspark.sql import functions as F

    rules = spec.normalisation
    null_tokens = [token.upper() for token in rules.null_tokens]

    for field in frame.schema.fields:
        if field.name.startswith("_") or field.dataType.typeName() != "string":
            continue
        column = F.trim(F.regexp_replace(F.col(field.name), r"\s+", " "))
        column = F.when((column == "") | F.upper(column).isin(null_tokens), F.lit(None)).otherwise(
            column
        )
        frame = frame.withColumn(field.name, column)

    for name in rules.upper:
        if name in frame.columns:
            frame = frame.withColumn(name, F.upper(F.col(name)))
    for name in rules.lower:
        if name in frame.columns:
            frame = frame.withColumn(name, F.lower(F.col(name)))
    for name in rules.title:
        if name in frame.columns:
            frame = frame.withColumn(name, F.initcap(F.col(name)))
    return frame


def cast_types(frame: DataFrame, spec: SilverSpec) -> DataFrame:
    """Cast to the canonical types declared in the schema registry.

    ``DecimalType(18, 2)`` for money, not double: floating point is the wrong
    representation for currency, and the difference shows up as cent-level drift
    once you sum a million rows.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, DateType, DecimalType, IntegerType, TimestampType

    mapping = {
        "int": IntegerType(),
        "long": IntegerType(),
        "float": DecimalType(18, 4),
        "decimal": DecimalType(18, 2),
        "bool": BooleanType(),
        "date": DateType(),
        "timestamp": TimestampType(),
    }

    for column, dtype in spec.schema.dtypes.items():
        if column not in frame.columns:
            continue
        if dtype == "timestamp":
            # to_timestamp with no format handles ISO-8601; invalid values become
            # null and are caught by validation rather than raising here.
            frame = frame.withColumn(column, F.to_timestamp(F.col(column)))
        elif dtype in mapping:
            frame = frame.withColumn(column, F.col(column).cast(mapping[dtype]))
        else:
            frame = frame.withColumn(column, F.col(column).cast("string"))
    return frame


def add_transaction_derivations(frame: DataFrame) -> DataFrame:
    """USD conversion, partition date and fiscal year."""
    from pyspark.sql import functions as F

    rate = F.lit(None).cast("decimal(18,6)")
    for code, value in USD_REFERENCE_RATES.items():
        rate = F.when(
            F.col("currency") == code, F.lit(float(value)).cast("decimal(18,6)")
        ).otherwise(rate)

    frame = frame.withColumn("_usd_rate", rate)
    frame = frame.withColumn(
        "amount_usd", F.round(F.col("amount").cast("decimal(18,2)") * F.col("_usd_rate"), 2)
    )
    frame = frame.withColumn("transaction_date", F.to_date(F.col("transaction_timestamp")))
    frame = frame.withColumn(
        "fiscal_year",
        F.when(
            F.month("transaction_timestamp") >= 10, F.year("transaction_timestamp") + 1
        ).otherwise(F.year("transaction_timestamp")),
    )
    if "posted_timestamp" in frame.columns:
        frame = frame.withColumn(
            "_is_late_arriving",
            F.coalesce(
                (F.unix_timestamp("posted_timestamp") - F.unix_timestamp("transaction_timestamp"))
                > 86_400,
                F.lit(False),
            ),
        )
    else:
        frame = frame.withColumn("_is_late_arriving", F.lit(False))
    return frame.drop("_usd_rate")


def validate(
    frame: DataFrame, spec: SilverSpec, settings: Settings, engine: SparkEngine
) -> DataFrame:
    """Attach ``error_code`` / ``error_message`` in a single pass.

    A chained ``when`` gives first-failure-wins semantics identical to the pandas
    engine, and computes every rule in one scan instead of one scan per rule.
    """
    from pyspark.sql import functions as F

    error_code = F.lit(None).cast("string")
    error_message = F.lit(None).cast("string")

    def rule(condition: Any, code: str, message: str) -> None:
        nonlocal error_code, error_message
        error_code = F.when(error_code.isNull() & condition, F.lit(code)).otherwise(error_code)
        error_message = F.when(error_message.isNull() & condition, F.lit(message)).otherwise(
            error_message
        )

    # 1. required fields
    for column in spec.effective_required():
        if column in frame.columns:
            rule(
                F.col(column).isNull(),
                ErrorCode.NULL_REQUIRED_FIELD,
                f"Required field '{column}' is null",
            )

    # 2. timestamps
    for index, column in enumerate(spec.timestamp_columns):
        if column not in frame.columns:
            continue
        if index == 0 or column in spec.effective_required():
            rule(
                (
                    F.col(column).isNull() & F.col(f"{column}_raw").isNotNull()
                    if f"{column}_raw" in frame.columns
                    else F.lit(False)
                ),
                ErrorCode.INVALID_TIMESTAMP,
                f"Field '{column}' could not be parsed",
            )
        rule(
            F.col(column) > F.expr("current_timestamp() + INTERVAL 5 MINUTES"),
            ErrorCode.FUTURE_TIMESTAMP,
            f"Field '{column}' is dated in the future",
        )

    # 3. amounts
    for column in spec.positive_columns:
        if column not in frame.columns:
            continue
        rule(
            F.col(column) <= 0,
            ErrorCode.INVALID_AMOUNT,
            f"Field '{column}' must be greater than zero",
        )
        rule(
            F.col(column) > F.lit(settings.quality.max_transaction_amount),
            ErrorCode.AMOUNT_EXCEEDS_LIMIT,
            f"Field '{column}' exceeds the platform limit",
        )

    # 4. currency
    if "currency" in frame.columns:
        allowed = sorted(ALLOWED_CURRENCIES & set(settings.quality.allowed_currencies))
        rule(
            F.col("currency").isNotNull() & ~F.col("currency").isin(allowed),
            ErrorCode.INVALID_CURRENCY,
            "Currency is not in the approved set",
        )

    frame = frame.withColumn("error_code", error_code).withColumn("error_message", error_message)

    # 5. referential integrity, via broadcast joins
    for foreign_key in spec.foreign_keys:
        if foreign_key.column not in frame.columns:
            continue
        reference_table = next(
            (t for t, key in spec.reference_tables.items() if key == foreign_key.column), None
        )
        if reference_table is None or not engine.exists("silver", reference_table):
            log.info("spark_referential_check_skipped", column=foreign_key.column)
            continue

        reference = (
            engine.read("silver", reference_table, columns=[foreign_key.column])
            .distinct()
            .withColumnRenamed(foreign_key.column, "_ref_key")
        )
        # Broadcast: the dimensions are small, so this becomes a map-side lookup
        # rather than a full shuffle of the fact table.
        frame = frame.join(
            F.broadcast(reference),
            frame[foreign_key.column] == reference["_ref_key"],
            how="left",
        )
        orphan = F.col(foreign_key.column).isNotNull() & F.col("_ref_key").isNull()

        if foreign_key.blocking:
            frame = frame.withColumn(
                "error_code",
                F.when(
                    F.col("error_code").isNull() & orphan, F.lit(foreign_key.error_code)
                ).otherwise(F.col("error_code")),
            ).withColumn(
                "error_message",
                F.when(
                    F.col("error_message").isNull() & orphan,
                    F.lit(
                        f"'{foreign_key.column}' does not exist in {foreign_key.references_dataset}"
                    ),
                ).otherwise(F.col("error_message")),
            )
        else:
            # Non-blocking: null the broken reference and keep the record.
            frame = frame.withColumn(
                foreign_key.column,
                F.when(orphan, F.lit(None)).otherwise(F.col(foreign_key.column)),
            )
        frame = frame.drop("_ref_key")

    return frame


def deduplicate(frame: DataFrame, spec: SilverSpec) -> tuple[DataFrame, DataFrame]:
    """Window-function deduplication: survivors and losers.

    ``row_number`` over the business key gives exactly the pandas engine's
    "sort, keep first" semantics, and the second pass over ``_record_hash``
    catches re-issued records that carry a different identifier.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    order_columns = [c for c in spec.dedup.order_by if c in frame.columns]
    tie_breakers = [c for c in spec.dedup.keys if c in frame.columns]
    ordering = [
        F.col(c).desc_nulls_last() if not spec.dedup.ascending else F.col(c).asc_nulls_last()
        for c in order_columns
    ] + [F.col(c).asc_nulls_last() for c in tie_breakers]

    keys = [c for c in spec.dedup.keys if c in frame.columns]
    if not keys:
        return frame, frame.limit(0)

    window = Window.partitionBy(*keys).orderBy(*ordering)
    ranked = frame.withColumn("_row_number", F.row_number().over(window))
    ranked = ranked.withColumn("_duplicate_count", F.count("*").over(Window.partitionBy(*keys)))

    survivors = ranked.filter(F.col("_row_number") == 1).drop("_row_number")
    duplicates = (
        ranked.filter(F.col("_row_number") > 1)
        .drop("_row_number")
        .withColumn("error_code", F.lit(ErrorCode.DUPLICATE_RECORD))
        .withColumn("error_message", F.lit(f"Duplicate {'+'.join(keys)}"))
    )

    if spec.dedup.include_content_hash and "_record_hash" in survivors.columns:
        hash_window = Window.partitionBy("_record_hash").orderBy(*ordering)
        ranked_hash = survivors.withColumn("_hash_rank", F.row_number().over(hash_window))
        content_duplicates = (
            ranked_hash.filter(F.col("_hash_rank") > 1)
            .drop("_hash_rank")
            .withColumn("error_code", F.lit(ErrorCode.DUPLICATE_RECORD))
            .withColumn("error_message", F.lit("Duplicate content under a different identifier"))
        )
        survivors = ranked_hash.filter(F.col("_hash_rank") == 1).drop("_hash_rank")
        duplicates = duplicates.unionByName(content_duplicates, allowMissingColumns=True)

    return survivors, duplicates


def build_silver_spark(
    dataset: str,
    engine: SparkEngine,
    context: RunContext,
    settings: Settings,
    lineage: LineageCollector | None = None,
) -> dict[str, Any]:
    """Build one Silver table with Spark."""
    from pyspark.sql import functions as F

    spec = get_silver_spec(dataset)
    started = time.perf_counter()

    if not engine.exists("bronze", spec.source_table):
        log.warning("spark_bronze_missing", table=spec.source_table)
        return {"dataset": dataset, "records_in": 0, "records_out": 0, "quarantined": 0}

    bronze = engine.read("bronze", spec.source_table)
    records_in = bronze.count()

    frame = normalise_strings(bronze, spec)
    frame = cast_types(frame, spec)
    if dataset == "transaction":
        frame = add_transaction_derivations(frame)

    frame = validate(frame, spec, settings, engine)

    valid = frame.filter(F.col("error_code").isNull()).drop("error_code", "error_message")
    rejected = frame.filter(F.col("error_code").isNotNull())

    survivors, duplicates = deduplicate(valid, spec)

    # Silver audit columns, matching the pandas engine exactly.
    survivors = survivors.withColumn("_silver_run_id", F.lit(context.run_id)).withColumn(
        "_processed_timestamp", F.current_timestamp()
    )
    if "_is_late_arriving" not in survivors.columns:
        survivors = survivors.withColumn("_is_late_arriving", F.lit(False))

    output_columns = [
        c
        for c in (
            spec.schema.column_names
            + list(spec.derived_columns)
            + [
                "_source_system",
                "_record_hash",
                "_ingestion_run_id",
                "_silver_run_id",
                "_processed_timestamp",
                "_schema_version",
                "_is_late_arriving",
                "_duplicate_count",
            ]
        )
        if c in survivors.columns
    ]
    # dict.fromkeys de-duplicates while preserving order: _is_late_arriving is
    # both a derived column and an audit column.
    survivors = survivors.select(*list(dict.fromkeys(output_columns)))

    # Cache: this frame is written to Silver and then aggregated for Gold. One
    # cache, one unpersist - not a blanket "cache everything".
    survivors.cache()
    records_out = survivors.count()

    partitions = [spec.partition_column] if spec.partition_column in survivors.columns else None
    engine.write(survivors, "silver", spec.target_table, mode="overwrite", partition_by=partitions)

    quarantined = 0
    if settings.quality.quarantine_invalid_records:
        all_rejected = rejected.unionByName(duplicates, allowMissingColumns=True)
        quarantined = _write_quarantine_spark(all_rejected, dataset, context, engine)

    duration = time.perf_counter() - started
    if lineage is not None:
        lineage.add(
            spec.source_table,
            spec.target_table,
            spec.transformation_names(),
            source_layer="bronze",
            target_layer="silver",
            records_in=records_in,
            records_out=records_out,
        )

    log.info(
        "spark_silver_built",
        table=spec.target_table,
        records_in=records_in,
        records_out=records_out,
        quarantined=quarantined,
        duration_seconds=round(duration, 2),
    )
    survivors.unpersist()
    return {
        "dataset": dataset,
        "records_in": records_in,
        "records_out": records_out,
        "quarantined": quarantined,
        "duration_seconds": duration,
    }


def _write_quarantine_spark(
    rejected: DataFrame, dataset: str, context: RunContext, engine: SparkEngine
) -> int:
    """Write rejects in the same shape the pandas engine produces."""
    from pyspark.sql import functions as F

    count = rejected.count()
    if count == 0:
        return 0

    payload_columns = [c for c in rejected.columns if c not in {"error_code", "error_message"}]
    quarantine = (
        rejected.withColumn("record_id", F.expr("uuid()"))
        .withColumn("dataset", F.lit(dataset))
        .withColumn("raw_record", F.to_json(F.struct(*[F.col(c) for c in payload_columns])))
        .withColumn("detected_at", F.current_timestamp())
        .withColumn("pipeline_run_id", F.lit(context.run_id))
        .withColumn("quarantine_date", F.current_date().cast("string"))
        .withColumn("error_field", F.lit(None).cast("string"))
        .select(
            "record_id",
            "dataset",
            F.col("_source_system").alias("source_system"),
            F.col("_source_file").alias("source_file"),
            F.col("_source_format").alias("source_format"),
            "raw_record",
            "error_code",
            "error_message",
            "error_field",
            "detected_at",
            "pipeline_run_id",
            F.col("_ingestion_run_id").alias("ingestion_run_id"),
            "quarantine_date",
        )
    )
    path = engine.settings.paths.quarantine / f"silver_quarantine_{dataset}s"
    quarantine.write.format(engine.storage_format).mode("append").partitionBy(
        "quarantine_date"
    ).save(path.as_posix())
    return count


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------


def build_gold_spark(
    engine: SparkEngine, context: RunContext, settings: Settings
) -> dict[str, int]:
    """Build the aggregate Gold datasets with Spark."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    if not engine.exists("silver", "silver_transactions"):
        return {}

    transactions = engine.read("silver", "silver_transactions")
    transactions.cache()
    results: dict[str, int] = {}

    # --- daily financial summary -------------------------------------------
    daily = transactions.groupBy("transaction_date").agg(
        F.count("*").alias("transaction_count"),
        F.round(F.sum("amount_usd"), 2).alias("total_amount_usd"),
        F.round(F.avg("amount_usd"), 2).alias("average_transaction_usd"),
        F.round(F.expr("percentile_approx(amount_usd, 0.5)"), 2).alias("median_transaction_usd"),
        F.round(F.max("amount_usd"), 2).alias("max_transaction_usd"),
        F.countDistinct("account_id").alias("unique_accounts"),
        F.countDistinct("vendor_id").alias("unique_vendors"),
        F.countDistinct("agency_id").alias("unique_agencies"),
    )
    # Rolling baseline via a window - the Spark equivalent of pandas .rolling().
    baseline = Window.orderBy("transaction_date").rowsBetween(-6, 0)
    daily = daily.withColumn(
        "rolling_7d_avg_volume", F.round(F.avg("transaction_count").over(baseline), 1)
    ).withColumn("rolling_7d_avg_amount_usd", F.round(F.avg("total_amount_usd").over(baseline), 2))
    results["gold_daily_financial_summary"] = engine.write(
        daily, "gold", "gold_daily_financial_summary", mode="overwrite"
    )

    # --- account profile ----------------------------------------------------
    accounts = transactions.groupBy("account_id").agg(
        F.count("*").alias("transaction_count"),
        F.round(F.sum("amount_usd"), 2).alias("total_amount_usd"),
        F.round(F.avg("amount_usd"), 2).alias("average_amount_usd"),
        F.round(F.max("amount_usd"), 2).alias("maximum_amount_usd"),
        F.round(F.stddev("amount_usd"), 2).alias("stddev_amount_usd"),
        F.countDistinct("vendor_id").alias("unique_vendors"),
        F.countDistinct("location_id").alias("unique_locations"),
        F.min("transaction_timestamp").alias("first_transaction_at"),
        F.max("transaction_timestamp").alias("last_transaction_at"),
        F.countDistinct("transaction_date").alias("active_days"),
    )
    results["gold_account_risk"] = engine.write(
        accounts, "gold", "gold_account_risk", mode="overwrite"
    )

    # --- vendor profile -----------------------------------------------------
    vendors = (
        transactions.groupBy("vendor_id")
        .agg(
            F.count("*").alias("transaction_count"),
            F.round(F.sum("amount_usd"), 2).alias("total_amount_usd"),
            F.round(F.avg("amount_usd"), 2).alias("average_amount_usd"),
            F.countDistinct("account_id").alias("unique_accounts"),
            F.countDistinct("agency_id").alias("unique_agencies"),
            F.sum(F.when(F.col("contract_id").isNotNull(), 1).otherwise(0)).alias(
                "contracted_payments"
            ),
            F.min("transaction_timestamp").alias("first_payment_at"),
            F.max("transaction_timestamp").alias("last_payment_at"),
        )
        .withColumn(
            "uncontracted_payment_pct",
            F.round(
                100.0
                * (F.col("transaction_count") - F.col("contracted_payments"))
                / F.col("transaction_count"),
                2,
            ),
        )
    )
    results["gold_vendor_risk"] = engine.write(
        vendors, "gold", "gold_vendor_risk", mode="overwrite"
    )

    # --- agency spending ----------------------------------------------------
    agencies = (
        transactions.filter(F.col("agency_id").isNotNull())
        .groupBy("agency_id", "fiscal_year")
        .agg(
            F.count("*").alias("transaction_count"),
            F.round(F.sum("amount_usd"), 2).alias("total_amount_usd"),
            F.countDistinct("vendor_id").alias("unique_vendors"),
            F.countDistinct("department_id").alias("unique_departments"),
        )
    )
    results["gold_agency_spending"] = engine.write(
        agencies, "gold", "gold_agency_spending", mode="overwrite"
    )

    transactions.unpersist()
    log.info("spark_gold_built", **results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_spark_pipeline(
    settings: Settings | None = None,
    datasets: tuple[str, ...] = SILVER_BUILD_ORDER,
    *,
    skip_gold: bool = False,
    optimise: bool = True,
) -> dict[str, Any]:
    """Execute the Spark pipeline and record metadata like the batch pipeline."""
    settings = settings or get_settings()
    store = get_metadata_store()

    context = RunContext(
        pipeline_name=PIPELINE_NAME,
        pipeline_version=settings.pipeline.version,
        source="bronze",
        engine="spark",
        config_version=settings.pipeline.version,
    )
    lineage = LineageCollector(context.run_id, PIPELINE_NAME, settings.pipeline.version)
    store.record_run(context)

    engine = SparkEngine(settings)
    outcomes: dict[str, Any] = {"silver": {}, "gold": {}}

    try:
        for dataset in datasets:
            result = build_silver_spark(dataset, engine, context, settings, lineage)
            outcomes["silver"][dataset] = result
            context.count(
                read=result.get("records_in", 0),
                written=result.get("records_out", 0),
                quarantined=result.get("quarantined", 0),
            )
            context.add_stage(
                f"silver_{dataset}",
                result.get("duration_seconds", 0.0),
                result.get("records_in", 0),
                result.get("records_out", 0),
            )

        if not skip_gold:
            outcomes["gold"] = build_gold_spark(engine, context, settings)

        if optimise and engine.storage_format == "delta":
            # Compact the small files that partitioned writes create, and
            # Z-ORDER the fact table on the columns queries filter by.
            engine.optimise("silver", "silver_transactions", zorder_by=["account_id", "vendor_id"])

        lineage.persist(store)
        lineage.export_json(settings)
        context.finish("SUCCESS")

    except Exception as exc:
        context.fail(exc)
        log.error("spark_pipeline_failed", error=str(exc)[:400])
        raise
    finally:
        store.record_run(context)
        engine.stop()

    outcomes["run_id"] = context.run_id
    outcomes["status"] = context.status
    outcomes["duration_seconds"] = context.duration_seconds
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipelines.spark_pipeline",
        description="Run Bronze -> Silver -> Gold on PySpark + Delta.",
    )
    parser.add_argument("--datasets", nargs="*", default=list(SILVER_BUILD_ORDER))
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--no-optimise", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    outcomes = run_spark_pipeline(
        settings,
        datasets=tuple(args.datasets),
        skip_gold=args.skip_gold,
        optimise=not args.no_optimise,
    )

    print("\n" + "=" * 56)
    print("SPARK PIPELINE")
    print("=" * 56)
    print(f"Run ID:   {outcomes['run_id']}")
    print(f"Status:   {outcomes['status']}")
    print(f"Duration: {outcomes['duration_seconds']:.1f}s")
    print("\nSilver:")
    for dataset, result in outcomes["silver"].items():
        print(
            f"  {dataset:<20} in={result.get('records_in', 0):>8,} "
            f"out={result.get('records_out', 0):>8,} "
            f"quarantined={result.get('quarantined', 0):>6,}"
        )
    if outcomes["gold"]:
        print("\nGold:")
        for table, count in outcomes["gold"].items():
            print(f"  {table:<34} {count:>8,}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
