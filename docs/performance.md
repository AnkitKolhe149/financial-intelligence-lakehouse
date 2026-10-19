# Performance

Every number here was measured by `scripts/benchmark.py` on the machine
described below. Nothing is estimated, and the raw output is written to
`metadata/benchmarks.json` so any figure can be traced back to the run that
produced it.

```bash
make benchmark
# or
python scripts/benchmark.py --scales small default --compare-optimisations --memory
```

## Measurement environment

| Property | Value |
| --- | --- |
| Python | 3.14.5 |
| Platform | Windows 11 (26200) |
| CPU count | 12 |
| pandas | 3.0.5 |
| Engine | pandas (single node) |
| Storage | Parquet + snappy |

These are laptop numbers on a single node. They are useful for comparing
*configurations against each other* on identical data, which is what the
optimisation section does. They are not a claim about cluster throughput.

## End-to-end, by scale

| Scale | Transactions generated | Bronze rows | Silver rows | Total | Lake size |
| --- | --- | --- | --- | --- | --- |
| small | 10,100 | 13,854 | 12,419 | 27.5 s | 5.5 MB |
| default | 50,500 | 63,891 | 56,805 | 65.9 s | 20.6 MB |

Bronze exceeds the generated count because three feeds overlap on purpose (two
CSV schema versions plus a REST slice), which is what gives cross-source
deduplication something real to do.

### Stage breakdown at the default scale

| Stage | Seconds | Records in | Records/sec |
| --- | --- | --- | --- |
| Generation | 23.30 | 50,500 | 2,167 |
| Ingestion + Bronze | 10.23 | 63,891 | 6,244 |
| Silver (validate, dedup, quarantine) | 13.42 | 63,891 | 4,759 |
| Data quality (7 datasets, 68 checks) | 0.91 | 56,805 | 62,543 |
| ML + risk + Gold (18 datasets) | 18.04 | 56,805 | 3,149 |
| **Total** | **65.91** | | |

Peak Python allocation: **65.7 MB** at the default scale — the whole pipeline
fits comfortably in memory, which is exactly why Spark would be the wrong tool
at this size.

### Throughput improves with scale

Silver throughput rises from 3,461 to 4,759 records/sec between the two scales.
Fixed costs — settings parsing, schema construction, opening the metadata store,
building check suites — are amortised over more rows. The corollary is that the
small profile *understates* per-record efficiency, so quoting its numbers as the
platform's throughput would be flattering in the wrong direction.

## Optimisations, measured with and without

The only honest way to claim an optimisation helped is to measure both ways on
the same data and the same machine. `--compare-optimisations` does exactly that,
averaged over five runs each.

Measured at the default scale (48,464 Silver transactions):

| Optimisation | Naive | Optimised | Change |
| --- | --- | --- | --- |
| Predicate pushdown + column pruning | 0.0368 s | 0.0156 s | **57.7% faster** |
| Row count from Parquet footers | 0.0292 s | 0.0237 s | **18.8% faster** |

*Pushdown* reads one monthly partition and two columns instead of the whole
table, via `pyarrow.dataset` filters that reach the Parquet reader.

*Footer counts* read row-group metadata rather than data pages —
`dataset_row_count()` returns 48,464 without decoding a single value.

## The partitioning mistake, and what it cost

This is the most useful measurement in the project, because the first result was
the *opposite* of what it should have been.

The original design partitioned `silver_transactions` by `transaction_date`,
which is the instinctive choice for a time-series fact table. The benchmark then
reported that a metadata-only row count was **48.9% slower** than reading the
entire table — which is backwards, and therefore worth chasing.

The cause was arithmetic, not code: 9,675 rows spread over **573 daily
partitions** is about 17 rows per file. Opening 573 files costs more than
reading the data in them. This is the small-file problem, produced by
partitioning finer than the data justifies.

Changing the partition key to `transaction_month`:

| Metric | Daily partitions | Monthly partitions | Change |
| --- | --- | --- | --- |
| Partition directories | 573 | 20 | 96% fewer |
| Full-table read | 0.4667 s | 0.0266 s | **17.5x faster** |
| Footer row count | slower than a full read | 18.8% faster | sign corrected |
| Test suite runtime | 162 s | 86 s | 47% faster |

No transformation logic changed. The entire gain came from one configuration
value, and it also halved the test suite's runtime as a side effect.

**The rule.** Size the partition, not the calendar. A partition should hold
enough data to be worth opening. At production volumes — millions of rows per
day — daily partitioning becomes correct again, and the configuration key
(`storage.partition_columns`) is where that decision lives.

## What else is optimised, and why

**Vectorised validation.** All record-level rules are column operations, not
per-row `apply`. Validating 55,550 transactions against nine rule families takes
about 1.5 seconds; a Pydantic model per record would take minutes. The Pydantic
contracts remain the enforcement path for REST and streaming sources, where
records genuinely arrive one at a time.

**O(n) history features.** Prior-window statistics use cumulative sums and
`searchsorted` rather than `expanding().apply()`, so building 17 features over
56,805 rows is a few seconds rather than minutes.

**Column pruning on reference reads.** Referential-integrity checks read one
column of a dimension, never the dimension.

**Single Silver read per Gold run.** Ten Gold builders share one scored frame
rather than re-reading Silver ten times.

**Deliberate, minimal caching.** In the Spark path exactly one frame is cached —
the validated Silver output, which is used by both the write and the
aggregations — and it is unpersisted afterwards. Caching everything is a common
way to make a Spark job slower.

## Spark-side tuning

The Spark engine sets these explicitly (`src/engines/spark_engine.py`):

| Setting | Value | Reason |
| --- | --- | --- |
| `spark.sql.adaptive.enabled` | true | Re-plans using real partition statistics |
| `spark.sql.adaptive.skewJoin.enabled` | true | A few vendors hold most rows |
| `spark.sql.shuffle.partitions` | 16 | The 200 default produces tiny tasks at this size |
| `spark.sql.autoBroadcastJoinThreshold` | 32 MB | Dimensions are small; broadcast beats shuffle |
| `spark.sql.sources.partitionOverwriteMode` | dynamic | Re-running one month rewrites one month |
| `spark.sql.execution.arrow.pyspark.enabled` | true | Fast collect at the ML boundary |

Delta `OPTIMIZE` with `ZORDER BY (account_id, vendor_id)` runs after the Silver
write to compact the small files that partitioned writes create.

**Not yet measured on this machine.** The Spark path could not be executed here —
the host runs Python 3.14 with no JVM, and Docker's engine could not be started
(see [`deployment.md`](deployment.md)). The settings above are applied in code
and the image builds in CI, but no Spark timing in this document is claimed,
because none was taken.

## Scaling beyond one machine

| Volume | Approach |
| --- | --- |
| < 5M rows | pandas engine as-is |
| 5M – 100M | Spark engine locally or on one large node; monthly partitions still fine |
| 100M – 10B | Spark on a cluster, daily partitions, Delta with `OPTIMIZE` and `ZORDER`, incremental Silver by ingestion date |
| Beyond | Partition by tenant or agency as well as time; separate compute for ingestion and serving |

The transformations do not change shape at any step — they are the same
declarative specifications — which is the point of the engine split.
