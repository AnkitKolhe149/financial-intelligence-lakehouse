# Databricks Deployment

**This project has not been deployed to Databricks.** It runs locally on Parquet
with a pandas engine, and on PySpark + Delta inside a container. This document
describes what would change, and why the local design makes those changes small.

Claiming a Databricks deployment that does not exist would be the easiest thing
in this repository to check and the fastest way to lose an interview.

## What already maps directly

| Local | Databricks | Change required |
| --- | --- | --- |
| `SparkEngine` session builder | Databricks runtime session | Delete the builder; the runtime provides a configured session |
| Delta writes via `spark.write.format("delta")` | Same | None |
| `Settings.dataset_path()` returning local paths | `abfss://` / `s3://` or a Unity Catalog name | One function |
| Declarative Silver specs | Same | None |
| Quality suites, risk rules, feature engineering | Same | None |
| `pipeline_runs` in PostgreSQL | Same, or system tables | Connection string |
| Monthly partitioning + `OPTIMIZE`/`ZORDER` | Same | None |

The transformations do not change at all. That is the point of keeping the
specification declarative and the engine pluggable.

## Unity Catalog design

```
financial_intelligence                       (catalog)
├── bronze                                   (schema)
│   ├── transactions            managed, partitioned by ingestion_date
│   ├── accounts, vendors, agencies, departments, contracts, budgets, locations
│   └── transaction_events
├── silver
│   ├── transactions            partitioned by transaction_month
│   ├── accounts, vendors, agencies, departments, contracts, budgets
│   ├── transaction_events
│   └── quarantine_transactions
├── gold
│   ├── daily_financial_summary
│   ├── account_risk, vendor_risk
│   ├── agency_spending, department_spending
│   ├── anomaly_transactions, risk_alerts, realtime_alerts
│   ├── transaction_velocity
│   ├── data_quality_summary, data_quality_trend, pipeline_runs
│   └── fact_transactions, dim_date, dim_account, dim_vendor, dim_agency,
│       dim_department, dim_location
└── volumes
    ├── landing                 raw source files
    └── models                  serialised model artifacts
```

The local catalogue (`dataset_catalog`) already records dataset, layer, owner,
classification, schema version, row count, size, partition columns and column
schema for every table — the same fields Unity Catalog holds. Migrating means
pointing at UC instead of maintaining the table.

### Layer semantics and grants

| Layer | Meaning | Who reads it |
| --- | --- | --- |
| bronze | Raw and immutable; the audit source | Data engineering only |
| silver | Validated and conformed | Engineering + analysts |
| gold | Business-ready | Analysts, BI, services |

```sql
GRANT USE CATALOG ON CATALOG financial_intelligence TO `data-analysts`;
GRANT USE SCHEMA, SELECT ON SCHEMA financial_intelligence.gold TO `data-analysts`;
GRANT SELECT ON SCHEMA financial_intelligence.silver TO `data-engineering`;
-- Bronze is not granted to analysts: raw data with known defects should not be
-- queried as if it were fact.
```

## Migration, step by step

### 1. Storage paths

The only code change that touches the pipeline:

```python
# src/core/config.py
def dataset_path(self, layer: str, dataset: str) -> str:
    if self.environment == "databricks":
        return f"{self.catalog}.{layer}.{dataset}"      # UC three-part name
    return str(self.paths_for(layer) / dataset)          # local
```

Readers and writers then use `spark.table(name)` / `saveAsTable(name)` instead
of path-based load and save.

### 2. Landing zone and ingestion

Local file discovery becomes Auto Loader, which is the same pattern the
streaming pipeline already implements — discover new files, process once, record
what was processed:

```python
(spark.readStream
   .format("cloudFiles")
   .option("cloudFiles.format", "csv")
   .option("cloudFiles.schemaLocation", f"{checkpoint}/schema")
   .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
   .load("/Volumes/financial_intelligence/bronze/landing/csv/")
   .writeStream
   .option("checkpointLocation", f"{checkpoint}/bronze_transactions")
   .trigger(availableNow=True)
   .toTable("financial_intelligence.bronze.transactions"))
```

`schemaEvolutionMode=addNewColumns` is Auto Loader's version of the schema union
the Bronze loader performs by hand. The custom content-hash ledger becomes
redundant for file sources — Auto Loader guarantees exactly-once — but stays
useful for the REST feeds, which have no file semantics.

### 3. Orchestration

Each pipeline stage becomes a task in one job:

```yaml
resources:
  jobs:
    financial_intelligence_batch:
      schedule: { quartz_cron_expression: "0 0 2 * * ?", timezone_id: UTC }
      tasks:
        - task_key: bronze
          python_wheel_task: { entry_point: bronze_load }
        - task_key: silver
          depends_on: [{ task_key: bronze }]
          python_wheel_task: { entry_point: silver_build }
        - task_key: quality
          depends_on: [{ task_key: silver }]
        - task_key: gold
          depends_on: [{ task_key: quality }]
        - task_key: graph
          depends_on: [{ task_key: gold }]
          # Optional stage: the job must not fail because the graph is down.
```

The stage boundaries already exist in `BatchPipeline._stage()`, so this is a
packaging change rather than a redesign.

### 4. Quality gates

The quality framework runs unchanged. Delta constraints would additionally
enforce the blocking rules at write time:

```sql
ALTER TABLE financial_intelligence.silver.transactions
  ADD CONSTRAINT positive_amount CHECK (amount > 0);
ALTER TABLE financial_intelligence.silver.transactions
  ADD CONSTRAINT known_currency CHECK (currency IN ('USD','EUR','GBP','CAD','JPY','AUD','CHF'));
```

Note the interaction: constraints *reject the write*, whereas this platform
*quarantines the record*. Both would be used — constraints as a backstop, the
quarantine as the recoverable path — because a failed write loses the evidence.

### 5. Lineage

Unity Catalog captures table- and column-level lineage automatically from Spark
execution. The custom `lineage_edges` table stays useful for what UC does not
record: which *transformations* were applied and how many records were dropped
between two datasets.

### 6. Model management

The local registry maps onto MLflow with no conceptual change — it already
records algorithm, hyperparameters, feature list, training window, evaluation
and artifact path:

```python
mlflow.sklearn.log_model(model.pipeline, "anomaly_model")
mlflow.log_params(report.hyperparameters)
mlflow.log_metrics({"pr_auc": ..., "precision_at_50": ...})
mlflow.register_model(uri, "financial_intelligence.ml.anomaly_model")
```

### 7. Serving

The FastAPI layer reads Gold tables. On Databricks it would read through a SQL
Warehouse instead of local Parquet — one connection change — or be replaced by
Databricks SQL and a dashboard, depending on who the consumers are.

## Cost and cluster shape

| Workload | Shape | Note |
| --- | --- | --- |
| Batch Bronze→Gold | Job cluster, autoscaling 2–8 workers | Terminates after the run; no idle cost |
| Streaming | Single small cluster, `availableNow` micro-batches | Continuous mode only if latency demands it |
| Ad-hoc SQL | Serverless SQL Warehouse | Auto-stop after 10 minutes |
| Model training | Single node with more memory | scikit-learn does not distribute |

The last row is worth stating: training runs on one node because Isolation
Forest is a single-machine algorithm. Sending it to a cluster would cost more and
run slower.

## What would need genuine work

Not everything is a configuration change.

1. **The pandas engine does not run on a cluster.** It stays as the local and
   test path; Spark becomes the only production engine. The parity test is what
   makes that switch safe.
2. **The DuckDB analytics layer is replaced** by Databricks SQL. The SQL library
   is standard enough to port, but window-function and date-function dialects
   differ and each file would need checking.
3. **Streaming state.** The in-process `VelocityTracker` is correct for a single
   process. On a cluster it becomes `flatMapGroupsWithState` or a stateful
   aggregation with a watermark.
4. **Secrets** move from `.env` to Databricks secret scopes.
5. **Neo4j** has no managed Databricks equivalent; it stays external, or the
   relationship queries fall back to the SQL implementations that already exist.

## Honest summary

The transformation logic, quality framework, risk rules, feature engineering and
metadata model are portable as written. The storage addressing, orchestration
and streaming state are the real work, and would take days rather than weeks.

What this project demonstrates is not a Databricks deployment. It is that the
same declarative specification runs on two engines and produces identical
output, which is the property that makes a third engine a manageable change
rather than a rewrite.
