# Interview Guide

Everything here is verifiable in the repository. If a claim cannot be backed by
a file, a test or a measured number, it is not in this document.

---

## 60-second explanation

> It is a financial oversight data platform. It ingests spending data from six
> different source shapes — CSV, JSON, XML, Excel, a REST API and an event
> stream — and lands it in a Bronze/Silver/Gold lakehouse.
>
> Bronze keeps exactly what the source sent, including the bad records. Silver
> validates, types and deduplicates, and every record it rejects goes to a
> quarantine dataset with an error code rather than being dropped. Gold answers
> the oversight questions: which accounts and vendors are risky, where the money
> goes, which transactions warrant investigation.
>
> Detection runs two ways deliberately — an Isolation Forest for patterns nobody
> wrote down, and a deterministic rule engine so every score can be explained to
> a committee. A transaction scoring 65 carries the list of rules that fired and
> the evidence for each.
>
> It runs end to end in about a minute with no containers, and the same
> transformation specification also executes on PySpark and Delta, with a parity
> test asserting both engines produce identical output.

---

## Five-minute architecture walkthrough

**Sources.** Six shapes, each with the quirks real sources have — XML with the
key in an attribute, Excel with a title row above the header, a REST API with
envelope pagination and a silently-enforced page cap. Plus USAspending.gov,
which is real: four documented endpoints, no auth, called live.

**Ingestion.** Every reader returns the same result type regardless of format.
Readers deliberately do *not* clean — they map field names, attach provenance
and stop. Everything is read as text so that `"not-a-date"` survives to be
reported as unparseable rather than silently coerced to null.

**Bronze.** Raw, append-only, schema-on-read. Every row carries source system,
file, format, ingestion run, content hash and schema version. Idempotent via a
content-hash ledger: re-running a load is a no-op, but a file re-sent with new
content under the same name is correctly treated as new data.

**Silver.** Where the platform takes a position. Required field missing →
quarantine. Optional field missing → keep the null. Broken *blocking* foreign
key → quarantine. Broken *non-blocking* key (contract) → null the reference and
keep the payment, because petty cash legitimately has no contract. Duplicates
collapse on both the business key and a content hash, which catches a payment
re-issued under a new reference.

**Quality.** Separate from validation, because they answer different questions.
Validation asks "may this record proceed?"; quality asks "is this dataset
getting worse?" Six weighted dimensions, scored per run, published into the lake
so an analyst can join spending to trustworthiness in one query.

**Detection.** Leakage-safe features — every historical value computed from
strictly prior rows — then Isolation Forest plus the rule engine, with their
disagreement reported.

**Serving.** FastAPI, a seven-page Streamlit dashboard, 53 DuckDB queries over
the Parquet in place, and an optional Neo4j graph that falls back to an
in-process implementation when it is not running.

---

## Technical questions

### 1. Why Bronze / Silver / Gold rather than one transformation?

Reprocessing. When a validation rule turns out to be wrong — and one did during
this build, where unparseable timestamps were being reported as missing fields —
the fix is to delete Silver and Gold and re-run. No source is re-queried,
nothing is lost. With a single source-to-answers transformation, the raw
evidence is gone the moment the job succeeds, and every future bug becomes a
data-recovery exercise.

### 2. Why preserve raw data that you know is wrong?

Because "wrong" is a judgement the platform makes, and judgements change. A
negative amount preserved in Bronze is evidence that an upstream system has a
bug; deleted at ingestion, it is a number in a log file. It also enables replay:
when the approved-currency list turns out to be missing a legitimate code, those
records are recovered from quarantine rather than re-requested from a source
that may no longer have them.

### 3. How do you handle schema evolution?

Additively, without migration. Bronze is schema-on-read; the loader unions
schemas across files and back-fills nulls; Silver promotes a column once it is
declared in the registry. The transaction feed ships in two versions
specifically to prove it — `transactions_2025.csv` has 16 columns,
`transactions_2026.csv` has 17 with `merchant_category` — and both live in one
Silver table. Breaking changes are *not* automatic and should not be: they need
a new version, a mapping and a decision about history.

### 4. What happens to a bad record?

It goes to quarantine with the raw payload as received, an error code, the field
that failed, the source file and the run that rejected it. Nothing is deleted.

The ordering matters: unparseable values are checked *before* missing ones,
because casting turns `"not-a-date"` into null and the two then look identical.
Reporting an unparseable timestamp as `NULL_REQUIRED_FIELD` is technically true
of the cast value and actively misleading to whoever has to fix the source. That
was a real bug, caught by a test.

### 5. How do you prevent duplicates?

Two different problems. Same id re-sent — the business key catches it. Different
id, same content — a source re-issuing a payment under a new reference — which
only a content hash catches, computed over account, vendor, amount, currency and
timestamp, deliberately excluding the id.

Normalisation has to be canonical or the hash is useless. A test caught `100`
and `100.0` hashing differently, which would have let the CSV and JSON feeds'
versions of the same payment both survive.

The survivor is chosen by a total ordering, so two runs over identical input
keep the same row.

### 6. How do you monitor the pipeline?

Every run writes a `pipeline_runs` row — recorded *before* work starts, so a
process killed mid-run leaves a `RUNNING` row rather than no trace. Per-stage
timings, record counters, config and model versions, host and Python version.
Failures are recorded too; a pipeline that only records successes is blind to
the runs worth investigating.

Structured logs carry `run_id` and `stage` on every line. `/metrics` exposes
operational counters, and the dashboard has a monitoring page showing throughput
in records-per-second — because duration alone should grow with volume, and only
throughput separates "more data" from a regression.

### 7. How would this scale from 50k to 50M records?

Below ~5M rows, the pandas engine as-is; at this size it is genuinely faster than
Spark, because Spark's scheduling overhead dominates. Beyond that, the Spark
engine — same declarative specification, so no transformation changes.

Then: partition granularity (measured, not guessed — see the next answer),
incremental Silver by ingestion date rather than full rebuild, Delta `OPTIMIZE`
with `ZORDER` on the join keys, broadcast joins for the dimensions, and
separating ingestion compute from serving compute.

### 8. Tell me about a performance problem you found.

The benchmark reported that a metadata-only row count was 49% *slower* than
reading the whole table, which is backwards, so it was worth chasing.

The cause was arithmetic: 9,675 rows across 573 daily partitions is about 17
rows per file, and opening 573 files costs more than reading their contents.
Classic small-file problem, caused by partitioning finer than the data justifies.

Switching to monthly partitions: 573 → 20 directories, a full-table read from
467 ms to 27 ms — 17x — and the test suite got 47% faster as a side effect. No
transformation logic changed.

The lesson is to size the partition, not the calendar. At millions of rows per
day, daily is correct again.

### 9. How would you deploy this to Databricks, and where does Unity Catalog fit?

The transformations port unchanged, because they are declarative specs the
engine interprets. What changes: `dataset_path()` returns a three-part catalog
name instead of a filesystem path; file discovery becomes Auto Loader with
`schemaEvolutionMode=addNewColumns`, which is the schema union the Bronze loader
does by hand; stages become job tasks; the local registry becomes MLflow.

Unity Catalog replaces the `dataset_catalog` table, which already records
dataset, layer, owner, classification, schema version, row count, size and
partition columns — the same fields. Bronze is not granted to analysts: raw data
with known defects should not be queried as if it were fact.

What is genuine work: the DuckDB analytics layer becomes Databricks SQL, and the
in-process velocity tracker becomes `flatMapGroupsWithState`.

**I have not deployed this to Databricks**, and `docs/databricks_deployment.md`
says so in its first line.

### 10. How is streaming implemented, and why not Kafka?

A file landing zone processed in micro-batches — the same pattern as cloud
object-store autoloading. Exactly-once via a content-hash checkpoint, event time
versus processing time, watermarking, bounded per-account velocity state.

Kafka would add a broker, a registry and a consumer-group story to a project
whose subject is the processing. `docs/kafka_optional.md` shows the diff: one
class, at the discovery-and-commit boundary. Everything downstream is
transport-agnostic, which is what makes the file version not a shortcut.

### 11. How do you handle late-arriving events?

Marked and kept, never dropped. The watermark bounds how long state stays open,
not what the platform accepts — a payment that arrives late is still a payment,
and discarding it loses money from the totals. Silver flags
`_is_late_arriving` when posting lags occurrence by more than 24 hours, and the
streaming pipeline records observed lateness per event so the watermark can be
set from evidence rather than a guess.

### 12. How do you ensure lineage?

Emitted by the code that performs each transformation, not reconstructed by
parsing SQL. Each edge records the transformations applied and the row counts on
both sides, so "where did 7,086 records go?" is answerable — which a parser
cannot tell you. Stored per run in `lineage_edges`, exported to
`metadata/lineage.json`, served by `/lineage/{dataset}`, and traversable
transitively.

### 13. How does anomaly detection work, and why unsupervised?

Because labelled fraud is scarce, late, and only covers fraud that was already
caught — a supervised model trained on it learns to find what is already being
found. Isolation Forest isolates points by random partitioning; anomalies need
fewer splits.

Scores are normalised to `[0,1]` because the raw output is a signed unbounded
quantity nobody outside the model can interpret. Contamination maps directly to
how many alerts a review team can absorb, which is a business decision made
explicit in configuration.

Measured on the held-out period at default scale — 7,270 transactions, 53
labelled frauds, 0.73% base rate: PR-AUC 0.276, ROC-AUC 0.900, precision@10 of
0.70, lift@10 of 96x. Seven of the ten highest-scored transactions are genuine
fraud.

I would add two caveats unprompted. ROC-AUC flatters imbalanced data because the
7,185 true negatives dominate the false-positive rate, so PR-AUC is the number
to quote. And at the small profile the same code scores PR-AUC 0.041 — nothing
changed but the number of positives in the window, from 53 to 17. Metrics
computed on a handful of positives are noise, which is why both are documented.

Accuracy is never reported: predicting "never fraud" would score 99.3%.

### 14. How do you prevent leakage?

Three ways, all enforced in code. Every historical feature uses strictly prior
rows — the first transaction of an account sees zero history, asserted by a test.
Splits are chronological by timestamp, never random. Labels live in a separate
file, are never features, and `assert_no_leakage()` fails if a label or model
output appears in the feature list.

There is a subtler one. The original feature set used cumulative counts, which
grow with calendar time, so the test period looked systematically unusual for
being *later*. Fixing it — ratios and bounded windows instead of running totals
— took ROC-AUC from 0.61 to 0.72. A test now asserts stationarity so it cannot
come back.

### 15. What is the most interesting bug you fixed?

Time-window features that were never windowing. Epoch seconds were computed as
`.astype("int64") // 10**9`, which assumes nanosecond storage; under pandas 3
the column was microsecond-resolution, so the window bounds collapsed and
"transactions in the trailing 5 minutes" silently became "every prior
transaction". An ordinary account appeared to have made 26 transactions in five
minutes.

It ran, produced plausible numbers, and was wrong — which is the worst kind. A
unit test with four hand-written timestamps caught it. It is now computed
resolution-independently.

---

## Questions worth asking back

- How much labelled fraud do you actually have, and how long after the event
  does it arrive? That determines whether supervised modelling is viable.
- What is the review team's daily capacity? That is what contamination and the
  alert threshold should be derived from, not a configured constant.
- Where does data quality currently get discovered — in the pipeline, or by an
  analyst noticing a number looks wrong?
- Is lineage needed for compliance, for incident response, or both? They imply
  different levels of granularity.

---

## Known limitations (state these before being asked)

1. **Spark, PostgreSQL and Neo4j paths are code-complete but not executed on the
   build machine** — Docker's engine could not be started (its service needs
   elevation). The images build in CI and the tests exist, marked and skipped.
2. **ML metrics depend heavily on sample size.** PR-AUC 0.276 at default scale,
   0.041 at small scale, on identical code — the difference is 53 versus 17
   labelled frauds in the test window. The rules carry the explainable load;
   the model adds coverage.
3. **Fixed FX rates.** A production system joins a rates dimension with effective
   dating; this is documented as a simplification in the data dictionary.
4. **The pandas engine does not distribute.** It is the local and test path;
   Spark is the production one.
5. **Entity resolution is reported, not solved.** Vendors sharing a tax id are
   surfaced for a steward, never auto-merged.
6. **Alert lifecycle is modelled but not driven.** Alerts have statuses; nothing
   updates them, because there is no case-management system to update from.
