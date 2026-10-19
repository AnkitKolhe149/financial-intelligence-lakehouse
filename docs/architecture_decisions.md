# Architecture Decisions

Each entry states the decision, the reasoning, what was rejected, and what it
would take to revisit. Several were changed during the build because
measurement contradicted the original choice; those are marked **revised** and
the original reasoning is kept, because the fact that it was wrong is the useful
part.

---

## ADR-001: Bronze / Silver / Gold rather than a single transformation

**Decision.** Three physical layers, each with a different contract.

**Why.** The layers are not decoration; each answers a different question.
Bronze answers *what did the source send?* and must stay true even after the
Silver logic is rewritten. Silver answers *what are we prepared to stand
behind?* Gold answers *what does the business want to know?*

The practical payoff is reprocessing. When a validation rule turns out to be
wrong - and one did, during this build, when unparseable timestamps were being
reported as missing fields - the fix is to delete Silver and Gold and re-run.
No source system is re-queried, no data is lost, and the corrected output is
reproducible. A single transformation from source to answers cannot do that: the
raw evidence is gone the moment the job succeeds.

**Rejected.** One ELT step into a warehouse. Faster to write, and it makes every
future bug a data-recovery exercise.

---

## ADR-002: pandas as the default engine, Spark as the production path

**Decision.** The pipeline ships with two engine implementations. pandas is the
default; PySpark + Delta runs in a container.

**Why.** At the volumes this project runs locally - 50,000 transactions, roughly
20 MB of Parquet - pandas is not a compromise, it is the faster choice. Spark's
JVM startup, scheduling and shuffle machinery cost seconds on work that pandas
finishes in milliseconds. Reaching for a cluster at this size is a common and
expensive instinct.

Spark earns its overhead when data exceeds one machine's memory or when work
genuinely parallelises. That is the production condition, and it is what maps
onto Databricks - so the Spark implementation is real, not sketched.

**How they stay honest.** Both engines interpret the *same* declarative
specifications (`src/silver/spec.py`, the quality suites, the risk rules in
`config/settings.yaml`). They are two interpreters, not shared code with a flag.
`tests/test_engine_parity.py` runs both over identical Bronze input and asserts
identical row counts, identical surviving transaction ids and identical USD
totals. Without that test the claim would be marketing.

**Rejected.** A dataframe abstraction layer wrapping both. That produces
something that is neither pandas nor Spark, is slower than both, and hides
exactly the engine-specific optimisations that justify using Spark.

---

## ADR-003: Content hashing for deduplication and idempotency

**Decision.** Every record carries a SHA-256 fingerprint over its *business*
columns. The transaction id is deliberately excluded.

**Why.** Two different duplicates exist, and only one is visible to a business
key. A replayed file sends `TXN-000123` twice - the key catches it. A source
system re-issuing the same payment under a new reference does not - the key sees
two distinct records. The content hash catches both, because it is computed over
account, vendor, amount, currency and timestamp.

The same hash makes ingestion idempotent: a ledger of `(source_file,
content_hash)` means re-running a load is a no-op, while a file re-sent with new
content under the same name is correctly treated as new data.

**Detail that mattered.** Normalisation has to be canonical or the hash is
useless. `" ACME "` and `"acme"` must agree; `100` and `100.0` must agree - a
test caught the latter, where the CSV feed's string `"100"` and the JSON feed's
number `100.0` were hashing differently and would have survived deduplication as
two records.

**Rejected.** Row-order or ingestion-time keys. Neither survives a re-run.

---

## ADR-004: Quarantine instead of dropping bad records

**Decision.** Invalid records are written to a quarantine dataset with the raw
payload, an error code, the source file and the run that rejected them.

**Why.** Three reasons, in increasing order of importance. *Recoverability*: when
the approved-currency list turns out to be missing a legitimate code, the
affected records are replayed from quarantine rather than re-requested from a
source that may no longer have them. *Diagnosis*: "which upstream system
produces the most invalid records" is only answerable if you kept them.
*Trust*: a pipeline that reports 48,464 of 55,550 records processed **and can
show you the other 7,086** is credible. One that reports 48,464 is not.

**Rejected.** Dropping with a log line. The count is in the logs and the
evidence is gone.

---

## ADR-005: Isolation Forest, unsupervised, alongside deterministic rules

**Decision.** Both an unsupervised model and a rule engine, with their
disagreement reported.

**Why unsupervised.** In a real oversight environment labelled fraud is scarce,
arrives months late, and covers only the fraud that was *caught*. A supervised
model trained on it learns to find what is already being found. Isolation Forest
needs no labels and flags statistically unusual activity, which is an honest
description of what the platform can actually do.

**Why also rules.** The model cannot explain itself. "Anomaly score 0.94" is not
something to put in front of an oversight committee. Rules encode what the
business already knows, are auditable line by line, and produce evidence
("Amount is 12.4x the account's prior average of $1,204").

**Why report the disagreement.** Where they agree, confidence is high. The
model-only population is where genuinely novel patterns appear - or where the
model has drifted. Both are worth knowing, and neither is visible if you only
look at the union.

**Constraint.** The ML rule contributes the largest single weight (30 points)
but cannot reach CRITICAL alone. An unexplainable score should not escalate a
case by itself.

---

## ADR-006: Labels held out of the feature path entirely

**Decision.** Ground-truth fraud labels live in a separate file
(`data/raw/labels/`), are never features, and are joined only for evaluation.

**Why.** Leakage in fraud modelling produces excellent offline metrics and a
useless production model. Holding labels physically apart makes accidental use
difficult rather than merely discouraged, and `assert_no_leakage` fails loudly
if a label or a model output ever appears in the feature list.

It also models reality: investigation outcomes arrive *after* the transactions,
which is why they are generated as a separate dataset with their own dates.

---

## ADR-007: Stationary features only — **revised**

**Original.** Feed the model the obvious history features: cumulative
transaction counts per account and per vendor.

**What happened.** Held-out performance was poor (ROC-AUC 0.61) and the flagged
rate drifted to four times the configured contamination. The cause was not the
model. Cumulative counts grow monotonically with calendar time, so a model
trained on the first 70% of the timeline sees systematically smaller values than
it later scores against. Every record in the test period looked mildly unusual
for the trivial reason that it was *later*.

**Revised decision.** Express history as **ratios and bounded time windows**:
"how does this amount compare to this account's own prior average", "how many
transactions in the last 24 hours", "what fraction of this vendor's payers are
distinct". Absolute history survives only as `log_account_history`,
log-compressed so 10 versus 20 prior transactions matters and 900 versus 1000
does not.

**Result.** ROC-AUC 0.61 → 0.72, PR-AUC 0.024 → 0.041, false positives 230 →
179 on the same held-out period. `tests/test_anomaly_detection.py` now asserts
stationarity directly, so the mistake cannot come back silently.

---

## ADR-008: Monthly partitions, not daily — **revised**

**Original.** Partition the transaction fact by `transaction_date`. Daily
partitioning is the obvious choice for a time-series fact table.

**What happened.** `scripts/benchmark.py --compare-optimisations` reported that
a metadata-only row count was **49% slower** than reading the whole table -
which is backwards. The cause: 9,675 rows spread across 573 daily partitions,
roughly 17 rows per file. Per-file overhead dominated everything.

**Revised decision.** Partition by `transaction_month`. Partition count fell
573 → 20, and a full-table read went from 467 ms to 27 ms - a 17x improvement
from a configuration change, with no code path altered. Both measured
optimisations then came out positive: pushdown 57.7% faster, footer counts 18.8%
faster at the default scale.

**The rule, stated properly.** Size the partition, not the calendar. At
production volumes - millions of rows per day - daily is right again. The
mistake was choosing granularity by instinct instead of by arithmetic.

---

## ADR-009: PostgreSQL for metadata, DuckDB for analytics, Parquet for data

**Decision.** Three stores, three jobs, no overlap.

| Store | Holds | Access pattern |
| --- | --- | --- |
| PostgreSQL (SQLite fallback) | runs, quality results, lineage, ledger, model registry | small rows, constant writes, transactional reads |
| Parquet lake | Bronze / Silver / Gold / quarantine | large, append-mostly, analytical scans |
| DuckDB | SQL over the lake, in place | columnar, vectorised, read-only |

**Why not one.** Running analytical scans against an OLTP database, or writing
pipeline heartbeats into a columnar store, both work badly for predictable
reasons. DuckDB reads the Parquet files directly - there is no load step and no
second copy, so an analytical query cannot be looking at stale data.

**SQLite fallback.** The demo must run before anyone starts a container. Same
schema, same queries, same code path - it goes through SQLAlchemy Core either
way.

---

## ADR-010: Neo4j as enrichment, never a dependency

**Decision.** Every graph entry point degrades to an in-process NetworkX
implementation over the same Parquet data, and the pipeline treats graph loading
as an optional stage.

**Why.** A payments oversight platform must not stop ingesting money movements
because a graph database is down. The graph earns its place on variable-depth
traversal - "everything within three hops of this vendor" is one line of Cypher
and a recursive CTE nobody enjoys maintaining - but that value does not justify
making it load-bearing.

`sql/10_graph_investigation.sql` implements the fixed-depth relationship
questions in plain SQL, precisely so the comparison is visible rather than
asserted. The rule this project follows: fixed-depth joins stay in SQL,
variable-depth traversal and path-finding go to the graph.

---

## ADR-011: File-based streaming, Kafka optional

**Decision.** The streaming source is an NDJSON landing zone processed in
micro-batches.

**Why.** It is a real pattern - cloud object-store autoloading works exactly
this way - and it exercises every concern that matters: exactly-once file
handling via a content-hash checkpoint, event time versus processing time,
watermarked late arrivals, bounded velocity state, and the small-file problem.

Kafka would add a broker, a schema registry and a consumer-group story to a
project whose subject is the *processing*, not the transport.
[`kafka_optional.md`](kafka_optional.md) shows exactly which class changes.

**Late data.** Marked, never dropped. The watermark bounds how long state stays
open, not what the platform accepts. A payment that arrives late is still a
payment, and a system that silently discards it has lost data.

---

## ADR-012: Streaming scores against a feature store — **revised**

**Original.** Compute streaming features from the event alone, defaulting
unknown history to zero.

**What happened.** The `new_vendor` rule fired on 300 of 300 events. Zero
history means "fewer than five prior transactions" is true for everything, so
every event produced an alert and the output carried no signal at all.

**Revised decision.** Streaming reads per-entity history from the Gold risk
tables - a feature-store read, refreshed by the batch pipeline. Where an entity
is genuinely unknown, the neutral default is chosen so the dependent rule does
**not** fire: an unknown vendor is given a history *above* the threshold.

**Why that direction.** "We have no data about this vendor" must never be
reported to an investigator as "this vendor is suspicious".

**Result.** `new_vendor` 300 → 0 false positives, the amount-versus-average rule
started working (0 → 13 genuine hits), alerts fell 59 → 35 with three real
criticals.

---

## ADR-013: Synthetic data as the primary source

**Decision.** Generate the source landscape; treat USAspending and Kaggle as
optional additions.

**Why.** No real dataset ships as CSV *and* JSON *and* XML *and* Excel *and* a
REST feed *and* a stream, so multi-format normalisation could not be
demonstrated on one. More importantly, a data-quality framework can only be
*tested* against data whose defects are known: the generator injects them at
configured rates, so expected quarantine volumes are known in advance.

And nothing here is sensitive. Every person, vendor and account is fictional.

**USAspending is still real and still used.** Four documented endpoints, live,
no authentication, with a synthetic fallback of identical shape so an offline
laptop produces the same pipeline behaviour.

---

## ADR-014: Fail soft by default

**Decision.** `pipeline.fail_fast` is false. A broken source, a failed Gold
dataset or an unavailable metadata store degrades the run rather than aborting
it, and the run is marked `SUCCESS_WITH_WARNINGS`.

**Why.** An oversight platform that refuses to publish yesterday's spending
because 0.6% of records were malformed has failed at its job more completely
than one that publishes with a documented quality score. The failure is
recorded, the affected records are in quarantine, and the score reflects it.

**Where it does not apply.** Blocking quality checks can stop a run when the
caller opts in, and `--fail-fast` exists for CI. Record-level problems are never
fatal; source-level and stage-level problems are configurable.

---

## ADR-015: Everything generated is generated from one definition

**Decision.** The canonical schema registry (`src/core/schema.py`) is the single
source for Silver casting, quality suites, the published JSON contracts, the
data dictionary and referential-integrity rules.

**Why.** A hand-maintained description of code is wrong within a month. Adding a
column here propagates to validation, documentation and the contract in one
edit. `docs/data_dictionary.md` and `schemas/*.json` carry a generated-file
banner so nobody edits the output and wonders why it reverts.
