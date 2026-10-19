# ML Methodology

## The problem, stated honestly

Detect financial transactions worth investigating, in an environment where
labelled fraud is scarce, arrives late, and covers only the fraud that was
already caught.

That last clause is the whole reason this is an unsupervised problem. A
supervised model trained on historical investigation outcomes learns to find
what the existing controls already find. Isolation Forest needs no labels and
flags statistically unusual activity — which is an honest description of what a
platform can actually do on day one.

## Model

**Isolation Forest**, 200 trees, contamination 0.01, `RobustScaler(5–95)`.

| Choice | Reason |
| --- | --- |
| Isolation Forest | No labels required; isolates points by random partitioning, so anomalies land in shallow leaves |
| Robust scaling | Transaction amounts are lognormal; mean/std scaling lets a handful of very large payments dominate every feature |
| Contamination 0.01 | Maps directly to "how many alerts per day can the review team absorb" — a business decision, made explicit in configuration |
| Normalised score | `score_samples` returns a signed, unbounded quantity nobody outside the model can interpret; output is mapped to `[0, 1]` where 1 is most anomalous |

The decision threshold is calibrated on the **training** distribution and reused
at scoring time. A per-batch percentile would make a transaction's
classification depend on which batch it happened to arrive in.

## Temporal integrity

Transaction data is a time series. Two rules are enforced in code, not by
convention.

### 1. Every historical feature uses strictly prior rows

```
account_avg_amount over the whole dataset   ->  leakage
account_avg_amount over strictly prior rows ->  correct
```

The first version lets a transaction influence the baseline it is compared
against. A large fraudulent payment raises the account average, which shrinks
its own deviation and hides it. Offline metrics improve; the model gets worse.

Implemented with cumulative arithmetic (O(n)) rather than
`expanding().apply()` (O(n·k)), so it stays fast at 500,000 rows.
`tests/test_anomaly_detection.py` asserts that the first transaction of every
account sees exactly zero history and that prior counts never exceed the rows
before them.

### 2. Splits are chronological, never random

```
|<--------- train 70% --------->|<-- val 15% -->|<-- test 15% -->|
                              time ---->
```

Cut points are *timestamps*, not row positions, so all transactions from the
same instant land on the same side of a boundary. A random split would train on
Wednesday and test on Tuesday, which no production model gets to do.

### 3. Labels are physically separate

Ground truth lives in `data/raw/labels/fraud_labels.csv`, is never a feature,
and is joined only for evaluation. `assert_no_leakage()` fails loudly if a label
or a model output ever appears in the feature list.

## Features — and why they are all stationary

17 features, listed in `src/ml/features.py`. Every one is approximately
stationary over calendar time, and that constraint was learned the hard way.

An earlier version fed the model raw cumulative counts (`account_txn_count_prior`,
`vendor_txn_count_prior`). Those grow monotonically with time, so a model trained
on the first 70% of the timeline sees systematically smaller values than it later
scores against. Every record in the test period looked mildly unusual for the
trivial reason that it was *later*; the flagged rate drifted to four times the
configured contamination and genuine anomalies were buried in the drift.

The fix was to express history as **ratios and bounded windows**:

| Group | Features |
| --- | --- |
| Amount | `amount_usd`, `log_amount` |
| Calendar | `hour_of_day`, `day_of_week`, `is_weekend`, `is_night` |
| Relative amount | `amount_to_account_avg`, `amount_zscore_account` |
| Bounded windows | `account_txn_count_24h`, `account_velocity_5min`, `vendor_txn_count_7d`, `log_seconds_since_last_txn` |
| Ratios | `account_vendor_diversity`, `vendor_account_concentration` |
| Binary | `is_new_vendor_for_account`, `is_region_mismatch` |
| Compressed maturity | `log_account_history` |

"What fraction of this account's payments went to distinct vendors" means the
same thing in January and in December. "How many vendors has it ever paid" only
ever increases.

`tests/test_anomaly_detection.py::test_features_are_stationary` compares
early-period and late-period means and fails if any feature drifts beyond a
tolerance — so the mistake cannot return silently.

### A bug this surfaced

While writing the stationarity test, the time-window features turned out never
to have been windowing at all. Epoch seconds were computed as
`.astype("int64") // 10**9`, which assumes nanosecond storage; under pandas 3 the
column was microsecond-resolution, so the divisor was wrong, the window bounds
collapsed, and "transactions in the trailing 5 minutes" silently became "every
prior transaction". That is why an ordinary account appeared to have made 26
transactions in five minutes.

It is now computed resolution-independently by subtracting the epoch and asking
for seconds. This is the single most valuable thing the test suite found: the
feature ran, produced plausible numbers, and was wrong.

## Warm-up rows are excluded from training

An account's first few transactions have no history, so their relative features
take neutral defaults. Those rows cluster at the start of the timeline —
exactly the period the chronological split assigns to training. Fitting on them
teaches the model that "neutral" is normal, and every mature account then scores
as unusual for having a history.

They are still **scored**: a new account's first payment is a legitimate thing to
assess, and the rule engine handles it explicitly. Only what the model learns
"normal" from changes.

## Evaluation

Fraud is roughly 0.4% of the generated population. **Accuracy is never
reported**: predicting "never fraud" would score 99.6% and detect nothing. The
metric note in every evaluation payload says so explicitly.

| Metric | Why |
| --- | --- |
| precision@k | Matches the operating reality — a team reviews k cases a day, and this is their hit rate |
| recall@k | What share of known fraud reaches that queue |
| PR-AUC (average precision) | The right summary for imbalanced problems |
| Lift@k | How many times better than random selection |
| ROC-AUC | Reported for comparability, with the caveat that it flatters imbalanced data |
| Confusion matrix | At the operating threshold |
| Rule agreement | Overlap between model and rules — disagreement is the interesting part |

### Measured results

**Default scale** — held-out test period of 7,270 transactions containing 53
labelled frauds, a 0.73% base rate:

| Metric | Value | Reading |
| --- | --- | --- |
| PR-AUC | **0.276** | 38x the 0.0073 a random ranker would achieve |
| ROC-AUC | **0.900** | Strong separation, with the imbalance caveat below |
| Precision@10 | **0.70** | 7 of the 10 highest-scored are genuine fraud |
| Precision@50 | 0.32 | 16 of the top 50 |
| Precision@100 | 0.22 | |
| Recall@500 | 0.70 | Reviewing 500 of 7,270 finds 70% of known fraud |
| Lift@10 | **96x** | The top 10 carry 96x the population's fraud density |
| Lift@50 | 44x | |
| Confusion (at threshold) | TP 16, FP 32, FN 37, TN 7,185 | |

For a review team that can process fifty cases, a third of what they open is
genuine — against a population where fewer than one in a hundred is. That is the
number that matters operationally, and it is the one precision@k reports.

### Sample size matters more than the model

The same code at the `small` profile scores far worse, because the test window
holds only 17 labelled frauds:

| Profile | Test rows | Frauds | PR-AUC | ROC-AUC | Precision@10 | Lift@10 |
| --- | --- | --- | --- | --- | --- | --- |
| small | 1,451 | 17 | 0.041 | 0.720 | 0.10 | 8.5x |
| **default** | 7,270 | 53 | **0.276** | **0.900** | **0.70** | **96x** |

Nothing changed but the amount of data. Quoting only the default figures would
be flattering; quoting only the small ones would be pessimistic. Both are here
because the gap is the lesson: **metrics computed on a handful of positives are
noise**, and a portfolio that reports a single impressive number without saying
how many positives produced it should be read sceptically.

### Effect of the two methodology fixes

Measured on the small profile, where the sequence was originally diagnosed, so
the comparison is like for like:

| Version | PR-AUC | ROC-AUC | Recall@500 | TP | FP |
| --- | --- | --- | --- | --- | --- |
| Cumulative-count features | 0.024 | 0.607 | 0.35 | 3 | 230 |
| Stationary features | 0.028 | 0.696 | 0.59 | 3 | 255 |
| **+ warm-up exclusion** | **0.041** | **0.720** | 0.59 | **5** | **179** |

False positives fell 22% and true positives rose from 3 to 5 — on identical
data, from changing what the model learns "normal" from.

### Three caveats worth stating

1. **ROC-AUC flatters imbalanced data.** 0.900 looks excellent partly because
   7,185 true negatives dominate the false-positive rate. PR-AUC of 0.276 is the
   more honest summary, and it is the one to quote.
2. **The labels include burst members.** A velocity burst labels every
   transaction in it, but only the *pattern* is anomalous — individual members
   look ordinary in isolation. This deflates precision in a way that is arguably
   correct: the platform should surface the burst, which
   `gold_transaction_velocity` does.
3. **The rules carry the explainable load.** Three of the five injected
   archetypes (round-amount structuring, after-hours new vendor, geographic
   anomaly) are patterns a rule states better than a model infers. That is the
   argument for running both, not evidence against the model.

## Model registry

MLflow would add a service for what is, at this scale, two concerns: where the
artifact is, and what produced it. Those are handled by a joblib file plus a row
in `model_registry`, which together record algorithm, hyperparameters, the
feature list **in order**, training row count and window, the pipeline run, and
the held-out evaluation.

Loading asserts that the stored feature list matches the current code. If someone
adds a feature without retraining, scoring fails loudly instead of feeding the
model a differently-shaped matrix — the failure mode that produces confidently
wrong fraud alerts.

## What would come next

1. **More labels, then a supervised comparison.** With enough investigation
   outcomes, a gradient-boosted baseline would be worth measuring against the
   unsupervised detector on the same held-out period.
2. **Sequence models.** Velocity bursts and structuring are sequence patterns;
   per-transaction features approximate them.
3. **Graph features.** Shared-vendor degree and component size from the Neo4j
   layer are strong fraud-ring signals and are currently used only for
   investigation, not scoring.
4. **Threshold tuning against reviewer capacity.** Contamination is currently a
   configured constant; it should be derived from how many cases the team can
   actually process.
5. **Automated drift-triggered retraining.** The PSI check already detects the
   condition; nothing acts on it yet.
