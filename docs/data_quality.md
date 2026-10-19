# Data Quality

Two things are deliberately separated here, because they answer different
questions:

| | Enforcement | Measurement |
| --- | --- | --- |
| Module | `src/silver/validation.py` | `src/data_quality/` |
| Question | "May this record proceed?" | "How healthy is this dataset, and is it getting worse?" |
| Output | valid / quarantined | a scorecard, stored per run |
| Granularity | record | dataset |

A dataset can be 100% valid — everything invalid was quarantined — and still be
unhealthy, because 40% of the day's records were quarantined. Only the second
question predicts an incident.

## Enforcement: what happens to a bad record

Rules are applied in a fixed order and **first failure wins**, so quarantine
counts by error code sum to the number of rejected records rather than
double-counting a row that is wrong three ways.

The order matters more than it looks. Unparseable values are checked *before*
missing ones:

```
1. unparseable    was there a value that could not be read?   -> INVALID_TIMESTAMP / INVALID_AMOUNT
2. required       is a mandatory field null?                  -> NULL_REQUIRED_FIELD
3. future dates                                               -> FUTURE_TIMESTAMP
4. amounts        non-positive, above the platform ceiling    -> INVALID_AMOUNT / AMOUNT_EXCEEDS_LIMIT
5. currency       outside the approved set                    -> INVALID_CURRENCY
6. enums          reported, not rejected
7. references     blocking -> quarantine; non-blocking -> null the reference
```

Casting turns `"not-a-date"` into `NaT`, which is indistinguishable from a
genuinely absent value once the cast has happened. Validation therefore compares
against the *pre-cast* frame. Without that, an unparseable timestamp gets
reported as `NULL_REQUIRED_FIELD` — technically true of the cast value, and
actively misleading to whoever has to go and fix the source system. This was a
real bug, caught by `tests/test_transformations.py`.

### Error codes

`NULL_REQUIRED_FIELD`, `INVALID_AMOUNT`, `INVALID_CURRENCY`, `DUPLICATE_RECORD`,
`UNKNOWN_ACCOUNT`, `UNKNOWN_VENDOR`, `UNKNOWN_CONTRACT`, `INVALID_TIMESTAMP`,
`FUTURE_TIMESTAMP`, `SCHEMA_MISMATCH`, `INVALID_ENUM_VALUE`,
`AMOUNT_EXCEEDS_LIMIT`, `REFERENTIAL_INTEGRITY`.

Stable strings, not library prose, so `WHERE error_code = 'INVALID_CURRENCY'`
keeps working across Pydantic versions.

### Blocking versus non-blocking references

An unknown **account** or **vendor** quarantines the record: the payment cannot
be attributed. An unknown **contract** nulls the reference and keeps the record:
petty-cash and travel spend legitimately has no contract, and discarding a real
payment over it would lose money from the totals.

### Quarantine

Every rejection is written with the raw payload as received, the error code and
field, the source system and file, and the run that rejected it. Nothing is
deleted — see [ADR-004](architecture_decisions.md#adr-004-quarantine-instead-of-dropping-bad-records).

## Measurement: six scored dimensions

| Dimension | Weight | Asks |
| --- | --- | --- |
| Completeness | 0.25 | Are required fields populated? |
| Validity | 0.25 | Are values in range, in the approved set, parseable? |
| Uniqueness | 0.15 | Does the business key identify one record? |
| Integrity | 0.20 | Do foreign keys resolve? |
| Freshness | 0.05 | Is the newest record recent enough to act on? |
| Volume | 0.10 | Is today's record count plausible against its own history? |

Schema conformance and distribution drift are also checked but carry no weight —
they are early warnings, not health measures.

### How scores aggregate

**Within a dimension**, checks are averaged weighted by records evaluated. A
check over 500,000 rows counts for more than one over a 36-row reference table;
a plain mean would let a tiny dataset's perfect score paper over the fact
table's problems.

**Across dimensions**, configured weights apply, renormalised over the
dimensions actually measured. A dataset with no timestamp is not penalised on
freshness for having no timestamp to be stale.

**Unmeasured dimensions report `None`, not 100%.** "Not measured" and "perfect"
must never look the same on a dashboard.

A failed **blocking** check forces `FAIL` regardless of the headline number:
99.4% overall is not reassuring if every transaction id is duplicated.

### Graded, not binary

Every check returns a score in `[0, 1]` as well as a boolean. "99.1% complete"
and "62% complete" both fail a strict threshold, but only one is an emergency.

## Check calibration: two that were wrong first

### Volume, at low counts

A flat "±40% from the median" rule is statistically wrong at small volumes. At
14 records a day, ordinary Poisson noise exceeds 40% constantly — the check
fired on a third of all days and reported nothing real. That is how monitoring
gets switched off.

The check now picks its regime from the data: above 50 records/day the
percentage band applies; below it, a Poisson band of `median ± 3·√median`, which
is the actual expected variation of a count process. The September
fiscal-year-end surge is real seasonality the check must *not* flag, which makes
it a better test than uniform data.

### Optional completeness, and schema evolution

`merchant_category` was added in schema v1.1, so it is null for every v1.0
record by design. Including it in the optional-completeness check guaranteed a
permanent warning that said nothing about health. Columns introduced by a later
schema version are now excluded and tracked by the schema-evolution report
instead.

## Distribution drift (PSI)

Population Stability Index against a stored baseline, the standard measure in
credit and fraud modelling:

| PSI | Reading |
| --- | --- |
| < 0.10 | stable |
| 0.10 – 0.25 | moderate shift |
| > 0.25 | significant shift |

The first run captures a baseline and reports informationally rather than
failing. A significant shift in transaction amounts is exactly the signal that a
model trained last month is no longer calibrated, so this is model monitoring as
much as data monitoring.

## Where results go

- `data_quality_results` — one scorecard per dataset per run
- `data_quality_checks` — every individual check, queryable
- `gold_data_quality_summary` — published into the lake so analytical SQL can
  join *"how much did agency X spend"* to *"how much should we trust that number"*
- `gold_data_quality_trend` — daily scores, because the trend is what lets
  someone intervene before an incident rather than after one

## A measured run

From the default-scale demo (50,500 generated transactions, 48,464 in Silver):

```
Transaction dataset, silver layer
  Completeness    91.19%
  Validity       100.00%
  Uniqueness     100.00%
  Integrity      100.00%
  Freshness      100.00%
  Volume          54.31%
  Overall         93.23%   [WARN]
  Checks         19/20 passed   (48,464 records evaluated)

Platform score (records-weighted across 7 datasets): 94.22%
```

Validity, uniqueness and integrity read 100% because everything that violated
them was quarantined — that is enforcement working. Completeness sits at 91%
because optional fields are legitimately sparse.

Volume at 54% is the honest one, and worth reading carefully rather than tuning
away. The generated data spans 19 months with a deliberate September
fiscal-year-end surge and a strong weekday bias, so daily counts genuinely do
swing well beyond the configured band. The check is reporting a real property of
the data. Two responses would be legitimate: widen the tolerance because this
variation is expected, or compare each day against the same weekday rather than
against a flat median. What is *not* legitimate is lowering the weight until the
headline score looks better.

Quarantine for the same run:

| Error code | Records |
| --- | --- |
| DUPLICATE_RECORD | 5,292 |
| NULL_REQUIRED_FIELD | 855 |
| INVALID_AMOUNT | 248 |
| UNKNOWN_ACCOUNT | 155 |
| UNKNOWN_VENDOR | 151 |
| FUTURE_TIMESTAMP | 124 |
| INVALID_TIMESTAMP | 101 |
| AMOUNT_EXCEEDS_LIMIT | 96 |
| INVALID_CURRENCY | 64 |
| **Total** | **7,086** |

These match the injection rates in `config/settings.yaml`, which is the point of
generating defects deliberately: the expected quarantine volume is known in
advance, so the framework can be *tested* rather than merely exercised.

## Adding a check

Most checks are inferred from the schema registry — a NOT NULL column implies a
completeness check, a declared `domain` implies allowed-values, a foreign key
implies referential integrity. Adding a column with a domain gets it validated
automatically.

For anything that cannot be inferred, subclass `DataQualityCheck`, return a
`CheckResult` with a graded score, and add it in `src/data_quality/suites.py`.
A check that raises is recorded as a *failed check*, never allowed to abort the
quality run — losing the whole scorecard because one check hit an unexpected
dtype is the fragility that makes teams stop running it.
