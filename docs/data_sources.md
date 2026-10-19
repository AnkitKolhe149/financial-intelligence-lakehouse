# Data Sources

The platform ingests six source shapes. Each exists to exercise a different part
of the ingestion layer, and each carries quirks that real sources have.

| Source | Format | Datasets | Reader |
| --- | --- | --- | --- |
| Payments core | CSV (2 schema versions) | transactions | `csv_ingestion.py` |
| Treasury core | JSON (enveloped) | accounts, locations | `json_ingestion.py` |
| Procurement system | XML (attributes + camelCase) | contracts | `xml_ingestion.py` |
| Finance planning | Excel (multi-sheet, title rows) | budgets, departments | `excel_ingestion.py` |
| Source-system REST API | HTTP + pagination | vendors, agencies, transactions | `api_ingestion.py` |
| Event stream | NDJSON landing zone | transaction events | `streaming/` |
| **USAspending.gov** | **Public REST API** | **federal awards, agencies, recipients** | `usaspending_client.py` |

---

## Source A — USAspending.gov (real, public)

**https://api.usaspending.gov** · [endpoint docs](https://api.usaspending.gov/docs/endpoints/)

US federal spending, published under an open licence, no authentication
required. This is the right external source for a financial oversight platform:
real agencies, real recipients, real obligation amounts.

### Endpoints used

| Endpoint | Method | Returns |
| --- | --- | --- |
| `/api/v2/references/toptier_agencies/` | GET | The agency reference list |
| `/api/v2/search/spending_by_award/` | POST | Award-level records for a period and award type |
| `/api/v2/search/spending_over_time/` | POST | Monthly obligation totals |
| `/api/v2/search/spending_by_category/recipient/` | POST | Recipient rankings |

All four were called live during development. A representative response:

```
toptier_agencies      111 agencies
spending_by_award      50 awards over 2 pages
  largest: LOCKHEED MARTIN CORP  $48,063,737,196  (Department of Energy)
spending_over_time      5 monthly periods
spending_by_recipient   OPTUM PUBLIC SECTOR SOLUTIONS  $15,267,792,417
```

### How it is used

Deliberately **bounded**. The client pulls a configurable slice — a date range,
a page cap — never "all federal spending". An ingestion job that tries to
download a national dataset in a demo is a bug, not a feature.

The client implements pagination via `page_metadata.hasNext`, retry with
exponential backoff and full jitter, `Retry-After` handling, a token-bucket rate
limiter and per-attempt logging.

### Fallback

`fetch_or_fallback()` never raises. If the API is unreachable, disabled
(`USASPENDING_ENABLED=false`, which CI sets) or returns nothing, a synthetic
federal-award generator produces **structurally identical** records, so every
downstream stage is exercised the same way. The result records which source
actually supplied the data, and that propagates into lineage.

### Licence

US Government work, public domain. Fetched directly by
`python scripts/download_data.py --usaspending`; the response is written under
`data/external/`, which is gitignored.

---

## Source B — Kaggle fraud datasets (optional, not bundled)

Two commonly-used public datasets:

| Dataset | Shape | Licence |
| --- | --- | --- |
| [kaushalnandania/credit-card-fraud-detection](https://www.kaggle.com/datasets/kaushalnandania/credit-card-fraud-detection) | ~1.85M synthetic transactions with a fraud label | See dataset page |
| [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 rows, 492 frauds; `Time`, `Amount`, `V1..V28`, `Class` | DbCL v1.0 — redistribution restricted |

**Neither is downloaded automatically and neither is committed.** The licences do
not permit redistribution, and committing someone else's dataset into a
portfolio repository is a licensing problem rather than a convenience.

```bash
python scripts/download_data.py --kaggle-instructions
```

Nothing in the platform depends on them. The generator already produces a fraud
population with held-out ground-truth labels, which is what
`src/ml/evaluation.py` scores against. The Kaggle data is useful only if you want
to compare this unsupervised detector against a supervised baseline on a public
benchmark.

---

## Source C — Synthetic data (the primary source)

Generated locally, deterministically, from seed 42.

### Why generated rather than downloaded

1. **Multi-format coverage.** No real dataset ships as CSV *and* JSON *and* XML
   *and* Excel *and* a REST feed *and* a stream. Generating one domain into all
   six is what makes the normalisation layer meaningful.
2. **Known defects.** A data-quality framework can only be *tested* against data
   whose defects are known in advance. The generator injects them at configured
   rates, so expected quarantine volumes are predictable.
3. **No sensitive data.** Every person, vendor, account and tax identifier is
   fictional.

### Realistic structure

Not random noise. The generator models:

- **Per-account spending scale** — each account draws its own amount
  distribution, so "12x the account average" is a different statement for each
  one
- **Vendor affinity** — accounts pay the same handful of vendors ~85% of the
  time, which is what makes a first-time vendor a real signal
- **Fiscal-year-end surge** — ~8% of activity re-dated into September, when US
  agencies obligate remaining budget. The volume check must *not* flag it
- **Weekday and business-hour bias**, with an overnight tail
- **Heavy-tailed amounts** (lognormal), so large payments are not automatically
  outliers
- **Vendor concentration** — award value follows a Zipf-like decay
- **Sole-source clusters** — a fifth of agencies award disproportionately
  without competition
- **Budgets derived from actual spend**, so utilisation lands in a realistic
  band and a configured minority genuinely overruns

### Injected defects

Configured in `config/settings.yaml`:

| Defect | Rate | Becomes |
| --- | --- | --- |
| Duplicate transactions | 1.0% | `DUPLICATE_RECORD` (half exact, half re-issued under a new id) |
| Null account / vendor / department | 0.8 / 0.6 / 1.0% | `NULL_REQUIRED_FIELD` |
| Negative or zero amounts | 0.5% | `INVALID_AMOUNT` |
| Implausibly large amounts | 0.2% | `AMOUNT_EXCEEDS_LIMIT` |
| Invalid currency codes | 0.2% | `INVALID_CURRENCY` |
| Orphan account / vendor ids | 0.3% each | `UNKNOWN_ACCOUNT` / `UNKNOWN_VENDOR` |
| Invalid contract ids | 0.2% | Reference nulled, record kept |
| Future timestamps | 0.1% | `FUTURE_TIMESTAMP` |
| Unparseable timestamps | 0.2% | `INVALID_TIMESTAMP` |
| Whitespace and casing noise | 2.0% | Normalised in Silver |
| Malformed emails | 1.5% | Nulled, not rejected |
| Near-duplicate vendors | 2.0% | Reported, never auto-merged |
| Out-of-order events | 5.0% | Marked late, retained |

### Ground-truth fraud labels

A small cohort (~0.4%) is generated with genuinely suspicious behaviour across
five archetypes: amount spike, velocity burst, after-hours new vendor,
geographic anomaly, round-amount structuring.

Labels are written to `data/raw/labels/fraud_labels.csv` — a **separate** file,
representing investigation outcomes that arrive later. They are never features
and are joined only for evaluation.

### Format quirks, on purpose

| Format | Quirk |
| --- | --- |
| CSV | No type information; everything arrives as text, including the invalid values |
| JSON | Wrapped in a metadata envelope with a declared record count, reconciled on read |
| XML | Natural key held as an **attribute**, elements in camelCase |
| Excel | Multi-sheet; the budgets sheet has a title row above the header, so the reader detects the header rather than assuming row 0 |
| REST | Envelope pagination, a silently-enforced page-size ceiling, optional fault injection for retry testing |
| Stream | Many small NDJSON files, out-of-order event times |

### Scale profiles

| Profile | Transactions | Accounts | Vendors | Agencies |
| --- | --- | --- | --- | --- |
| small | 10,000 | 500 | 150 | 20 |
| default | 50,000 | 2,000 | 500 | 50 |
| large | 500,000 | 10,000 | 2,000 | 80 |

```bash
python -m src.data_generation.generate_all --scale large
python -m src.data_generation.generate_all --transactions 500000 --accounts 10000
```

---

## Source D — Neo4j example datasets

Neo4j publishes example graphs (Movies, Northwind). They are **not** used here:
the platform's own financial graph matches the business domain, and a movie
graph in a financial oversight tool would be a distraction. The graph is loaded
from the platform's own Silver and Gold data.

---

## What is committed to the repository

| Committed | Not committed |
| --- | --- |
| `data/sample/` — tiny samples of each format | `data/raw/` — regenerated in seconds |
| The generators themselves | `data/lake/` — rebuildable from a seed |
| `metadata/lineage.json`, `metadata/benchmarks.json` | `data/external/` — third-party licences |
| Schema contracts | `data/models/` — retrained on demand |

The whole dataset is reproducible from seed 42, so committing it would add
megabytes and prove nothing.
