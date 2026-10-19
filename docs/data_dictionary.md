# Data Dictionary

> **Generated file.** Produced by `python scripts/generate_docs.py` from `src/core/schema.py`. Edit the schema registry, not this document.

Generated 2026-08-16 02:59 UTC - 9 canonical datasets.

Every dataset below is declared once in the schema registry and consumed by ingestion, Silver casting, the data-quality suites, the published JSON contracts and this document. A column added there appears in all five.

## Contents

- [agency](#agency)
- [department](#department)
- [vendor](#vendor)
- [contract](#contract)
- [account](#account)
- [budget](#budget)
- [location](#location)
- [transaction](#transaction)
- [transaction_event](#transaction_event)

## agency

A federal agency that owns budget authority and awards contracts.

| Property | Value |
| --- | --- |
| Business key | `agency_id` |
| Bronze table | `bronze_agencies` |
| Silver table | `silver_agencies` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | public |
| Partition column | `none` |
| Deduplication hash columns | `agency_id, agency_name, agency_abbreviation` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `agency_id` | string | **NOT NULL** | Surrogate agency identifier. | Primary key for the agency dimension. |
| `agency_name` | string | **NOT NULL** | Official agency name. | Human-readable agency label used in reporting. |
| `agency_abbreviation` | string | yes | Short agency code, e.g. DOT. | - |
| `agency_type` | string | yes | Branch of government the agency belongs to. (one of: EXECUTIVE, LEGISLATIVE, JUDICIAL, INDEPENDENT) | - |
| `toptier_code` | string | yes | USAspending top-tier agency code where known. | - |
| `established_year` | int | yes | Year the agency was established. (min 1780; max 2100) | - |
| `headquarters_state` | string | yes | Two-letter US state code of the HQ. | - |
| `annual_budget_usd` | decimal | yes | Published annual budget authority. (min 0) | Denominator for budget-utilisation metrics. |

## department

An organisational unit inside an agency that incurs spending.

| Property | Value |
| --- | --- |
| Business key | `department_id` |
| Bronze table | `bronze_departments` |
| Silver table | `silver_departments` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `department_id, department_name, agency_id` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `department_id` | string | **NOT NULL** | Surrogate department identifier. | - |
| `department_name` | string | **NOT NULL** | Department name. | - |
| `agency_id` | string | **NOT NULL** | Owning agency. | Roll-up key from department spending to agency spending. |
| `cost_center` | string | yes | Finance cost-centre code. | - |
| `budget_allocation_usd` | decimal | yes | Budget allocated to the department. (min 0) | - |
| `manager_name` | string | yes | Fictional responsible manager. | Ownership contact for spending oversight. |

### Referential integrity

| Column | References | On violation |
| --- | --- | --- |
| `agency_id` | `agency.agency_id` | quarantine (`REFERENTIAL_INTEGRITY`) |

## vendor

A supplier receiving payments; the 'recipient' in USAspending terms.

| Property | Value |
| --- | --- |
| Business key | `vendor_id` |
| Bronze table | `bronze_vendors` |
| Silver table | `silver_vendors` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `vendor_id, vendor_name, tax_id, country` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `vendor_id` | string | **NOT NULL** | Surrogate vendor identifier. | - |
| `vendor_name` | string | **NOT NULL** | Legal vendor name. | Subject of vendor-concentration and risk analysis. |
| `vendor_legal_name` | string | yes | Registered legal entity name. | - |
| `tax_id` | string | yes | Fictional tax identifier (never a real EIN/SSN). | Entity-resolution key for near-duplicate vendors. |
| `vendor_type` | string | yes | Business classification. (one of: SMALL_BUSINESS, LARGE_BUSINESS, NON_PROFIT, GOVERNMENT, FOREIGN_ENTITY, SOLE_PROPRIETOR) | - |
| `industry_code` | string | yes | NAICS-style industry code. | - |
| `contact_email` | string | yes | Fictional contact address. | - |
| `phone` | string | yes | Fictional contact phone number. | - |
| `address_line1` | string | yes | Street address. | - |
| `city` | string | yes | City. | - |
| `state` | string | yes | Two-letter state code where applicable. | - |
| `postal_code` | string | yes | Postal code. | - |
| `country` | string | yes | ISO-3166 alpha-2 country code. | - |
| `registration_date` | date | yes | Date the vendor was registered. | Drives the 'new vendor' risk rule. |
| `is_active` | bool | yes | Whether the vendor is currently active. | - |
| `risk_flag` | string | yes | Externally supplied watch-list flag. (one of: NONE, WATCH, SANCTIONED) | - |

## contract

An award linking a vendor to an agency with a ceiling value.

| Property | Value |
| --- | --- |
| Business key | `contract_id` |
| Bronze table | `bronze_contracts` |
| Silver table | `silver_contracts` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `contract_id, vendor_id, agency_id, contract_value_usd` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `contract_id` | string | **NOT NULL** | Surrogate contract/award identifier. | - |
| `award_id` | string | yes | External award identifier (PIID-style). | - |
| `vendor_id` | string | **NOT NULL** | Awarded vendor. | - |
| `agency_id` | string | **NOT NULL** | Awarding agency. | - |
| `contract_type` | string | yes | Award instrument type. (one of: FIXED_PRICE, COST_PLUS, TIME_AND_MATERIALS, IDIQ, GRANT, PURCHASE_ORDER) | - |
| `contract_value_usd` | decimal | yes | Total obligated ceiling value. (min 0) | Compared against cumulative payments to detect overruns. |
| `start_date` | date | yes | Period-of-performance start. | - |
| `end_date` | date | yes | Period-of-performance end. | - |
| `status` | string | yes | Lifecycle status. (one of: ACTIVE, COMPLETED, TERMINATED, PENDING) | - |
| `description` | string | yes | Free-text description of the award. | - |
| `competition_type` | string | yes | How the award was competed. (one of: FULL_AND_OPEN, LIMITED, SOLE_SOURCE, SET_ASIDE) | Sole-source concentration is an oversight red flag. |

### Referential integrity

| Column | References | On violation |
| --- | --- | --- |
| `vendor_id` | `vendor.vendor_id` | quarantine (`REFERENTIAL_INTEGRITY`) |
| `agency_id` | `agency.agency_id` | quarantine (`REFERENTIAL_INTEGRITY`) |

## account

A departmental disbursement account that transactions are made from.

| Property | Value |
| --- | --- |
| Business key | `account_id` |
| Bronze table | `bronze_accounts` |
| Silver table | `silver_accounts` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `account_id, department_id, account_type` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `account_id` | string | **NOT NULL** | Surrogate account identifier. | - |
| `account_number` | string | yes | Masked account number (fictional). | - |
| `department_id` | string | **NOT NULL** | Owning department. | - |
| `account_type` | string | yes | Kind of disbursement account. (one of: OPERATING, PAYROLL, GRANT, CAPITAL, TRAVEL, PROCUREMENT) | - |
| `currency` | string | yes | Account base currency. | - |
| `opened_date` | date | yes | Date the account was opened. | - |
| `status` | string | yes | Account status. (one of: ACTIVE, DORMANT, CLOSED, FROZEN) | - |
| `credit_limit_usd` | decimal | yes | Spending limit for the account. (min 0) | - |
| `owner_name` | string | yes | Fictional account owner. | - |
| `region` | string | yes | Operating region of the account. | - |

### Referential integrity

| Column | References | On violation |
| --- | --- | --- |
| `department_id` | `department.department_id` | quarantine (`REFERENTIAL_INTEGRITY`) |

## budget

Fiscal-year budget lines by agency and department.

| Property | Value |
| --- | --- |
| Business key | `budget_id` |
| Bronze table | `bronze_budgets` |
| Silver table | `silver_budgets` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `budget_id, agency_id, department_id, fiscal_year` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `budget_id` | string | **NOT NULL** | Surrogate budget-line identifier. | - |
| `fiscal_year` | int | **NOT NULL** | US federal fiscal year. (min 2000; max 2100) | - |
| `agency_id` | string | **NOT NULL** | Agency the budget belongs to. | - |
| `department_id` | string | yes | Department the budget belongs to. | - |
| `budget_category` | string | yes | Spending category. (one of: PERSONNEL, OPERATIONS, CAPITAL, GRANTS, IT, TRAVEL, MAINTENANCE) | - |
| `allocated_amount_usd` | decimal | yes | Amount allocated. (min 0) | - |
| `spent_amount_usd` | decimal | yes | Amount spent to date. (min 0) | Numerator of budget utilisation. |
| `quarter` | string | yes | Fiscal quarter. (one of: Q1, Q2, Q3, Q4) | - |

### Referential integrity

| Column | References | On violation |
| --- | --- | --- |
| `agency_id` | `agency.agency_id` | quarantine (`REFERENTIAL_INTEGRITY`) |
| `department_id` | `department.department_id` | null the reference and keep the record |

## location

Geography where a transaction was initiated.

| Property | Value |
| --- | --- |
| Business key | `location_id` |
| Bronze table | `bronze_locations` |
| Silver table | `silver_locations` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `location_id` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `location_id` | string | **NOT NULL** | Surrogate location identifier. | - |
| `city` | string | yes | City name. | - |
| `state` | string | yes | Two-letter state code. | - |
| `country` | string | yes | ISO-3166 alpha-2 country code. | - |
| `latitude` | float | yes | Latitude. (min -90; max 90) | - |
| `longitude` | float | yes | Longitude. (min -180; max 180) | - |
| `region` | string | yes | Reporting region. | - |

## transaction

A single financial disbursement - the fact table of the platform.

| Property | Value |
| --- | --- |
| Business key | `transaction_id` |
| Bronze table | `bronze_transactions` |
| Silver table | `silver_transactions` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `transaction_date` |
| Deduplication hash columns | `account_id, vendor_id, amount, currency, transaction_timestamp` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `transaction_id` | string | **NOT NULL** | Unique transaction identifier. | Primary key; uniqueness is a blocking quality check. |
| `account_id` | string | **NOT NULL** | Account the funds were disbursed from. | Join key to the account dimension. |
| `vendor_id` | string | **NOT NULL** | Vendor that received the funds. | Join key to the vendor dimension. |
| `contract_id` | string | yes | Contract the payment was made under, if any. | Nullable: petty-cash and travel spend has no contract. |
| `department_id` | string | yes | Department that incurred the spend. | - |
| `agency_id` | string | yes | Agency the spend rolls up to. | - |
| `amount` | decimal | **NOT NULL** | Transaction amount in the stated currency. (min 0) | Measure for every spending aggregation. |
| `currency` | string | **NOT NULL** | ISO-4217 currency code. | Normalised to upper case; unknown codes are quarantined. |
| `amount_usd` | decimal | yes | Amount converted to USD at a fixed reference rate. (min 0) | Currency-neutral measure for cross-agency totals. |
| `transaction_timestamp` | timestamp | **NOT NULL** | When the transaction occurred. | Drives temporal analysis, velocity and ML train/test splits. |
| `transaction_date` | date | yes | Date component, used as the partition key. | - |
| `posted_timestamp` | timestamp | yes | When the source system posted the record. | Difference from transaction_timestamp reveals late arrivals. |
| `transaction_type` | string | yes | Payment instrument. (one of: PAYMENT, REFUND, ADJUSTMENT, TRANSFER, PAYROLL, GRANT_DISBURSEMENT) | - |
| `payment_method` | string | yes | Method of payment. (one of: ACH, WIRE, CHECK, CARD, EFT) | - |
| `merchant_category` | string | yes | Merchant category code. Added in schema version 1.1 - see docs/schema_evolution. | Demonstrates additive schema evolution. |
| `description` | string | yes | Free-text payment description. | - |
| `location_id` | string | yes | Where the transaction was initiated. | - |
| `channel` | string | yes | Origination channel. (one of: ONLINE, BRANCH, BATCH, MOBILE, API) | - |
| `status` | string | yes | Settlement status. (one of: SETTLED, PENDING, REVERSED, FAILED) | - |
| `fiscal_year` | int | yes | US federal fiscal year of the transaction. | - |

### Referential integrity

| Column | References | On violation |
| --- | --- | --- |
| `account_id` | `account.account_id` | quarantine (`UNKNOWN_ACCOUNT`) |
| `vendor_id` | `vendor.vendor_id` | quarantine (`UNKNOWN_VENDOR`) |
| `contract_id` | `contract.contract_id` | null the reference and keep the record |

## transaction_event

Streaming envelope carrying a transaction plus event-time metadata.

| Property | Value |
| --- | --- |
| Business key | `event_id` |
| Bronze table | `bronze_transaction_events` |
| Silver table | `silver_transaction_events` |
| Schema version | 1.0 |
| Owner | data-engineering |
| Classification | internal |
| Partition column | `none` |
| Deduplication hash columns | `transaction_id, account_id, vendor_id, amount, event_timestamp` |

### Fields

| Field | Type | Nullable | Description | Business meaning |
| --- | --- | --- | --- | --- |
| `event_id` | string | **NOT NULL** | Unique event identifier. | - |
| `event_type` | string | yes | Event kind. (one of: TRANSACTION_CREATED, TRANSACTION_UPDATED, TRANSACTION_REVERSED) | - |
| `event_timestamp` | timestamp | **NOT NULL** | Event time as stamped by the producer. | Event time, not processing time - watermarking uses this. |
| `ingest_timestamp` | timestamp | yes | When the platform received the event. | Gap vs event_timestamp is the observed lateness. |
| `producer` | string | yes | Source system that emitted the event. | - |
| `schema_version` | string | yes | Producer's schema version. | - |
| `transaction_id` | string | **NOT NULL** | Transaction the event describes. | - |
| `account_id` | string | yes | Account involved. | - |
| `vendor_id` | string | yes | Vendor involved. | - |
| `amount` | decimal | yes | Transaction amount. (min 0) | - |
| `currency` | string | yes | ISO-4217 currency code. | - |
| `location_id` | string | yes | Origination location. | - |
| `channel` | string | yes | Origination channel. | - |

## Audit columns

Provenance is attached at ingestion and carried through every layer. Underscore-prefixed names never collide with business columns and are trivially strippable when publishing.

### Bronze

| Column | Type | Purpose |
| --- | --- | --- |
| `_source_system` | string | Which upstream system supplied the record. |
| `_source_file` | string | The file or endpoint it arrived in. |
| `_source_format` | string | csv | json | xml | xlsx | rest_api. |
| `_ingestion_timestamp` | timestamp | When the platform read it. |
| `_ingestion_run_id` | string | The pipeline run that ingested it. |
| `_record_hash` | string | Content fingerprint; drives deduplication and idempotency. |
| `_schema_version` | string | Which source schema version the record conforms to. |
| `_source_row_number` | int | Position within the source file, for tracing back. |

### Silver

| Column | Type | Purpose |
| --- | --- | --- |
| `_source_system` | string | Carried forward from Bronze. |
| `_record_hash` | string | Carried forward; the dedup key. |
| `_ingestion_run_id` | string | The run that ingested the record. |
| `_silver_run_id` | string | The run that validated and typed it. |
| `_processed_timestamp` | timestamp | When Silver processed it. |
| `_schema_version` | string | Source schema version. |
| `_is_late_arriving` | bool | True when posting lagged occurrence by over 24 hours. |

## Schema evolution

### transaction

| Version | Columns added |
| --- | --- |
| 1.0 | (initial) |
| 1.1 | `merchant_category` |

Bronze accepts new columns without a migration (schema-on-read) and Silver back-fills nulls for records written before a column existed. Both schema versions coexist in one table; `sql/05_data_quality.sql` reports the split.

## Derived columns

| Column | Computed in | Definition |
| --- | --- | --- |
| `amount_usd` | Silver | `amount` x a fixed reference rate for its currency. A production system would join a rates dimension with effective dating; the fixed rate keeps the demo deterministic and is documented as a simplification. |
| `transaction_date` | Silver | Date part of `transaction_timestamp` (UTC). |
| `transaction_month` | Silver | `YYYY-MM`; the partition key. Monthly rather than daily because daily partitions measured badly at this volume - see `docs/performance.md`. |
| `fiscal_year` | Silver | US federal fiscal year: the calendar year, plus one from 1 October. |
| `_is_late_arriving` | Silver | `posted_timestamp` more than 24 hours after `transaction_timestamp`. |
| `risk_score` / `risk_level` / `risk_reasons` | Gold | Rule engine output; every point is attributable to a named rule. |
| `anomaly_score` / `is_anomaly` | Gold | Isolation Forest output, normalised to [0, 1] where 1 is most anomalous. |
