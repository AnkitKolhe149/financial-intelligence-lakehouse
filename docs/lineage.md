# Data Lineage

> **Generated file.** Produced by `python scripts/generate_docs.py` from `metadata/lineage.json`, which the pipelines emit as they run.

Lineage here is **emitted by the code that performs each transformation**, not reconstructed by parsing SQL afterwards. Each edge records the transformations applied and the row counts on both sides, so the number of records lost between any two datasets is a fact rather than an inference.

Captured from run `20260816T025758-faa81a` at 2026-08-16T02:58:07+00:00 - 39 edges.

## Flow

```mermaid
graph LR
    accounts_json(["accounts.json"])
    bronze_accounts["bronze_accounts"]
    agencies_json(["agencies.json"])
    bronze_agencies["bronze_agencies"]
    budgets_xlsx_budgets(["budgets.xlsx#budgets"])
    bronze_budgets["bronze_budgets"]
    contracts_xml(["contracts.xml"])
    bronze_contracts["bronze_contracts"]
    budgets_xlsx_departments(["budgets.xlsx#departments"])
    bronze_departments["bronze_departments"]
    locations_json(["locations.json"])
    bronze_locations["bronze_locations"]
    transactions_json(["transactions.json"])
    bronze_transactions["bronze_transactions"]
    transactions_2025_csv(["transactions_2025.csv"])
    transactions_2026_csv(["transactions_2026.csv"])
    vendors_json(["vendors.json"])
    bronze_vendors["bronze_vendors"]
    silver_vendors["silver_vendors"]
    dim_vendor{{"dim_vendor"}}
    silver_transactions["silver_transactions"]
    fact_transactions{{"fact_transactions"}}
    silver_accounts["silver_accounts"]
    gold_account_risk{{"gold_account_risk"}}
    silver_agencies["silver_agencies"]
    gold_agency_spending{{"gold_agency_spending"}}
    silver_budgets["silver_budgets"]
    gold_anomaly_transactions{{"gold_anomaly_transactions"}}
    gold_daily_financial_summary{{"gold_daily_financial_summary"}}
    data_quality_results{{"data_quality_results"}}
    gold_data_quality_summary{{"gold_data_quality_summary"}}
    silver_quarantine_transactions["silver_quarantine_transactions"]
    gold_data_quality_trend{{"gold_data_quality_trend"}}
    gold_department_spending{{"gold_department_spending"}}
    silver_departments["silver_departments"]
    pipeline_runs{{"pipeline_runs"}}
    gold_pipeline_runs{{"gold_pipeline_runs"}}
    gold_risk_alerts{{"gold_risk_alerts"}}
    gold_transaction_velocity{{"gold_transaction_velocity"}}
    silver_contracts["silver_contracts"]
    gold_vendor_risk{{"gold_vendor_risk"}}
    silver_locations["silver_locations"]
    accounts_json -->|4 steps| bronze_accounts
    agencies_json -->|4 steps| bronze_agencies
    budgets_xlsx_budgets -->|4 steps| bronze_budgets
    contracts_xml -->|4 steps| bronze_contracts
    budgets_xlsx_departments -->|4 steps| bronze_departments
    locations_json -->|4 steps| bronze_locations
    transactions_json -->|4 steps| bronze_transactions
    transactions_2025_csv -->|4 steps| bronze_transactions
    transactions_2026_csv -->|4 steps| bronze_transactions
    vendors_json -->|4 steps| bronze_vendors
    silver_vendors -->|2 steps| dim_vendor
    silver_transactions -->|3 steps| fact_transactions
    silver_accounts -->|4 steps| gold_account_risk
    silver_transactions -->|4 steps| gold_account_risk
    silver_agencies -->|3 steps| gold_agency_spending
    silver_budgets -->|3 steps| gold_agency_spending
    silver_transactions -->|3 steps| gold_agency_spending
    silver_transactions -->|3 steps| gold_anomaly_transactions
    silver_transactions -->|4 steps| gold_daily_financial_summary
    data_quality_results -->|3 steps| gold_data_quality_summary
    silver_quarantine_transactions -->|3 steps| gold_data_quality_summary
    data_quality_results -->|1 steps| gold_data_quality_trend
    silver_budgets -->|3 steps| gold_department_spending
    silver_departments -->|3 steps| gold_department_spending
    silver_transactions -->|3 steps| gold_department_spending
    pipeline_runs -->|2 steps| gold_pipeline_runs
    gold_anomaly_transactions -->|3 steps| gold_risk_alerts
    silver_transactions -->|3 steps| gold_transaction_velocity
    silver_contracts -->|6 steps| gold_vendor_risk
    silver_transactions -->|6 steps| gold_vendor_risk
    silver_vendors -->|6 steps| gold_vendor_risk
    bronze_accounts -->|8 steps| silver_accounts
    bronze_agencies -->|6 steps| silver_agencies
    bronze_budgets -->|7 steps| silver_budgets
    bronze_contracts -->|7 steps| silver_contracts
    bronze_departments -->|7 steps| silver_departments
    bronze_locations -->|6 steps| silver_locations
    bronze_transactions -->|10 steps| silver_transactions
    bronze_vendors -->|6 steps| silver_vendors
```

## Edges

### `bronze_accounts`

**From `accounts.json`**

- Records: 500 in -> 500 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_agencies`

**From `agencies.json`**

- Records: 20 in -> 20 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_budgets`

**From `budgets.xlsx#budgets`**

- Records: 1,526 in -> 1,526 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_contracts`

**From `contracts.xml`**

- Records: 400 in -> 400 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_departments`

**From `budgets.xlsx#departments`**

- Records: 109 in -> 109 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_locations`

**From `locations.json`**

- Records: 36 in -> 36 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_transactions`

**From `transactions.json`**

- Records: 1,010 in -> 11,110 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

**From `transactions_2025.csv`**

- Records: 6,512 in -> 11,110 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

**From `transactions_2026.csv`**

- Records: 3,588 in -> 11,110 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `bronze_vendors`

**From `vendors.json`**

- Records: 153 in -> 153 out
- Transformations:
  - `schema_mapping`
  - `audit_stamping`
  - `content_hash`
  - `idempotency_check`

### `dim_vendor`

**From `silver_vendors`**

- Records: 0 in -> 153 out
- Transformations:
  - `scd_type_2`
  - `attribute_hash_change_detection`

### `fact_transactions`

**From `silver_transactions`**

- Records: 0 in -> 9,675 out
- Transformations:
  - `surrogate_key_generation`
  - `fact_narrowing`
  - `measure_selection`

### `gold_account_risk`

**From `silver_accounts`**

- Records: 9,675 in -> 481 out (9,194 dropped)
- Transformations:
  - `account_aggregation`
  - `behavioural_metrics`
  - `composite_risk_scoring`
  - `risk_banding`

**From `silver_transactions`**

- Records: 9,675 in -> 481 out (9,194 dropped)
- Transformations:
  - `account_aggregation`
  - `behavioural_metrics`
  - `composite_risk_scoring`
  - `risk_banding`

### `gold_agency_spending`

**From `silver_agencies`**

- Records: 9,675 in -> 40 out (9,635 dropped)
- Transformations:
  - `agency_aggregation`
  - `budget_join`
  - `vendor_concentration`

**From `silver_budgets`**

- Records: 9,675 in -> 40 out (9,635 dropped)
- Transformations:
  - `agency_aggregation`
  - `budget_join`
  - `vendor_concentration`

**From `silver_transactions`**

- Records: 9,675 in -> 40 out (9,635 dropped)
- Transformations:
  - `agency_aggregation`
  - `budget_join`
  - `vendor_concentration`

### `gold_anomaly_transactions`

**From `silver_transactions`**

- Records: 9,675 in -> 286 out (9,389 dropped)
- Transformations:
  - `anomaly_selection`
  - `risk_scoring`
  - `explanation_attachment`

### `gold_daily_financial_summary`

**From `silver_transactions`**

- Records: 9,675 in -> 573 out (9,102 dropped)
- Transformations:
  - `daily_aggregation`
  - `distribution_statistics`
  - `anomaly_counts`
  - `rolling_7d_baseline`

### `gold_data_quality_summary`

**From `data_quality_results`**

- Records: 0 in -> 7 out
- Transformations:
  - `latest_scorecard_selection`
  - `quarantine_join`
  - `rate_computation`

**From `silver_quarantine_transactions`**

- Records: 0 in -> 7 out
- Transformations:
  - `latest_scorecard_selection`
  - `quarantine_join`
  - `rate_computation`

### `gold_data_quality_trend`

**From `data_quality_results`**

- Records: 0 in -> 7 out
- Transformations:
  - `daily_quality_aggregation`

### `gold_department_spending`

**From `silver_budgets`**

- Records: 9,675 in -> 210 out (9,465 dropped)
- Transformations:
  - `department_aggregation`
  - `budget_join`
  - `overrun_detection`

**From `silver_departments`**

- Records: 9,675 in -> 210 out (9,465 dropped)
- Transformations:
  - `department_aggregation`
  - `budget_join`
  - `overrun_detection`

**From `silver_transactions`**

- Records: 9,675 in -> 210 out (9,465 dropped)
- Transformations:
  - `department_aggregation`
  - `budget_join`
  - `overrun_detection`

### `gold_pipeline_runs`

**From `pipeline_runs`**

- Records: 0 in -> 1 out
- Transformations:
  - `metadata_snapshot`
  - `throughput_computation`

### `gold_risk_alerts`

**From `gold_anomaly_transactions`**

- Records: 9,675 in -> 46 out (9,629 dropped)
- Transformations:
  - `alert_threshold_filter`
  - `deterministic_alert_id`
  - `case_lifecycle`

### `gold_transaction_velocity`

**From `silver_transactions`**

- Records: 9,675 in -> 9,554 out (121 dropped)
- Transformations:
  - `hourly_aggregation`
  - `velocity_computation`
  - `burst_detection`

### `gold_vendor_risk`

**From `silver_contracts`**

- Records: 9,675 in -> 153 out (9,522 dropped)
- Transformations:
  - `vendor_aggregation`
  - `payer_breadth`
  - `contract_coverage`
  - `growth_vs_baseline`
  - `duplicate_registration_detection`
  - `composite_risk_scoring`

**From `silver_transactions`**

- Records: 9,675 in -> 153 out (9,522 dropped)
- Transformations:
  - `vendor_aggregation`
  - `payer_breadth`
  - `contract_coverage`
  - `growth_vs_baseline`
  - `duplicate_registration_detection`
  - `composite_risk_scoring`

**From `silver_vendors`**

- Records: 9,675 in -> 153 out (9,522 dropped)
- Transformations:
  - `vendor_aggregation`
  - `payer_breadth`
  - `contract_coverage`
  - `growth_vs_baseline`
  - `duplicate_registration_detection`
  - `composite_risk_scoring`

### `silver_accounts`

**From `bronze_accounts`**

- Records: 500 in -> 500 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `currency_normalisation`
  - `deduplication`
  - `referential_integrity`
  - `quarantine_invalid_records`

### `silver_agencies`

**From `bronze_agencies`**

- Records: 20 in -> 20 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `deduplication`
  - `quarantine_invalid_records`

### `silver_budgets`

**From `bronze_budgets`**

- Records: 1,526 in -> 1,526 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `deduplication`
  - `referential_integrity`
  - `quarantine_invalid_records`

### `silver_contracts`

**From `bronze_contracts`**

- Records: 400 in -> 400 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `deduplication`
  - `referential_integrity`
  - `quarantine_invalid_records`

### `silver_departments`

**From `bronze_departments`**

- Records: 109 in -> 109 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `deduplication`
  - `referential_integrity`
  - `quarantine_invalid_records`

### `silver_locations`

**From `bronze_locations`**

- Records: 36 in -> 36 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `deduplication`
  - `quarantine_invalid_records`

### `silver_transactions`

**From `bronze_transactions`**

- Records: 11,110 in -> 9,675 out (1,435 dropped)
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `timestamp_parsing`
  - `currency_normalisation`
  - `deduplication`
  - `referential_integrity`
  - `derived_columns`
  - `quarantine_invalid_records`

### `silver_vendors`

**From `bronze_vendors`**

- Records: 153 in -> 153 out
- Transformations:
  - `schema_alignment`
  - `type_casting`
  - `string_normalisation`
  - `null_policy`
  - `deduplication`
  - `quarantine_invalid_records`

## Where records go

Records dropped between Bronze and Silver are **not deleted**. Every one is written to the quarantine dataset with its error code, the raw payload as received, the source file and the run that rejected it. `sql/05_data_quality.sql` reports them by error code and by source system.
