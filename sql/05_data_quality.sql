-- ===========================================================================
-- 05 - Data quality
--
-- The questions that decide whether the other nine files can be believed.
-- ===========================================================================

-- name: current_quality_scorecard
-- Latest score per dataset across all six dimensions.
SELECT
    dataset,
    layer,
    ROUND(overall_score, 2)     AS overall,
    ROUND(completeness, 2)      AS completeness,
    ROUND(validity, 2)          AS validity,
    ROUND(uniqueness, 2)        AS uniqueness,
    ROUND(integrity, 2)         AS integrity,
    ROUND(freshness, 2)         AS freshness,
    ROUND(volume, 2)            AS volume,
    records_evaluated,
    checks_passed || '/' || checks_total AS checks,
    status,
    quarantined_records
FROM gold_data_quality_summary
ORDER BY overall_score ASC;

-- name: quality_trend
-- Is quality improving or degrading? The day-over-day delta is what turns a
-- score into an early warning.
SELECT
    dataset,
    evaluated_date,
    ROUND(overall_score, 2)                                             AS overall,
    ROUND(overall_score - LAG(overall_score) OVER (
        PARTITION BY dataset ORDER BY evaluated_date
    ), 3)                                                               AS change_vs_previous,
    ROUND(AVG(overall_score) OVER (
        PARTITION BY dataset ORDER BY evaluated_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)                                                               AS rolling_7d_mean
FROM gold_data_quality_trend
ORDER BY dataset, evaluated_date DESC;

-- name: quarantine_by_error_code
-- Why records are being rejected, ranked. Drives where remediation effort goes.
SELECT
    error_code,
    COUNT(*)                                                        AS records,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)              AS share_pct,
    COUNT(DISTINCT source_file)                                     AS source_files,
    MIN(detected_at)                                                AS first_seen,
    MAX(detected_at)                                                AS last_seen
FROM silver_quarantine_transactions
GROUP BY error_code
ORDER BY records DESC;

-- name: worst_source_systems
-- Which upstream system produces the most invalid records. This is the query
-- that turns a data-quality complaint into a conversation with a named team.
SELECT
    source_system,
    source_file,
    COUNT(*)                                                    AS quarantined,
    COUNT(DISTINCT error_code)                                  AS distinct_error_types,
    STRING_AGG(DISTINCT error_code, ', ')                       AS error_codes
FROM silver_quarantine_transactions
GROUP BY source_system, source_file
ORDER BY quarantined DESC;

-- name: quarantine_over_time
-- Rejection volume per day. A step change here means something broke upstream.
SELECT
    quarantine_date,
    COUNT(*)                                                            AS quarantined,
    COUNT(DISTINCT error_code)                                          AS error_types,
    COUNT(*) FILTER (WHERE error_code = 'DUPLICATE_RECORD')             AS duplicates,
    COUNT(*) FILTER (WHERE error_code = 'NULL_REQUIRED_FIELD')          AS missing_required,
    COUNT(*) FILTER (WHERE error_code LIKE 'UNKNOWN_%')                 AS broken_references,
    COUNT(*) FILTER (WHERE error_code LIKE 'INVALID_%')                 AS invalid_values
FROM silver_quarantine_transactions
GROUP BY quarantine_date
ORDER BY quarantine_date DESC;

-- name: completeness_by_column
-- Column-level null rates on the fact table. The optional columns are expected
-- to have nulls; a jump in a required one is an incident.
SELECT 'account_id'        AS column_name, ROUND(100.0 * COUNT(account_id) / COUNT(*), 3)        AS populated_pct FROM silver_transactions
UNION ALL SELECT 'vendor_id',        ROUND(100.0 * COUNT(vendor_id) / COUNT(*), 3)        FROM silver_transactions
UNION ALL SELECT 'contract_id',      ROUND(100.0 * COUNT(contract_id) / COUNT(*), 3)      FROM silver_transactions
UNION ALL SELECT 'department_id',    ROUND(100.0 * COUNT(department_id) / COUNT(*), 3)    FROM silver_transactions
UNION ALL SELECT 'agency_id',        ROUND(100.0 * COUNT(agency_id) / COUNT(*), 3)        FROM silver_transactions
UNION ALL SELECT 'merchant_category',ROUND(100.0 * COUNT(merchant_category) / COUNT(*), 3) FROM silver_transactions
UNION ALL SELECT 'location_id',      ROUND(100.0 * COUNT(location_id) / COUNT(*), 3)      FROM silver_transactions
ORDER BY populated_pct ASC;

-- name: schema_version_distribution
-- Evidence that schema evolution is real and handled: v1.0 rows have no
-- merchant_category, v1.1 rows do, and both live in the same table.
SELECT
    _schema_version                                                     AS schema_version,
    COUNT(*)                                                            AS records,
    COUNT(merchant_category)                                            AS with_merchant_category,
    ROUND(100.0 * COUNT(merchant_category) / COUNT(*), 2)               AS populated_pct
FROM silver_transactions
GROUP BY _schema_version
ORDER BY schema_version;
