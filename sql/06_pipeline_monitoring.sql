-- ===========================================================================
-- 06 - Pipeline monitoring
--
-- Operational health, queryable alongside business data because the two
-- questions are usually asked together: "spending looks wrong on the 14th" is
-- often answered by "the load failed on the 14th".
-- ===========================================================================

-- name: recent_pipeline_runs
SELECT
    run_id,
    pipeline_name,
    pipeline_version,
    engine,
    source,
    start_time,
    ROUND(duration_seconds, 2)      AS duration_s,
    records_read,
    records_written,
    records_quarantined,
    records_failed,
    ROUND(records_per_second, 1)    AS records_per_second,
    ROUND(quarantine_rate_pct, 2)   AS quarantine_rate_pct,
    status
FROM gold_pipeline_runs
ORDER BY start_time DESC
LIMIT 30;

-- name: pipeline_failures
-- Failed runs with their error, newest first.
SELECT
    run_id,
    pipeline_name,
    start_time,
    ROUND(duration_seconds, 2)  AS duration_s,
    records_read,
    status,
    SUBSTRING(COALESCE(error_message, ''), 1, 300) AS error
FROM gold_pipeline_runs
WHERE NOT succeeded
ORDER BY start_time DESC
LIMIT 25;

-- name: throughput_trend
-- Is the pipeline getting slower as data grows? Duration alone would not tell
-- you - it should grow with volume. Records-per-second is the metric that
-- separates "more data" from "a regression".
SELECT
    run_date,
    COUNT(*)                                            AS runs,
    SUM(records_read)                                   AS records_read,
    ROUND(AVG(duration_seconds), 2)                     AS mean_duration_s,
    ROUND(MAX(duration_seconds), 2)                     AS slowest_run_s,
    ROUND(AVG(records_per_second), 1)                   AS mean_records_per_second,
    COUNT(*) FILTER (WHERE NOT succeeded)               AS failures
FROM gold_pipeline_runs
GROUP BY run_date
ORDER BY run_date DESC;

-- name: pipeline_reliability
-- Success rate per pipeline over its whole history.
SELECT
    pipeline_name,
    COUNT(*)                                                            AS total_runs,
    COUNT(*) FILTER (WHERE succeeded)                                   AS successful,
    ROUND(100.0 * COUNT(*) FILTER (WHERE succeeded) / COUNT(*), 2)      AS success_rate_pct,
    ROUND(AVG(duration_seconds), 2)                                     AS mean_duration_s,
    SUM(records_written)                                                AS lifetime_records_written,
    SUM(records_quarantined)                                            AS lifetime_quarantined,
    MAX(start_time)                                                     AS last_run_at
FROM gold_pipeline_runs
GROUP BY pipeline_name
ORDER BY total_runs DESC;

-- name: data_freshness_by_layer
-- How stale is each layer? A Silver table that stopped updating while Bronze
-- kept growing is a broken transformation, and this is where that shows.
SELECT
    'silver_transactions'   AS dataset,
    MAX(transaction_timestamp)  AS latest_event_time,
    MAX(_processed_timestamp)   AS last_processed_at,
    COUNT(*)                    AS records
FROM silver_transactions
UNION ALL
SELECT
    'gold_anomaly_transactions',
    MAX(transaction_timestamp),
    MAX(scored_at),
    COUNT(*)
FROM gold_anomaly_transactions;
