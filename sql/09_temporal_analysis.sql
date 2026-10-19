-- ===========================================================================
-- 09 - Temporal analysis
--
-- Seasonality, late arrivals and period-over-period comparison. These queries
-- exist partly to prove the platform handles event time correctly - a system
-- that only knows processing time cannot answer any of them.
-- ===========================================================================

-- name: fiscal_year_end_effect
-- US federal spending spikes in September as agencies obligate remaining
-- budget. Confirming the effect matters because the volume quality check must
-- tolerate it rather than flag it every year.
SELECT
    MONTH(CAST(transaction_date AS DATE))                       AS month_number,
    MONTHNAME(CAST(transaction_date AS DATE))                   AS month_name,
    SUM(transaction_count)                                      AS transactions,
    ROUND(SUM(total_amount_usd), 2)                             AS total_usd,
    ROUND(AVG(transaction_count), 1)                            AS mean_daily_volume,
    ROUND(
        100.0 * AVG(transaction_count) / (SELECT AVG(transaction_count) FROM gold_daily_financial_summary) - 100,
        1
    )                                                           AS pct_vs_overall_daily_mean
FROM gold_daily_financial_summary
GROUP BY month_number, month_name
ORDER BY month_number;

-- name: day_of_week_profile
-- Weekday vs weekend behaviour. The weekend rule adds risk points; this is the
-- evidence for how unusual weekend activity actually is.
SELECT
    DAYNAME(CAST(transaction_date AS DATE))                     AS day_name,
    DAYOFWEEK(CAST(transaction_date AS DATE))                   AS day_number,
    COUNT(*)                                                    AS days_observed,
    SUM(transaction_count)                                      AS transactions,
    ROUND(AVG(transaction_count), 1)                            AS mean_daily_volume,
    ROUND(SUM(total_amount_usd), 2)                             AS total_usd,
    SUM(anomaly_count)                                          AS anomalies
FROM gold_daily_financial_summary
GROUP BY day_name, day_number
ORDER BY day_number;

-- name: late_arriving_transactions
-- Records whose posting lagged their occurrence. Late data is normal; measuring
-- it is what lets the streaming watermark be set from evidence rather than a
-- guess.
SELECT
    CASE
        WHEN posted_timestamp IS NULL THEN 'not posted'
        WHEN DATE_DIFF('hour', transaction_timestamp, posted_timestamp) < 1  THEN 'under 1h'
        WHEN DATE_DIFF('hour', transaction_timestamp, posted_timestamp) < 24 THEN '1-24h'
        WHEN DATE_DIFF('hour', transaction_timestamp, posted_timestamp) < 72 THEN '1-3 days'
        ELSE 'over 3 days'
    END                                                     AS posting_lag_band,
    COUNT(*)                                                AS transactions,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)      AS share_pct,
    ROUND(SUM(amount_usd), 2)                               AS total_usd,
    ROUND(AVG(DATE_DIFF('minute', transaction_timestamp, posted_timestamp)), 1) AS mean_lag_minutes
FROM silver_transactions
GROUP BY posting_lag_band
ORDER BY transactions DESC;

-- name: quarter_over_quarter
-- Fiscal-quarter comparison with the prior quarter carried alongside.
WITH quarterly AS (
    SELECT
        fiscal_year,
        CASE
            WHEN MONTH(transaction_timestamp) BETWEEN 10 AND 12 THEN 'Q1'
            WHEN MONTH(transaction_timestamp) BETWEEN 1 AND 3   THEN 'Q2'
            WHEN MONTH(transaction_timestamp) BETWEEN 4 AND 6   THEN 'Q3'
            ELSE 'Q4'
        END                             AS fiscal_quarter,
        COUNT(*)                        AS transactions,
        SUM(amount_usd)                 AS total_usd,
        COUNT(DISTINCT vendor_id)       AS vendors
    FROM silver_transactions
    GROUP BY fiscal_year, fiscal_quarter
)
SELECT
    fiscal_year,
    fiscal_quarter,
    transactions,
    ROUND(total_usd, 2)                                                     AS total_usd,
    vendors,
    ROUND(LAG(total_usd) OVER (ORDER BY fiscal_year, fiscal_quarter), 2)    AS previous_quarter_usd,
    ROUND(
        100.0 * (total_usd - LAG(total_usd) OVER (ORDER BY fiscal_year, fiscal_quarter))
        / NULLIF(LAG(total_usd) OVER (ORDER BY fiscal_year, fiscal_quarter), 0), 1
    )                                                                       AS change_pct
FROM quarterly
ORDER BY fiscal_year, fiscal_quarter;

-- name: first_and_last_activity_per_vendor
-- Vendor lifecycle: when a supplier appeared, when it was last paid, and how
-- long its relationship has run. Short-lived, high-value vendors are worth a look.
SELECT
    vendor_id,
    COUNT(*)                                                                AS transactions,
    MIN(transaction_timestamp)                                              AS first_paid,
    MAX(transaction_timestamp)                                              AS last_paid,
    DATE_DIFF('day', MIN(transaction_timestamp), MAX(transaction_timestamp)) AS relationship_days,
    ROUND(SUM(amount_usd), 2)                                               AS total_usd,
    ROUND(SUM(amount_usd) / NULLIF(DATE_DIFF('day', MIN(transaction_timestamp), MAX(transaction_timestamp)), 0), 2)
                                                                            AS usd_per_day
FROM silver_transactions
GROUP BY vendor_id
HAVING COUNT(*) >= 5
ORDER BY usd_per_day DESC NULLS LAST
LIMIT 30;
