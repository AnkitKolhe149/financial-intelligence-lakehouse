-- ===========================================================================
-- 01 - Financial overview
--
-- Executive-level questions: how much moved, through whom, and how did that
-- compare to the recent baseline.
--
-- Run with:  python -m src.analytics.run_sql --file sql/01_financial_overview.sql
-- ===========================================================================

-- name: platform_totals
-- Headline figures for the whole platform. Reads the pre-aggregated daily
-- summary rather than the fact table: this powers a dashboard tile that must
-- render instantly, and re-scanning millions of rows to produce six numbers is
-- the most common self-inflicted performance problem in a lakehouse.
SELECT
    COUNT(*)                                    AS days_covered,
    SUM(transaction_count)                      AS total_transactions,
    ROUND(SUM(total_amount_usd), 2)             AS total_value_usd,
    ROUND(AVG(average_transaction_usd), 2)      AS mean_transaction_usd,
    ROUND(MEDIAN(median_transaction_usd), 2)    AS typical_transaction_usd,
    SUM(anomaly_count)                          AS total_anomalies,
    SUM(high_risk_count)                        AS total_high_risk,
    SUM(critical_risk_count)                    AS total_critical,
    ROUND(100.0 * SUM(anomaly_count) / NULLIF(SUM(transaction_count), 0), 3) AS anomaly_rate_pct,
    MIN(transaction_date)                       AS first_date,
    MAX(transaction_date)                       AS last_date
FROM gold_daily_financial_summary;

-- name: monthly_spending_trend
-- Month-over-month movement with the previous month carried alongside, so the
-- change is a column rather than something the reader computes by eye.
SELECT
    STRFTIME(CAST(transaction_date AS DATE), '%Y-%m')          AS month,
    SUM(transaction_count)                                     AS transactions,
    ROUND(SUM(total_amount_usd), 2)                            AS total_usd,
    ROUND(LAG(SUM(total_amount_usd)) OVER (ORDER BY STRFTIME(CAST(transaction_date AS DATE), '%Y-%m')), 2)
                                                               AS previous_month_usd,
    ROUND(
        100.0 * (SUM(total_amount_usd) - LAG(SUM(total_amount_usd)) OVER (ORDER BY STRFTIME(CAST(transaction_date AS DATE), '%Y-%m')))
        / NULLIF(LAG(SUM(total_amount_usd)) OVER (ORDER BY STRFTIME(CAST(transaction_date AS DATE), '%Y-%m')), 0),
        2
    )                                                          AS change_pct,
    SUM(anomaly_count)                                         AS anomalies
FROM gold_daily_financial_summary
GROUP BY 1
ORDER BY 1;

-- name: largest_transactions
-- The single largest payments, with their risk assessment attached. "Big" and
-- "suspicious" are different questions, and putting them side by side is what
-- lets a reviewer tell a legitimate capital payment from a problem.
SELECT
    t.transaction_id,
    t.transaction_date,
    t.account_id,
    t.vendor_id,
    t.agency_id,
    ROUND(t.amount_usd, 2)      AS amount_usd,
    t.currency,
    COALESCE(a.risk_score, 0)   AS risk_score,
    COALESCE(a.risk_level, 'LOW') AS risk_level
FROM silver_transactions t
LEFT JOIN gold_anomaly_transactions a USING (transaction_id)
ORDER BY t.amount_usd DESC
LIMIT 25;

-- name: currency_mix
-- Currency exposure. Everything is reported in USD downstream, so this is the
-- check that the conversion is actually load-bearing rather than decorative.
SELECT
    currency,
    COUNT(*)                                                   AS transactions,
    ROUND(SUM(amount), 2)                                      AS total_native,
    ROUND(SUM(amount_usd), 2)                                  AS total_usd,
    ROUND(100.0 * SUM(amount_usd) / SUM(SUM(amount_usd)) OVER (), 2) AS share_of_usd_pct
FROM silver_transactions
GROUP BY currency
ORDER BY total_usd DESC;

-- name: payment_method_profile
-- Method mix by value. Wires are few and large, cards are many and small;
-- a shift in that shape is an operational signal.
SELECT
    COALESCE(payment_method, 'UNKNOWN')     AS payment_method,
    COUNT(*)                                AS transactions,
    ROUND(SUM(amount_usd), 2)               AS total_usd,
    ROUND(AVG(amount_usd), 2)               AS average_usd,
    ROUND(MEDIAN(amount_usd), 2)            AS median_usd,
    ROUND(MAX(amount_usd), 2)               AS largest_usd
FROM silver_transactions
GROUP BY 1
ORDER BY total_usd DESC;
