-- ===========================================================================
-- 04 - Anomaly and fraud analytics
--
-- What the detection layer found, how the model and the rules agree, and where
-- the alerts are concentrated.
-- ===========================================================================

-- name: critical_alerts_queue
-- The review queue, highest first. This is the operational output of the whole
-- platform - everything upstream exists to make this list trustworthy.
SELECT
    alert_id,
    transaction_id,
    account_id,
    vendor_id,
    agency_id,
    ROUND(amount_usd, 2)    AS amount_usd,
    risk_score,
    risk_level,
    reason_count,
    reason,
    status,
    CAST(transaction_timestamp AS DATE) AS occurred_on
FROM gold_risk_alerts
WHERE risk_level IN ('HIGH', 'CRITICAL')
ORDER BY risk_score DESC, amount_usd DESC
LIMIT 50;

-- name: daily_anomaly_rate
-- Anomaly rate over time against a 7-day trailing baseline. A rate without a
-- baseline is a number nobody can act on.
SELECT
    transaction_date,
    transaction_count,
    anomaly_count,
    anomaly_rate_pct,
    high_risk_count,
    critical_risk_count,
    ROUND(AVG(anomaly_rate_pct) OVER (
        ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 3)                                       AS rolling_7d_anomaly_rate,
    ROUND(anomaly_rate_pct - AVG(anomaly_rate_pct) OVER (
        ORDER BY transaction_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ), 3)                                       AS deviation_from_baseline
FROM gold_daily_financial_summary
WHERE transaction_count > 0
ORDER BY transaction_date DESC
LIMIT 60;

-- name: risk_rule_effectiveness
-- Which rules actually fire, and how much value they touch. A rule that never
-- fires is dead configuration; one that fires on everything is noise. Both
-- show up here.
SELECT
    rule_hit                                                    AS rule_name,
    COUNT(*)                                                    AS transactions_flagged,
    ROUND(AVG(risk_score), 1)                                   AS mean_risk_score,
    ROUND(SUM(amount_usd), 2)                                   AS value_flagged_usd,
    COUNT(*) FILTER (WHERE risk_level = 'CRITICAL')             AS critical,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM gold_anomaly_transactions), 2)
                                                                AS pct_of_flagged
FROM (
    SELECT
        UNNEST(CAST(risk_rule_hits AS VARCHAR[])) AS rule_hit,
        risk_score,
        amount_usd,
        risk_level
    FROM gold_anomaly_transactions
    WHERE risk_rule_hits IS NOT NULL
)
GROUP BY rule_name
ORDER BY transactions_flagged DESC;

-- name: model_versus_rules_agreement
-- Where the statistical model and the deterministic rules agree and disagree.
-- The ML-only population is where genuinely novel patterns would appear - or
-- where the model has drifted. Either way it is the population to review.
SELECT
    CASE
        WHEN is_anomaly AND risk_level IN ('HIGH', 'CRITICAL') THEN 'both'
        WHEN is_anomaly                                        THEN 'model_only'
        WHEN risk_level IN ('HIGH', 'CRITICAL')                THEN 'rules_only'
        ELSE 'neither'
    END                                     AS detected_by,
    COUNT(*)                                AS transactions,
    ROUND(AVG(anomaly_score), 4)            AS mean_anomaly_score,
    ROUND(AVG(risk_score), 1)               AS mean_risk_score,
    ROUND(SUM(amount_usd), 2)               AS total_usd
FROM gold_anomaly_transactions
GROUP BY detected_by
ORDER BY transactions DESC;

-- name: anomaly_concentration_by_vendor
-- Vendors where anomalies cluster. Concentration is the signal: one anomalous
-- payment is noise, thirty to the same vendor is a pattern.
SELECT
    a.vendor_id,
    v.vendor_name,
    COUNT(*)                                                        AS flagged_transactions,
    ROUND(SUM(a.amount_usd), 2)                                     AS flagged_usd,
    ROUND(AVG(a.risk_score), 1)                                     AS mean_risk_score,
    COUNT(*) FILTER (WHERE a.risk_level = 'CRITICAL')               AS critical,
    COUNT(DISTINCT a.account_id)                                    AS distinct_accounts,
    v.risk_level                                                    AS vendor_risk_level
FROM gold_anomaly_transactions a
LEFT JOIN gold_vendor_risk v USING (vendor_id)
GROUP BY a.vendor_id, v.vendor_name, v.risk_level
HAVING COUNT(*) >= 3
ORDER BY flagged_transactions DESC, flagged_usd DESC
LIMIT 30;

-- name: anomaly_amount_distribution
-- Are flagged transactions actually larger, or is the detector just finding
-- odd-looking small ones? Compares the flagged population against the rest.
SELECT
    CASE WHEN a.transaction_id IS NOT NULL THEN 'flagged' ELSE 'not_flagged' END AS population,
    COUNT(*)                        AS transactions,
    ROUND(AVG(t.amount_usd), 2)     AS mean_usd,
    ROUND(MEDIAN(t.amount_usd), 2)  AS median_usd,
    ROUND(QUANTILE_CONT(t.amount_usd, 0.95), 2) AS p95_usd,
    ROUND(MAX(t.amount_usd), 2)     AS max_usd
FROM silver_transactions t
LEFT JOIN gold_anomaly_transactions a USING (transaction_id)
GROUP BY population;
