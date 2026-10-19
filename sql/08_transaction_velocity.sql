-- ===========================================================================
-- 08 - Transaction velocity
--
-- Rate-of-activity questions. Velocity is the signal that distinguishes a large
-- payment (normal) from a burst of payments (rarely normal).
-- ===========================================================================

-- name: highest_velocity_bursts
-- Account-hours with the most extreme five-minute burst.
SELECT
    account_id,
    transaction_hour,
    transaction_count,
    peak_velocity_5min,
    ROUND(total_amount_usd, 2)  AS total_usd,
    ROUND(max_amount_usd, 2)    AS largest_usd,
    unique_vendors,
    anomaly_count,
    high_risk_count,
    max_risk_score
FROM gold_transaction_velocity
WHERE is_burst
ORDER BY peak_velocity_5min DESC, total_usd DESC
LIMIT 40;

-- name: hourly_activity_profile
-- Activity by hour of day. The overnight band is where the unusual-hour rule
-- operates, and this shows how much legitimate traffic sits there.
SELECT
    hour_of_day,
    SUM(transaction_count)                                          AS transactions,
    ROUND(SUM(total_amount_usd), 2)                                 AS total_usd,
    SUM(anomaly_count)                                              AS anomalies,
    ROUND(100.0 * SUM(anomaly_count) / NULLIF(SUM(transaction_count), 0), 3) AS anomaly_rate_pct,
    CASE WHEN hour_of_day BETWEEN 0 AND 4 THEN 'overnight'
         WHEN hour_of_day BETWEEN 8 AND 18 THEN 'business hours'
         ELSE 'off hours' END                                       AS window_label
FROM gold_transaction_velocity
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- name: accounts_with_sustained_high_velocity
-- Not one burst, but repeated bursts - a pattern rather than an incident.
SELECT
    account_id,
    COUNT(*)                            AS burst_hours,
    MAX(peak_velocity_5min)             AS worst_burst,
    ROUND(SUM(total_amount_usd), 2)     AS total_in_bursts_usd,
    SUM(anomaly_count)                  AS anomalies,
    MIN(transaction_hour)               AS first_burst,
    MAX(transaction_hour)               AS last_burst
FROM gold_transaction_velocity
WHERE is_burst
GROUP BY account_id
HAVING COUNT(*) >= 2
ORDER BY burst_hours DESC, worst_burst DESC
LIMIT 30;

-- name: velocity_versus_risk
-- Does higher velocity actually correspond to higher risk? A sanity check on
-- the rule engine's own premise - if these were uncorrelated, the velocity rule
-- would be contributing noise.
SELECT
    CASE
        WHEN peak_velocity_5min = 0 THEN '0'
        WHEN peak_velocity_5min <= 2 THEN '1-2'
        WHEN peak_velocity_5min <= 5 THEN '3-5'
        WHEN peak_velocity_5min <= 10 THEN '6-10'
        ELSE '10+'
    END                                                     AS velocity_band,
    COUNT(*)                                                AS account_hours,
    SUM(transaction_count)                                  AS transactions,
    ROUND(AVG(max_risk_score), 1)                           AS mean_max_risk_score,
    SUM(anomaly_count)                                      AS anomalies,
    ROUND(100.0 * SUM(anomaly_count) / NULLIF(SUM(transaction_count), 0), 3) AS anomaly_rate_pct
FROM gold_transaction_velocity
GROUP BY velocity_band
ORDER BY
    CASE velocity_band WHEN '0' THEN 1 WHEN '1-2' THEN 2 WHEN '3-5' THEN 3
    WHEN '6-10' THEN 4 ELSE 5 END;
