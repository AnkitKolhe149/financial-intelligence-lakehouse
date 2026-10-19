-- ===========================================================================
-- 02 - Account risk
--
-- "Which accounts show unusual spending?" - asked four different ways, because
-- "unusual" means different things to an auditor, a controller and an analyst.
-- ===========================================================================

-- name: highest_risk_accounts
-- The ranked review list, with enough context to triage without a second query.
SELECT
    account_id,
    account_type,
    region,
    transaction_count,
    ROUND(total_amount_usd, 2)      AS total_usd,
    ROUND(average_amount_usd, 2)    AS average_usd,
    ROUND(maximum_amount_usd, 2)    AS largest_usd,
    unique_vendors,
    unique_locations,
    anomaly_count,
    high_risk_count,
    risk_score,
    risk_level
FROM gold_account_risk
WHERE risk_level IN ('HIGH', 'CRITICAL')
ORDER BY risk_score DESC, total_usd DESC
LIMIT 50;

-- name: accounts_spending_far_above_peers
-- Outliers *relative to their own account type*. Comparing a payroll account to
-- a travel account would flag every payroll account and find nothing; the
-- window partition is what makes the comparison fair.
WITH peer_stats AS (
    SELECT
        account_type,
        AVG(total_amount_usd)     AS peer_average,
        STDDEV(total_amount_usd)  AS peer_stddev
    FROM gold_account_risk
    WHERE account_type IS NOT NULL
    GROUP BY account_type
)
SELECT
    r.account_id,
    r.account_type,
    ROUND(r.total_amount_usd, 2)                AS total_usd,
    ROUND(p.peer_average, 2)                    AS peer_average_usd,
    ROUND(
        (r.total_amount_usd - p.peer_average) / NULLIF(p.peer_stddev, 0), 2
    )                                           AS z_score_vs_peers,
    r.transaction_count,
    r.risk_level
FROM gold_account_risk r
JOIN peer_stats p USING (account_type)
WHERE p.peer_stddev > 0
  AND (r.total_amount_usd - p.peer_average) / p.peer_stddev > 2.5
ORDER BY z_score_vs_peers DESC
LIMIT 40;

-- name: dormant_accounts_suddenly_active
-- An account that was quiet and then was not. The 30-day windows are computed
-- from the data's own maximum date, so the query works on a static demo dataset
-- as well as on a live one.
WITH bounds AS (
    SELECT MAX(CAST(transaction_date AS DATE)) AS latest FROM silver_transactions
),
windows AS (
    SELECT
        t.account_id,
        COUNT(*) FILTER (
            WHERE CAST(t.transaction_date AS DATE) > b.latest - INTERVAL 30 DAY
        ) AS recent_30d,
        COUNT(*) FILTER (
            WHERE CAST(t.transaction_date AS DATE) <= b.latest - INTERVAL 30 DAY
              AND CAST(t.transaction_date AS DATE) >  b.latest - INTERVAL 120 DAY
        ) AS prior_90d,
        ROUND(SUM(t.amount_usd) FILTER (
            WHERE CAST(t.transaction_date AS DATE) > b.latest - INTERVAL 30 DAY
        ), 2) AS recent_usd
    FROM silver_transactions t
    CROSS JOIN bounds b
    GROUP BY t.account_id
)
SELECT
    account_id,
    recent_30d,
    prior_90d,
    recent_usd,
    ROUND(recent_30d / NULLIF(prior_90d / 3.0, 0), 2) AS activity_multiple
FROM windows
WHERE recent_30d >= 5
  AND (prior_90d = 0 OR recent_30d > 3 * (prior_90d / 3.0))
ORDER BY activity_multiple DESC NULLS LAST, recent_30d DESC
LIMIT 30;

-- name: account_risk_distribution
-- How the portfolio is distributed across risk bands. The share column is what
-- turns "200 high-risk accounts" into a number with a denominator.
SELECT
    risk_level,
    COUNT(*)                                                AS accounts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)      AS share_pct,
    ROUND(SUM(total_amount_usd), 2)                         AS total_usd,
    ROUND(100.0 * SUM(total_amount_usd) / SUM(SUM(total_amount_usd)) OVER (), 2)
                                                            AS value_share_pct,
    SUM(anomaly_count)                                      AS anomalies
FROM gold_account_risk
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2
        WHEN 'MEDIUM' THEN 3 ELSE 4
    END;
