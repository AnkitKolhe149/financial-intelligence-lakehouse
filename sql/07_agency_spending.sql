-- ===========================================================================
-- 07 - Agency and department spending
--
-- Where the money goes, against what it was allocated.
-- ===========================================================================

-- name: top_spending_agencies
SELECT
    agency_id,
    agency_name,
    fiscal_year,
    transaction_count,
    ROUND(total_amount_usd, 2)      AS total_usd,
    unique_vendors,
    unique_departments,
    ROUND(budget_utilisation_pct, 2) AS budget_used_pct,
    ROUND(top_vendor_share_pct, 2)   AS top_vendor_share_pct,
    ROUND(anomaly_rate_pct, 2)       AS anomaly_rate_pct
FROM gold_agency_spending
ORDER BY total_usd DESC
LIMIT 25;

-- name: departments_over_budget
-- Overruns, largest first. The generator deliberately produces a minority of
-- over-spending departments so this branch is genuinely exercised.
SELECT
    department_id,
    department_name,
    agency_id,
    fiscal_year,
    ROUND(total_amount_usd, 2)                              AS spent_usd,
    ROUND(allocated_amount_usd, 2)                          AS allocated_usd,
    ROUND(total_amount_usd - allocated_amount_usd, 2)       AS overrun_usd,
    ROUND(budget_utilisation_pct, 1)                        AS used_pct,
    transaction_count,
    high_risk_count
FROM gold_department_spending
WHERE is_over_budget
ORDER BY overrun_usd DESC
LIMIT 30;

-- name: budget_utilisation_bands
-- Portfolio view of utilisation. Under-spending is a finding too: an agency at
-- 20% of allocation either did not need the money or could not spend it.
SELECT
    CASE
        WHEN budget_utilisation_pct >= 100 THEN 'over budget'
        WHEN budget_utilisation_pct >= 85  THEN '85-100%'
        WHEN budget_utilisation_pct >= 60  THEN '60-85%'
        WHEN budget_utilisation_pct >= 30  THEN '30-60%'
        ELSE 'under 30%'
    END                                     AS utilisation_band,
    COUNT(*)                                AS department_years,
    ROUND(SUM(total_amount_usd), 2)         AS spent_usd,
    ROUND(SUM(allocated_amount_usd), 2)     AS allocated_usd
FROM gold_department_spending
WHERE allocated_amount_usd > 0
GROUP BY utilisation_band
ORDER BY
    CASE utilisation_band
        WHEN 'over budget' THEN 1 WHEN '85-100%' THEN 2 WHEN '60-85%' THEN 3
        WHEN '30-60%' THEN 4 ELSE 5
    END;

-- name: spending_by_region
-- Geographic distribution, joined through the location dimension.
SELECT
    COALESCE(l.region, 'UNKNOWN')       AS region,
    COALESCE(l.state, 'UNKNOWN')        AS state,
    COUNT(*)                            AS transactions,
    ROUND(SUM(t.amount_usd), 2)         AS total_usd,
    ROUND(AVG(t.amount_usd), 2)         AS average_usd,
    COUNT(DISTINCT t.vendor_id)         AS vendors,
    COUNT(DISTINCT t.agency_id)         AS agencies
FROM silver_transactions t
LEFT JOIN silver_locations l USING (location_id)
GROUP BY region, state
ORDER BY total_usd DESC
LIMIT 40;

-- name: agency_vendor_dependency
-- Agencies whose spend is concentrated in a single supplier. High
-- concentration is a continuity risk as much as an integrity one.
SELECT
    agency_id,
    agency_name,
    fiscal_year,
    unique_vendors,
    top_vendor_id,
    ROUND(top_vendor_amount_usd, 2)     AS top_vendor_usd,
    ROUND(top_vendor_share_pct, 1)      AS top_vendor_share_pct,
    ROUND(total_amount_usd, 2)          AS total_usd
FROM gold_agency_spending
WHERE top_vendor_share_pct > 25
  AND transaction_count >= 20
ORDER BY top_vendor_share_pct DESC
LIMIT 25;

-- name: spending_by_category
-- What the money is bought for, from the merchant category added in schema
-- v1.1. Nulls are the v1.0 records that predate the column.
SELECT
    COALESCE(merchant_category, '(not supplied - schema v1.0)') AS category,
    COUNT(*)                        AS transactions,
    ROUND(SUM(amount_usd), 2)       AS total_usd,
    ROUND(AVG(amount_usd), 2)       AS average_usd
FROM silver_transactions
GROUP BY category
ORDER BY total_usd DESC;
