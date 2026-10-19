-- ===========================================================================
-- 03 - Vendor risk and concentration
--
-- Procurement oversight questions: who receives the money, is it competed, is
-- any single supplier becoming load-bearing, and are we paying the same company
-- twice under two names.
-- ===========================================================================

-- name: top_vendors_by_value
-- Ranked recipients with a running cumulative share - the standard way to see
-- concentration. If ten vendors hold 60% of spend, that is visible here.
SELECT
    vendor_id,
    vendor_name,
    transaction_count,
    ROUND(total_amount_usd, 2)                                          AS total_usd,
    unique_accounts,
    unique_agencies,
    contract_count,
    ROUND(100.0 * total_amount_usd / SUM(total_amount_usd) OVER (), 2)  AS share_pct,
    ROUND(100.0 * SUM(total_amount_usd) OVER (
        ORDER BY total_amount_usd DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / SUM(total_amount_usd) OVER (), 2)                               AS cumulative_share_pct,
    risk_level
FROM gold_vendor_risk
ORDER BY total_amount_usd DESC
LIMIT 30;

-- name: fastest_growing_vendors
-- Growth measured against each vendor's own prior rate, not against other
-- vendors - a small supplier tripling in volume is the finding, and ranking by
-- absolute value would bury it beneath the incumbents.
SELECT
    vendor_id,
    vendor_name,
    transaction_count,
    ROUND(recent_30d_amount_usd, 2)     AS last_30d_usd,
    ROUND(total_amount_usd, 2)          AS lifetime_usd,
    growth_ratio,
    unique_accounts,
    contract_count,
    risk_level
FROM gold_vendor_risk
WHERE growth_ratio > 2.0
  AND recent_30d_amount_usd > 0
ORDER BY growth_ratio DESC, last_30d_usd DESC
LIMIT 30;

-- name: payments_without_contracts
-- Money leaving without an award behind it. A direct audit finding, and the
-- reason contract_id is nullable-but-tracked rather than dropped.
SELECT
    vendor_id,
    vendor_name,
    transaction_count,
    ROUND(total_amount_usd, 2)      AS total_usd,
    contract_count,
    uncontracted_payment_pct,
    unique_agencies,
    risk_level
FROM gold_vendor_risk
WHERE uncontracted_payment_pct > 50
  AND transaction_count >= 10
ORDER BY total_usd DESC
LIMIT 40;

-- name: vendors_paid_over_contract_ceiling
-- Cumulative payments exceeding the awarded value.
SELECT
    vendor_id,
    vendor_name,
    contract_count,
    ROUND(contract_value_usd, 2)                                        AS awarded_usd,
    ROUND(total_amount_usd, 2)                                          AS paid_usd,
    ROUND(total_amount_usd - contract_value_usd, 2)                     AS overrun_usd,
    ROUND(100.0 * total_amount_usd / NULLIF(contract_value_usd, 0), 1)  AS paid_pct_of_award
FROM gold_vendor_risk
WHERE payments_over_contract_value
ORDER BY overrun_usd DESC
LIMIT 30;

-- name: duplicate_vendor_registrations
-- The same tax id registered under multiple vendor ids: an entity-resolution
-- problem surfaced for a steward, never auto-merged. Merging master data on a
-- fuzzy match destroys the audit trail.
SELECT
    tax_id,
    COUNT(DISTINCT vendor_id)                       AS registrations,
    STRING_AGG(DISTINCT vendor_id, ', ')            AS vendor_ids,
    STRING_AGG(DISTINCT vendor_name, ' | ')         AS names_used,
    ROUND(SUM(total_amount_usd), 2)                 AS combined_paid_usd
FROM gold_vendor_risk
WHERE tax_id IS NOT NULL
GROUP BY tax_id
HAVING COUNT(DISTINCT vendor_id) > 1
ORDER BY combined_paid_usd DESC
LIMIT 30;

-- name: sole_source_concentration
-- Agencies awarding disproportionately without competition.
SELECT
    c.agency_id,
    COUNT(*)                                                                    AS contracts,
    COUNT(*) FILTER (WHERE c.competition_type = 'SOLE_SOURCE')                  AS sole_source,
    ROUND(100.0 * COUNT(*) FILTER (WHERE c.competition_type = 'SOLE_SOURCE') / COUNT(*), 1)
                                                                                AS sole_source_pct,
    ROUND(SUM(c.contract_value_usd), 2)                                         AS awarded_usd,
    ROUND(SUM(c.contract_value_usd) FILTER (WHERE c.competition_type = 'SOLE_SOURCE'), 2)
                                                                                AS sole_source_usd
FROM silver_contracts c
GROUP BY c.agency_id
HAVING COUNT(*) >= 5
ORDER BY sole_source_pct DESC, awarded_usd DESC
LIMIT 25;
