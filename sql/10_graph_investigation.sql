-- ===========================================================================
-- 10 - Relationship investigation in SQL
--
-- The same relationship questions the Neo4j layer answers with Cypher, written
-- as SQL self-joins. Both exist on purpose:
--
--   * SQL is always available - no extra service, no extra dependency.
--   * Cypher expresses variable-depth traversal ("anything within three hops of
--     this vendor") in a line, where SQL needs a recursive CTE that nobody
--     enjoys reading or tuning.
--
-- The rule of thumb this project follows: fixed-depth relationship queries stay
-- in SQL; variable-depth traversal and path-finding go to the graph. Anyone who
-- claims one tool replaces the other has not tried the other one's use case.
-- ===========================================================================

-- name: accounts_sharing_vendors
-- Pairs of accounts paying the same vendor - the SQL form of the classic
-- "two accounts, one supplier" graph query. The a1 < a2 predicate stops each
-- pair appearing twice and stops an account pairing with itself.
SELECT
    t1.account_id                       AS account_a,
    t2.account_id                       AS account_b,
    COUNT(DISTINCT t1.vendor_id)        AS shared_vendors,
    ROUND(SUM(t1.amount_usd), 2)        AS account_a_paid_usd,
    ROUND(SUM(t2.amount_usd), 2)        AS account_b_paid_usd
FROM silver_transactions t1
JOIN silver_transactions t2
  ON t1.vendor_id = t2.vendor_id
 AND t1.account_id < t2.account_id
GROUP BY account_a, account_b
HAVING COUNT(DISTINCT t1.vendor_id) >= 3
ORDER BY shared_vendors DESC, account_a_paid_usd DESC
LIMIT 30;

-- name: vendors_linked_to_multiple_high_risk_accounts
-- The headline fraud-network question: which suppliers sit at the centre of
-- several accounts that are independently flagged as risky.
SELECT
    t.vendor_id,
    v.vendor_name,
    COUNT(DISTINCT t.account_id)                        AS high_risk_accounts,
    COUNT(*)                                            AS transactions,
    ROUND(SUM(t.amount_usd), 2)                         AS total_usd,
    ROUND(AVG(a.risk_score), 1)                         AS mean_account_risk,
    v.risk_level                                        AS vendor_risk_level
FROM silver_transactions t
JOIN gold_account_risk a  ON a.account_id = t.account_id
LEFT JOIN gold_vendor_risk v ON v.vendor_id = t.vendor_id
WHERE a.risk_level IN ('HIGH', 'CRITICAL')
GROUP BY t.vendor_id, v.vendor_name, v.risk_level
HAVING COUNT(DISTINCT t.account_id) >= 2
ORDER BY high_risk_accounts DESC, total_usd DESC
LIMIT 30;

-- name: agency_vendor_network
-- Which vendors are paid by several agencies. Cross-agency suppliers are not
-- inherently suspicious, but they are where a single relationship has the
-- widest blast radius.
SELECT
    t.vendor_id,
    v.vendor_name,
    COUNT(DISTINCT t.agency_id)                     AS agencies_served,
    STRING_AGG(DISTINCT t.agency_id, ', ')          AS agency_ids,
    COUNT(DISTINCT t.account_id)                    AS paying_accounts,
    COUNT(*)                                        AS transactions,
    ROUND(SUM(t.amount_usd), 2)                     AS total_usd
FROM silver_transactions t
LEFT JOIN silver_vendors v USING (vendor_id)
WHERE t.agency_id IS NOT NULL
GROUP BY t.vendor_id, v.vendor_name
HAVING COUNT(DISTINCT t.agency_id) >= 3
ORDER BY agencies_served DESC, total_usd DESC
LIMIT 30;

-- name: suspicious_transaction_clusters
-- Account-vendor pairs where flagged activity concentrates. This is the seed
-- list an investigator would then expand in the graph.
SELECT
    a.account_id,
    a.vendor_id,
    v.vendor_name,
    COUNT(*)                                    AS flagged_transactions,
    ROUND(SUM(a.amount_usd), 2)                 AS flagged_usd,
    ROUND(AVG(a.risk_score), 1)                 AS mean_risk_score,
    MAX(a.risk_level)                           AS worst_level,
    MIN(a.transaction_timestamp)                AS first_flagged,
    MAX(a.transaction_timestamp)                AS last_flagged
FROM gold_anomaly_transactions a
LEFT JOIN silver_vendors v USING (vendor_id)
GROUP BY a.account_id, a.vendor_id, v.vendor_name
HAVING COUNT(*) >= 2
ORDER BY flagged_transactions DESC, flagged_usd DESC
LIMIT 30;

-- name: contract_vendor_agency_paths
-- The Agency -> Contract -> Vendor -> Transaction path, flattened. This is the
-- exact structure the graph loader materialises as nodes and edges; seeing it
-- both ways makes the modelling choice concrete rather than theoretical.
SELECT
    c.agency_id,
    c.contract_id,
    c.competition_type,
    ROUND(c.contract_value_usd, 2)      AS awarded_usd,
    c.vendor_id,
    v.vendor_name,
    COUNT(t.transaction_id)             AS payments,
    ROUND(SUM(t.amount_usd), 2)         AS paid_usd,
    ROUND(100.0 * SUM(t.amount_usd) / NULLIF(c.contract_value_usd, 0), 1) AS paid_pct_of_award
FROM silver_contracts c
LEFT JOIN silver_vendors v USING (vendor_id)
LEFT JOIN silver_transactions t ON t.contract_id = c.contract_id
GROUP BY c.agency_id, c.contract_id, c.competition_type, c.contract_value_usd, c.vendor_id, v.vendor_name
HAVING COUNT(t.transaction_id) > 0
ORDER BY paid_usd DESC
LIMIT 30;
