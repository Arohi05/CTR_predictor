-- CTR campaign analysis: SQL queries (SQLite dialect; table `ads` is loaded from Dataset.csv)
-- Every query is executed by analysis/ctr_analysis.ipynb and its output saved to analysis/results/.
-- Valid rows = impressions > 0 AND clicks/impressions <= 0.20 (higher values are reporting glitches).
-- Use SUM(clicks)/SUM(impressions), not AVG of row-level CTR: AVG over-weights tiny rows.

-- name: q01_data_quality
-- How many rows are unusable, and why?
SELECT COUNT(*)                                                   AS total_rows,
       SUM(CASE WHEN impressions = 0 THEN 1 ELSE 0 END)           AS zero_impressions,
       SUM(CASE WHEN impressions > 0 AND 1.0*clicks/impressions > 0.20 THEN 1 ELSE 0 END) AS ctr_over_20pct,
       SUM(CASE WHEN creative_width IS NULL THEN 1 ELSE 0 END)    AS missing_creative_size,
       SUM(CASE WHEN max_bid_cpm IS NULL THEN 1 ELSE 0 END)       AS missing_max_bid
FROM ads;

-- name: q02_overall_ctr
-- Weighted vs simple-average CTR: the two differ a lot.
SELECT COUNT(*)                                         AS valid_rows,
       SUM(impressions)                                 AS impressions,
       SUM(clicks)                                      AS clicks,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3)     AS weighted_ctr_pct,
       ROUND(100.0*AVG(1.0*clicks/impressions), 3)      AS simple_avg_ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20;

-- name: q03_ctr_by_platform
SELECT ext_service_name AS platform,
       COUNT(*) AS rows_, SUM(impressions) AS impressions, SUM(clicks) AS clicks,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY ext_service_name
ORDER BY ctr_pct DESC;

-- name: q04_ctr_by_channel
SELECT channel_name AS channel,
       COUNT(*) AS rows_, SUM(impressions) AS impressions,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY channel_name
ORDER BY ctr_pct DESC;

-- name: q05_ctr_by_targeting_type
SELECT search_tag_cat AS targeting,
       COUNT(*) AS rows_, SUM(impressions) AS impressions,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY search_tag_cat
HAVING COUNT(*) >= 100
ORDER BY ctr_pct DESC;

-- name: q06_top_advertisers
-- Advertisers with at least 100 valid rows, ranked by weighted CTR.
SELECT advertiser_name,
       COUNT(*) AS rows_, SUM(impressions) AS impressions,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY advertiser_name
HAVING COUNT(*) >= 100
ORDER BY ctr_pct DESC
LIMIT 10;

-- name: q07_ctr_by_budget_quintile
-- Window function: split rows into 5 equal budget groups, compare CTR.
WITH valid AS (
  SELECT *, NTILE(5) OVER (ORDER BY campaign_budget_usd) AS budget_quintile
  FROM ads
  WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
)
SELECT budget_quintile,
       ROUND(MIN(campaign_budget_usd),0) AS min_budget_usd,
       ROUND(MAX(campaign_budget_usd),0) AS max_budget_usd,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM valid
GROUP BY budget_quintile
ORDER BY budget_quintile;

-- name: q08_monthly_trend
SELECT SUBSTR(time,1,7) AS month,
       COUNT(*) AS rows_, SUM(impressions) AS impressions, SUM(clicks) AS clicks,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY month
ORDER BY month;

-- name: q09_campaign_length_buckets
SELECT CASE WHEN no_of_days <= 3  THEN '1-3'
            WHEN no_of_days <= 7  THEN '4-7'
            WHEN no_of_days <= 14 THEN '8-14'
            WHEN no_of_days <= 30 THEN '15-30'
            ELSE '31+' END AS campaign_day_bucket,
       COUNT(*) AS rows_,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY campaign_day_bucket
ORDER BY MIN(no_of_days);

-- name: q10_creative_size
SELECT CASE WHEN creative_width IS NULL THEN 'missing'
            WHEN creative_width = 0     THEN '0x0 (no size recorded)'
            ELSE CAST(creative_width AS INT) || 'x' || CAST(creative_height AS INT) END AS creative_size,
       COUNT(*) AS rows_,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY creative_size
ORDER BY rows_ DESC;

-- name: q11_best_campaigns
-- Subquery + HAVING: top campaigns by CTR with meaningful volume (>= 1M impressions).
SELECT campaign_item_id, advertiser_name,
       COUNT(*) AS days_, SUM(impressions) AS impressions,
       ROUND(100.0*SUM(clicks)/SUM(impressions), 3) AS ctr_pct
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions <= 0.20
GROUP BY campaign_item_id, advertiser_name
HAVING SUM(impressions) >= 1000000
ORDER BY ctr_pct DESC
LIMIT 10;

-- name: q12_glitch_rows_by_platform
-- Where do the invalid (CTR > 20%) rows come from?
SELECT ext_service_name AS platform, COUNT(*) AS glitch_rows
FROM ads
WHERE impressions > 0 AND 1.0*clicks/impressions > 0.20
GROUP BY ext_service_name
ORDER BY glitch_rows DESC;
