# Ad Campaign CTR: Exploratory Analysis

Which platforms, channels, targeting types, advertisers and budgets get more clicks per impression?

- **Data:** `../Dataset.csv`, 72,612 campaign-day rows, 157 campaigns, 47 advertisers, May to Dec 2022
- **Tools:** Python (pandas, Matplotlib, Seaborn), SQL (SQLite, 12 queries incl. window functions, CTEs, HAVING)
- **Files:** `ctr_analysis.ipynb` (executed, with charts) · `sql_queries.sql` · `charts/` · `results/` (CSV output of every query)

Run it: `pip install pandas matplotlib seaborn jupyter` then open the notebook from this folder.

## Data cleaning
| Issue | Rows | What I did |
|---|---|---|
| CTR above 20% (clicks close to or above impressions) | 157 | Excluded as reporting glitches (spread evenly over the 3 platforms) |
| `position_in_content`, `unique_reach`, `total_reach` | 100% empty | Not usable |
| `max_bid_cpm` | 90% empty | Not usable |
| Creative size missing / recorded as 0x0 | 3,314 / 7,425 | Kept, reported separately |

72,455 valid rows remain.

## Key insights
1. **Use weighted CTR.** Total clicks / total impressions is **3.22%**, but the simple average of row-level CTRs is **1.64%**. Big campaign days click much more than the many small ones, so a plain average understates performance by about half.
2. **Platform and channel barely matter.** The five channels sit between 3.17% and 3.30%; the three ad platforms between 3.09% and 3.33%.
3. **Targeting type matters a lot.** Retargeting (3.61%) and Others (3.54%) beat Youtube (1.51%), Inmarket (1.37%) and Affinity (1.19%) by more than 2x.
4. **Advertisers differ widely.** Among advertisers with 100+ rows, the best reach 4.3% to 7.3% against a 3.2% overall average. Some single campaigns with 1M+ impressions exceed 7% (e.g. campaign 3056 at 8.1%).
5. **Higher budget goes with higher CTR.** Top budget quintile (over $761) gets 4.34% against 1.43% for the bottom quintile (under $151). This is a correlation: large budgets may belong to advertisers or targeting types that already perform well, so it does not prove that spending more causes more clicks.
6. **Campaigns fade.** CTR peaks around days 8 to 14 (3.73%) and drops to 1.69% after day 30, which suggests ad fatigue and a case for refreshing creatives.
7. **Time pattern.** October is the peak (4.05%); November and December fall to 1.6% to 1.8%. With one year of data, seasonality cannot be told apart from changes in campaign mix.
8. **Leakage warning.** `media_cost_usd` is strongly correlated with clicks (0.85) and impressions (0.93) because it is an outcome of the campaign. That is why the predictor in this repo does not use it.

## Limitations
One year of data; a single dataset with no experiment, so everything above is observational; missing creative size is not random (those rows have 5.59% CTR), so dropping them could bias results.
