# Trader Performance vs Market Sentiment — Primetrade.ai Assignment

Analysis of how Bitcoin Fear/Greed sentiment correlates with Hyperliquid trader behavior and performance.

---

## Quick Start

```bash
# 1. Place datasets in the data/ folder
#    - data/fear_greed_index.csv
#    - data/historical_data.csv

# 2. Run the full analysis (generates all charts)
python analysis.py

# 3. Open the notebook for annotated walkthrough
jupyter notebook analysis.ipynb
```

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (all standard)

---

## Dataset Overview

| Dataset | Rows | Columns | Period |
|---------|------|---------|--------|
| Fear/Greed Index | 2,644 | 4 | Feb 2018 – May 2025 |
| Hyperliquid Trades | 211,224 | 16 | May 2023 – May 2025 |

**Key cleaning note:** The trade dataset contains 211,224 rows but only 2,810 unique Trade IDs. The remaining rows are **sub-fills of the same order** (partial executions) — these are valid and retained. This was confirmed by checking that Transaction Hashes match across sub-fill rows.

---

## Methodology

### Part A — Data Preparation
- Parsed `Timestamp IST` (format: `%d-%m-%Y %H:%M`) to extract trade dates
- Merged datasets on `date` (inner join → 211,218 matched rows)
- Simplified 5-class sentiment (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) into 3 buckets
- Computed **daily per-trader metrics**: PnL, trade count, volume, win rate, long/short ratio
- Computed **trader-level aggregates**: total PnL, average size, win rate, Sharpe proxy

### Part B — Analysis
- **B1 Performance:** Compared avg daily PnL, median PnL, and win rate across sentiment regimes
- **B2 Behavior:** Analyzed trade frequency, position size, and directional bias by sentiment
- **B3 Segmentation:** 3 segmentation schemes — frequency (Infrequent/Moderate/Frequent), position size (Small/Medium/Large), consistency (Sharpe proxy)
- **B4 Clustering:** KMeans (k=4) on scaled features → 4 behavioral archetypes

---

## Key Insights

### Insight 1 — Fear Days Generate More PnL (Counter-Intuitive)
Fear/Extreme Fear days produce **$5,185 avg daily PnL** vs **$4,144 on Greed days** (+25%), despite a slightly lower win rate (84.2% vs 85.6%). The wins on Fear days are larger in magnitude — traders who hold conviction during panic capture oversold bounces.

### Insight 2 — Traders Become More Aggressive During Fear
- **37% more trades** per day on Fear vs Greed (105 vs 77)
- **43% larger** average position size ($8,530 vs $5,955)
- **Long/Short ratio 37% higher** on Fear (2.24 vs 1.63) — systematic dip-buying

### Insight 3 — Active, High-Frequency Traders Extract Most Alpha on Fear Days
The Sentiment × Segment heatmap shows Frequent traders and Large position-size traders peak specifically during Fear regimes. This is not coincidence — these traders are structurally set up to exploit volatility.

---

## Strategy Recommendations (Part C)

### Strategy 1: Fear-Day Position Scaling for Active Winners
> During Fear/Extreme Fear (F&G score < 40), increase position size by 25% for Active Winner accounts.

*Rationale:* Active Winners (49% win rate, avg $1.03M PnL) concentrate their alpha during volatility. Their long bias during fear is deliberate and statistically profitable.

### Strategy 2: Trade Throttle for Passive Traders on Greed Days
> Cap daily trades at 60% of average for Passive/Inactive traders during Greed sentiment.

*Rationale:* Passive traders show no PnL improvement from increased Greed-day activity. Reducing noise trades cuts fees without sacrificing alpha.

### Strategy 3: Long Bias Signal from Extreme Fear
> When F&G score < 25 (Extreme Fear), bias all new entries toward BUY and avoid new short entries for 24–48 hours.

*Rationale:* L/S ratio hits 2.24 on Fear days. The F&G numeric score ranks 3rd in predictive importance for next-day profitability.

---

## Bonus: Predictive Model

**Random Forest Classifier** predicting next-day profitability bucket (loss/profit):

| Metric | Profit Day | Loss Day |
|--------|-----------|----------|
| Precision | 72% | 43% |
| Recall | 93% | 13% |
| **Overall Accuracy** | **70%** | |

**Top features by importance:**
1. `total_pnl` (today's PnL — momentum)
2. `win_rate` (today's win rate)
3. `value` (Fear/Greed numeric score)
4. `long_trades`
5. `avg_size_usd`

---

## Output Charts

| Chart | Description |
|-------|-------------|
| chart1 | Avg daily PnL & Win Rate by sentiment |
| chart2 | Trades/day, position size, L/S ratio by sentiment |
| chart3 | PnL distribution boxplots by sentiment & frequency |
| chart4 | Heatmap — sentiment × segment avg PnL |
| chart5 | Win rate by segment × sentiment |
| chart6 | Trader archetype profiles (KMeans) |
| chart7 | Cumulative PnL timeseries with sentiment overlay |
| chart8 | Feature importance for next-day prediction model |
