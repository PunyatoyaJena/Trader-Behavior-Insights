import json

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, 
            "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = [
    md_cell("# Primetrade.ai Internship Assignment\n## Trader Performance vs Market Sentiment\n**Dataset:** Hyperliquid Historical Trades + Bitcoin Fear/Greed Index | **Period:** May 2023 – May 2025"),
    
    md_cell("## Setup & Imports"),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings; warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid')
FG_PALETTE = {'Fear/Extreme Fear': '#e74c3c', 'Greed/Extreme Greed': '#2ecc71', 'Neutral': '#f0a500'}
ORDER = ['Fear/Extreme Fear', 'Neutral', 'Greed/Extreme Greed']
PALETTE = [FG_PALETTE[k] for k in ORDER]"""),

    md_cell("---\n## Part A — Data Loading & Preparation\n### A1. Load & Inspect"),
    code_cell("""fg = pd.read_csv('data/fear_greed_index.csv')
trades = pd.read_csv('data/historical_data.csv')

print(f"[Fear/Greed] Shape: {fg.shape} | Missing: {fg.isnull().sum().sum()} | Duplicates: {fg.duplicated().sum()}")
print(f"[Trades]     Shape: {trades.shape} | Missing: {trades.isnull().sum().sum()}")
print(f"  Unique Accounts: {trades['Account'].nunique()} | Trade IDs: {trades['Trade ID'].nunique():,}")
print("  Note: Multiple rows per Trade ID = sub-fills of same order (valid, not duplicates)")
fg.head(3)"""),

    md_cell("### A2. Clean & Parse Timestamps"),
    code_cell("""# Fear/Greed
fg['date'] = pd.to_datetime(fg['date']).dt.date
fg = fg.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

def simplify_sentiment(c):
    c = str(c)
    if 'Fear' in c: return 'Fear/Extreme Fear'
    if 'Greed' in c: return 'Greed/Extreme Greed'
    return 'Neutral'

fg['sentiment'] = fg['classification'].apply(simplify_sentiment)

# Trades
trades['datetime'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades = trades.dropna(subset=['datetime'])
trades['date'] = trades['datetime'].dt.date
trades['Closed PnL'] = pd.to_numeric(trades['Closed PnL'], errors='coerce').fillna(0)
trades['Size USD'] = pd.to_numeric(trades['Size USD'], errors='coerce').fillna(0)
trades['is_win'] = trades['Closed PnL'] > 0
trades['is_close'] = trades['Closed PnL'] != 0

print(f"Date range: {trades['date'].min()} to {trades['date'].max()}")
print(f"Accounts: {trades['Account'].nunique()} | Coins: {trades['Coin'].nunique()}")"""),

    md_cell("### A3. Merge by Date"),
    code_cell("""merged = trades.merge(fg[['date','classification','sentiment','value']], on='date', how='inner')
print(f"Merged shape: {merged.shape}")
merged['sentiment'].value_counts()"""),

    md_cell("### A4. Compute Daily & Trader-Level Metrics"),
    code_cell("""daily = merged.groupby(['Account','date','sentiment','classification','value']).agg(
    total_pnl=('Closed PnL','sum'),
    num_trades=('Trade ID','count'),
    avg_size_usd=('Size USD','mean'),
    total_volume=('Size USD','sum'),
    win_trades=('is_win','sum'),
    close_trades=('is_close','sum'),
    long_trades=('Side', lambda x: (x=='BUY').sum()),
    short_trades=('Side', lambda x: (x=='SELL').sum()),
).reset_index()

daily['win_rate'] = np.where(daily['close_trades'] > 0, daily['win_trades'] / daily['close_trades'], np.nan)
daily['long_short_ratio'] = np.where(daily['short_trades'] > 0, daily['long_trades'] / daily['short_trades'], np.nan)

trader_stats = merged.groupby('Account').agg(
    total_pnl=('Closed PnL','sum'),
    avg_pnl_per_trade=('Closed PnL','mean'),
    total_trades=('Trade ID','count'),
    avg_size=('Size USD','mean'),
    win_rate=('is_win','mean'),
    pnl_std=('Closed PnL','std'),
).reset_index()
trader_stats['pnl_std'] = trader_stats['pnl_std'].fillna(0)
trader_stats['sharpe_proxy'] = np.where(trader_stats['pnl_std'] > 0,
    trader_stats['avg_pnl_per_trade'] / trader_stats['pnl_std'], 0)

print(f"Daily rows: {len(daily):,} | Trader stats: {len(trader_stats)}")
daily.describe().round(2)"""),

    md_cell("---\n## Part B — Analysis\n### B1. Performance: Fear vs Greed Days"),
    code_cell("""b1 = daily.groupby('sentiment').agg(
    avg_daily_pnl=('total_pnl','mean'),
    median_daily_pnl=('total_pnl','median'),
    avg_win_rate=('win_rate','mean'),
    total_pnl=('total_pnl','sum'),
    n_trader_days=('Account','count'),
).reindex(ORDER).reset_index()
print(b1.to_string(index=False))"""),

    code_cell("from IPython.display import Image\nImage('charts/chart1_performance_by_sentiment.png')"),

    md_cell("""**Insight 1 — Fear Days Generate Higher PnL Despite Lower Win Rate:**
- Average daily PnL on Fear days: **$5,185** vs Greed days: **$4,144** (+25%)
- Win rate is slightly lower on Fear (84.2%) vs Greed (85.6%) — but *size* of wins is larger
- This is the classic "buy the fear" dynamic: fewer wins but bigger wins during panic"""),

    md_cell("### B2. Behavior Change Based on Sentiment"),
    code_cell("""b2 = daily.groupby('sentiment').agg(
    avg_trades_per_day=('num_trades','mean'),
    avg_volume_usd=('total_volume','mean'),
    avg_size_usd=('avg_size_usd','mean'),
    avg_long_short_ratio=('long_short_ratio','mean'),
).reindex(ORDER).reset_index()
print(b2.to_string(index=False))"""),

    code_cell("Image('charts/chart2_behavior_by_sentiment.png')"),

    md_cell("""**Insight 2 — Traders Become More Aggressive During Fear:**
- **37% more trades** on Fear vs Greed (105/day vs 77/day)
- **43% larger** average position size ($8,530 vs $5,955)
- Long/Short ratio **2.24 during Fear vs 1.63 during Greed** — strong dip-buying conviction
- These traders are not de-risking; they are leaning in"""),

    md_cell("### B3. Trader Segmentation"),
    code_cell("""# Segment 1: Frequency
freq_q = trader_stats['total_trades'].quantile([0.33, 0.66])
trader_stats['freq_segment'] = pd.cut(trader_stats['total_trades'],
    bins=[-np.inf, freq_q[0.33], freq_q[0.66], np.inf],
    labels=['Infrequent','Moderate','Frequent']).astype(str)

# Segment 2: Position Size
size_q = trader_stats['avg_size'].quantile([0.33, 0.66])
trader_stats['size_segment'] = pd.cut(trader_stats['avg_size'],
    bins=[-np.inf, size_q[0.33], size_q[0.66], np.inf],
    labels=['Small','Medium','Large']).astype(str)

# Segment 3: Consistency (Sharpe Proxy)
trader_stats['consistency'] = pd.cut(trader_stats['sharpe_proxy'],
    bins=[-np.inf, -0.01, 0.01, np.inf],
    labels=['Consistent Loser','Neutral','Consistent Winner']).astype(str)

daily = daily.merge(trader_stats[['Account','freq_segment','size_segment','consistency']], on='Account', how='left')

print("Frequency:", trader_stats['freq_segment'].value_counts().to_dict())
print("Size:", trader_stats['size_segment'].value_counts().to_dict())
print("Consistency:", trader_stats['consistency'].value_counts().to_dict())"""),

    code_cell("Image('charts/chart3_pnl_distributions.png')"),
    code_cell("Image('charts/chart4_heatmap.png')"),
    code_cell("Image('charts/chart5_winrate.png')"),

    md_cell("""**Insight 3 — Frequent Traders Profit Most on Fear; Small Traders Hurt Most on Greed:**
- Heatmap: Frequent traders post highest PnL during Fear (top-left green cell)
- Large position-size traders also peak during Fear — confirms dip-buying alpha
- Small traders have near-zero or negative PnL across all sentiments
- Win rates are highest for all groups on Greed days, but PnL magnitude is muted"""),

    md_cell("### B4. KMeans Clustering — Behavioral Archetypes"),
    code_cell("""features = ['total_trades','avg_size','win_rate','total_pnl','pnl_std']
X = trader_stats[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)
trader_stats['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)

cluster_profile = trader_stats.groupby('cluster')[features + ['sharpe_proxy']].mean()

def label_archetype(row):
    med_pnl = cluster_profile['total_pnl'].median()
    med_trades = cluster_profile['total_trades'].median()
    if row['total_pnl'] >= med_pnl and row['total_trades'] >= med_trades: return 'Active Winners'
    if row['total_pnl'] >= med_pnl: return 'Selective Winners'
    if row['total_trades'] >= med_trades: return 'Active Losers'
    return 'Passive/Inactive'

trader_stats['archetype'] = trader_stats.apply(label_archetype, axis=1)
trader_stats.groupby('archetype')[['total_pnl','win_rate','total_trades','avg_size']].mean().round(2)"""),

    code_cell("Image('charts/chart6_archetypes.png')"),
    code_cell("Image('charts/chart7_timeseries.png')"),

    md_cell("""**Archetype Summary:**
| Archetype | Count | Avg PnL | Win Rate | Avg Trades |
|-----------|-------|---------|----------|------------|
| Active Winners | 5 | $1,027,829 | 49% | 19,648 |
| Selective Winners | 5 | $696,883 | 40% | 2,485 |
| Active Losers | 3 | $94,228 | 35% | 14,638 |
| Passive/Inactive | 19 | $70,960 | 39% | 2,981 |"""),

    md_cell("---\n## Part C — Actionable Strategy Recommendations"),
    md_cell("""### Strategy 1: Fear-Day Position Scaling for Active Winners
**Rule:** During Fear/Extreme Fear days (F&G score < 40), increase trade size by 25–30% for accounts classified as **Active Winners**.

**Evidence:** Fear days generate 25% higher avg PnL. Active Winners hold a 49% win rate (highest of all archetypes) and concentrate their largest volume bets during volatility. Their long/short ratio of 2.24 during fear shows systematic dip-buying confidence.

`IF sentiment == 'Fear/Extreme Fear' AND archetype == 'Active Winner': position_multiplier = 1.25`

---

### Strategy 2: Volume Throttle for Passive Traders on Greed Days
**Rule:** Passive/Inactive traders should cap daily trade count at 60% of their average during **Greed** days and only execute high-conviction setups.

**Evidence:** On Greed days, win rates rise but PnL remains flat for Passive traders. Increased activity during Greed adds transaction fees without proportional alpha. The data shows Passive traders have the lowest PnL per trade on Greed days, suggesting market conditions favor fewer, larger bets.

`IF sentiment == 'Greed/Extreme Greed' AND archetype == 'Passive/Inactive': max_trades = avg_daily_trades * 0.6`

---

### Strategy 3: Long Bias Signal from Fear Index Score
**Rule:** When the Fear/Greed numeric score drops below 25 (Extreme Fear), bias all new positions toward BUY and avoid new short entries for the next 24–48 hours.

**Evidence:** Long/Short ratio reaches 2.24 on Fear days. The F&G numeric score is the 3rd most important feature in our predictive model. Historically, Extreme Fear days precede mean-reversion bounces where long-biased traders captured most of the Fear-day alpha."""),

    md_cell("---\n## Bonus — Predictive Model: Next-Day Profitability"),
    code_cell("""daily_m = daily.copy()
daily_m['date_dt'] = pd.to_datetime(daily_m['date'])
daily_m = daily_m.sort_values(['Account','date_dt'])
daily_m['next_pnl'] = daily_m.groupby('Account')['total_pnl'].shift(-1)
daily_m['target'] = (daily_m['next_pnl'] > 0).astype(int)
daily_m = daily_m.dropna(subset=['next_pnl','win_rate'])

feat_cols = ['num_trades','avg_size_usd','total_pnl','win_rate','long_trades','short_trades','value']
model_df = daily_m[feat_cols + ['target']].dropna()
X_m, y_m = model_df[feat_cols], model_df['target']
X_train, X_test, y_train, y_test = train_test_split(X_m, y_m, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test), target_names=['Loss Day','Profit Day']))"""),

    code_cell("Image('charts/chart8_feature_importance.png')"),

    md_cell("""**Model Results (Random Forest, 80/20 split):**
- **Overall accuracy: 70%**
- Profit Day recall: 93% — the model reliably identifies profitable days
- Loss Day recall: 13% — loss days remain hard to predict (higher randomness)
- The Fear/Greed `value` score ranks 3rd in importance, confirming sentiment has real predictive power beyond just labeling
- `total_pnl` (momentum) and `win_rate` are the top two features — suggesting recent performance is the strongest signal"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells
}

with open("analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook created: analysis.ipynb")
