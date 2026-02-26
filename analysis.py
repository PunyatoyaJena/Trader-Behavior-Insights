"""
Primetrade.ai Internship Assignment
Trader Performance vs Market Sentiment (Fear/Greed)
Dataset: Hyperliquid Historical Trades + Bitcoin Fear/Greed Index
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid", palette="muted")
FG_PALETTE = {"Fear/Extreme Fear": "#e74c3c", "Greed/Extreme Greed": "#2ecc71", "Neutral": "#f0a500"}
ORDER = ['Fear/Extreme Fear', 'Neutral', 'Greed/Extreme Greed']
PALETTE = [FG_PALETTE[k] for k in ORDER]

print("=" * 60)
print("PART A — DATA LOADING & PREPARATION")
print("=" * 60)

fg = pd.read_csv("data/fear_greed_index.csv")
trades = pd.read_csv("data/historical_data.csv")

print(f"\n[Fear/Greed] Rows: {len(fg):,}  |  Columns: {fg.shape[1]}")
print(f"  Missing: {fg.isnull().sum().sum()}  |  Duplicates: {fg.duplicated().sum()}")

print(f"\n[Trades] Rows: {len(trades):,}  |  Columns: {trades.shape[1]}")
print(f"  Missing: {trades.isnull().sum().sum()}")
print(f"  Note: Rows sharing Trade ID are sub-fills of same order (valid, not duplicates)")
print(f"  Unique Trade IDs: {trades['Trade ID'].nunique():,}")
print(f"  Unique Accounts: {trades['Account'].nunique()}")

# Clean Fear/Greed
fg['date'] = pd.to_datetime(fg['date']).dt.date
fg = fg.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

def simplify_sentiment(c):
    c = str(c)
    if 'Fear' in c: return 'Fear/Extreme Fear'
    if 'Greed' in c: return 'Greed/Extreme Greed'
    return 'Neutral'

fg['sentiment'] = fg['classification'].apply(simplify_sentiment)

# Clean Trades
trades['datetime'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades = trades.dropna(subset=['datetime'])
trades['date'] = trades['datetime'].dt.date
trades['Closed PnL'] = pd.to_numeric(trades['Closed PnL'], errors='coerce').fillna(0)
trades['Size USD'] = pd.to_numeric(trades['Size USD'], errors='coerce').fillna(0)
trades['is_win'] = trades['Closed PnL'] > 0
trades['is_close'] = trades['Closed PnL'] != 0

print(f"\n[After Cleaning] Date range: {trades['date'].min()} to {trades['date'].max()}")
print(f"  Unique accounts: {trades['Account'].nunique()}, Coins: {trades['Coin'].nunique()}")

# Merge
fg_slim = fg[['date','classification','sentiment','value']].copy()
merged = trades.merge(fg_slim, on='date', how='inner')
print(f"\n[Merged] {len(merged):,} rows | Sentiment breakdown:")
print(merged['sentiment'].value_counts().to_string())

# Daily metrics
daily = merged.groupby(['Account','date','sentiment','classification','value']).agg(
    total_pnl=('Closed PnL','sum'),
    num_trades=('Trade ID','count'),
    avg_size_usd=('Size USD','mean'),
    total_volume=('Size USD','sum'),
    win_trades=('is_win','sum'),
    close_trades=('is_close','sum'),
    long_trades=('Side', lambda x: (x=='BUY').sum()),
    short_trades=('Side', lambda x: (x=='SELL').sum()),
).reset_index()

daily['win_rate'] = np.where(daily['close_trades'] > 0,
                              daily['win_trades'] / daily['close_trades'], np.nan)
daily['long_short_ratio'] = np.where(daily['short_trades'] > 0,
                                      daily['long_trades'] / daily['short_trades'], np.nan)

# Trader-level stats
trader_stats = merged.groupby('Account').agg(
    total_pnl=('Closed PnL','sum'),
    avg_pnl_per_trade=('Closed PnL','mean'),
    total_trades=('Trade ID','count'),
    avg_size=('Size USD','mean'),
    win_rate=('is_win','mean'),
    pnl_std=('Closed PnL','std'),
).reset_index()
trader_stats['pnl_std'] = trader_stats['pnl_std'].fillna(0)
trader_stats['sharpe_proxy'] = np.where(
    trader_stats['pnl_std'] > 0,
    trader_stats['avg_pnl_per_trade'] / trader_stats['pnl_std'], 0)

print(f"\n[Metrics] Daily rows: {len(daily):,} | Trader stats: {len(trader_stats)}")

print("\n" + "=" * 60)
print("PART B — ANALYSIS")
print("=" * 60)

b1 = daily.groupby('sentiment').agg(
    avg_daily_pnl=('total_pnl','mean'),
    median_daily_pnl=('total_pnl','median'),
    avg_win_rate=('win_rate','mean'),
    total_pnl=('total_pnl','sum'),
    n_trader_days=('Account','count'),
).reindex(ORDER).reset_index()
print("\n[B1] Performance by Sentiment:")
print(b1.to_string(index=False))

b2 = daily.groupby('sentiment').agg(
    avg_trades_per_day=('num_trades','mean'),
    avg_volume_usd=('total_volume','mean'),
    avg_size_usd=('avg_size_usd','mean'),
    avg_long_short_ratio=('long_short_ratio','mean'),
).reindex(ORDER).reset_index()
print("\n[B2] Behavior by Sentiment:")
print(b2.to_string(index=False))

# Segmentation
freq_q = trader_stats['total_trades'].quantile([0.33, 0.66])
trader_stats['freq_segment'] = pd.cut(
    trader_stats['total_trades'],
    bins=[-np.inf, freq_q[0.33], freq_q[0.66], np.inf],
    labels=['Infrequent', 'Moderate', 'Frequent']
).astype(str)

size_q = trader_stats['avg_size'].quantile([0.33, 0.66])
trader_stats['size_segment'] = pd.cut(
    trader_stats['avg_size'],
    bins=[-np.inf, size_q[0.33], size_q[0.66], np.inf],
    labels=['Small', 'Medium', 'Large']
).astype(str)

trader_stats['consistency'] = pd.cut(
    trader_stats['sharpe_proxy'],
    bins=[-np.inf, -0.01, 0.01, np.inf],
    labels=['Consistent Loser', 'Neutral', 'Consistent Winner']
).astype(str)

daily = daily.merge(
    trader_stats[['Account','freq_segment','size_segment','consistency']],
    on='Account', how='left'
)

# KMeans
features = ['total_trades','avg_size','win_rate','total_pnl','pnl_std']
X = trader_stats[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)
trader_stats['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)
cluster_profile = trader_stats.groupby('cluster')[features + ['sharpe_proxy']].mean()
print("\n[Cluster Profiles]:")
print(cluster_profile.round(2).to_string())

def label_archetype(row):
    med_pnl = cluster_profile['total_pnl'].median()
    med_trades = cluster_profile['total_trades'].median()
    if row['total_pnl'] >= med_pnl and row['total_trades'] >= med_trades: return 'Active Winners'
    if row['total_pnl'] >= med_pnl: return 'Selective Winners'
    if row['total_trades'] >= med_trades: return 'Active Losers'
    return 'Passive/Inactive'

trader_stats['archetype'] = trader_stats.apply(label_archetype, axis=1)

print("\n" + "=" * 60)
print("GENERATING CHARTS")
print("=" * 60)

b1_ord = b1.set_index('sentiment').reindex(ORDER).reset_index()
b2_ord = b2.set_index('sentiment').reindex(ORDER).reset_index()

# Chart 1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Chart 1 — Trader Performance by Market Sentiment', fontsize=15, fontweight='bold')
axes[0].bar(b1_ord['sentiment'], b1_ord['avg_daily_pnl'], color=PALETTE, edgecolor='white', linewidth=1.5)
axes[0].axhline(0, color='white', linewidth=0.8, linestyle='--')
axes[0].set_title('Avg Daily PnL per Trader-Day')
axes[0].set_ylabel('USD')
for i, v in enumerate(b1_ord['avg_daily_pnl']):
    axes[0].text(i, v + (30 if v >= 0 else -120), f"${v:,.0f}", ha='center', fontsize=11, fontweight='bold', color='white')
axes[1].bar(b1_ord['sentiment'], b1_ord['avg_win_rate'] * 100, color=PALETTE, edgecolor='white', linewidth=1.5)
axes[1].set_title('Avg Win Rate (%)')
axes[1].set_ylabel('Win Rate %')
axes[1].set_ylim(0, 100)
for i, v in enumerate(b1_ord['avg_win_rate'] * 100):
    axes[1].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig('charts/chart1_performance_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 1 saved")

# Chart 2
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 2 — Trader Behavior by Market Sentiment', fontsize=15, fontweight='bold')
for ax, col, title, ylabel in zip(axes,
    ['avg_trades_per_day','avg_size_usd','avg_long_short_ratio'],
    ['Avg Trades per Day','Avg Trade Size (USD)','Long/Short Ratio'],
    ['Trades','USD','Ratio (>1 = more longs)']):
    ax.bar(b2_ord['sentiment'], b2_ord[col], color=PALETTE, edgecolor='white')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
axes[2].axhline(1.0, color='white', linewidth=1, linestyle='--', label='Equal L/S')
axes[2].legend(fontsize=9)
plt.tight_layout()
plt.savefig('charts/chart2_behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 2 saved")

# Chart 3
q_lo = daily['total_pnl'].quantile(0.05)
q_hi = daily['total_pnl'].quantile(0.95)
daily['pnl_capped'] = daily['total_pnl'].clip(q_lo, q_hi)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Chart 3 — PnL Distribution by Sentiment & Frequency Segment', fontsize=14, fontweight='bold')
sent_sub = daily[daily['sentiment'].isin(ORDER)].copy()
sns.boxplot(data=sent_sub, x='sentiment', y='pnl_capped', order=ORDER, palette=PALETTE, ax=axes[0], width=0.5, fliersize=3)
axes[0].axhline(0, color='white', linestyle='--', linewidth=1)
axes[0].set_title('Daily PnL by Sentiment')
axes[0].set_ylabel('Daily PnL USD (5–95th pct)')
freq_order = ['Infrequent', 'Moderate', 'Frequent']
freq_sub = daily[daily['freq_segment'].isin(freq_order)].copy()
sns.boxplot(data=freq_sub, x='freq_segment', y='pnl_capped', order=freq_order,
            palette=['#3498db','#9b59b6','#e67e22'], ax=axes[1], width=0.5, fliersize=3)
axes[1].axhline(0, color='white', linestyle='--', linewidth=1)
axes[1].set_title('Daily PnL by Frequency Segment')
axes[1].set_ylabel('Daily PnL USD (5–95th pct)')
plt.tight_layout()
plt.savefig('charts/chart3_pnl_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 3 saved")

# Chart 4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 4 — Avg Daily PnL Heatmap', fontsize=14, fontweight='bold')
size_order = ['Small', 'Medium', 'Large']
heat1 = daily[daily['freq_segment'].isin(freq_order)].groupby(['sentiment','freq_segment'])['total_pnl'].mean().unstack(fill_value=0)
heat1 = heat1.reindex(ORDER).reindex(columns=freq_order)
sns.heatmap(heat1, annot=True, fmt='.0f', cmap='RdYlGn', center=0, linewidths=0.5, ax=axes[0], cbar_kws={'label':'Avg PnL'})
axes[0].set_title('Frequency Segment')
axes[0].set_ylabel('Sentiment')
heat2 = daily[daily['size_segment'].isin(size_order)].groupby(['sentiment','size_segment'])['total_pnl'].mean().unstack(fill_value=0)
heat2 = heat2.reindex(ORDER).reindex(columns=size_order)
sns.heatmap(heat2, annot=True, fmt='.0f', cmap='RdYlGn', center=0, linewidths=0.5, ax=axes[1], cbar_kws={'label':'Avg PnL'})
axes[1].set_title('Position Size Segment')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('charts/chart4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 4 saved")

# Chart 5
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 5 — Win Rate by Segment & Sentiment', fontsize=14, fontweight='bold')
wr1 = daily[daily['freq_segment'].isin(freq_order)].groupby(['sentiment','freq_segment'])['win_rate'].mean().reset_index()
wr1 = wr1[wr1['sentiment'].isin(ORDER)]
sns.barplot(data=wr1, x='freq_segment', y='win_rate', hue='sentiment', hue_order=ORDER, palette=PALETTE, ax=axes[0], order=freq_order)
axes[0].set_title('Win Rate: Frequency × Sentiment')
axes[0].set_ylabel('Win Rate')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
axes[0].legend(title='', fontsize=8)
wr2 = daily[daily['size_segment'].isin(size_order)].groupby(['sentiment','size_segment'])['win_rate'].mean().reset_index()
wr2 = wr2[wr2['sentiment'].isin(ORDER)]
sns.barplot(data=wr2, x='size_segment', y='win_rate', hue='sentiment', hue_order=ORDER, palette=PALETTE, ax=axes[1], order=size_order)
axes[1].set_title('Win Rate: Position Size × Sentiment')
axes[1].set_ylabel('Win Rate')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
axes[1].legend(title='', fontsize=8)
plt.tight_layout()
plt.savefig('charts/chart5_winrate.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 5 saved")

# Chart 6
arch = trader_stats.groupby('archetype').agg(
    count=('Account','count'),
    avg_total_pnl=('total_pnl','mean'),
    avg_win_rate=('win_rate','mean'),
    avg_trades=('total_trades','mean'),
).reset_index().sort_values('avg_total_pnl', ascending=False)
pal_arch = ['#2ecc71','#3498db','#f39c12','#e74c3c']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 6 — Trader Archetypes (KMeans Clusters)', fontsize=14, fontweight='bold')
axes[0].barh(arch['archetype'], arch['avg_total_pnl'], color=pal_arch, edgecolor='white')
axes[0].axvline(0, color='white', linewidth=1, linestyle='--')
axes[0].set_title('Avg Total PnL')
axes[0].set_xlabel('PnL (USD)')
axes[1].bar(arch['archetype'], arch['avg_trades'], color=pal_arch, edgecolor='white')
axes[1].set_title('Avg Total Trades')
axes[1].tick_params(axis='x', rotation=12)
axes[2].bar(arch['archetype'], arch['avg_win_rate'] * 100, color=pal_arch, edgecolor='white')
axes[2].set_title('Avg Win Rate %')
axes[2].tick_params(axis='x', rotation=12)
plt.tight_layout()
plt.savefig('charts/chart6_archetypes.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 6 saved")

# Chart 7
daily_agg = daily.groupby(['date','sentiment'])['total_pnl'].sum().reset_index()
daily_agg['date'] = pd.to_datetime(daily_agg['date'])
daily_agg = daily_agg.sort_values('date')
daily_agg['cum_pnl'] = daily_agg['total_pnl'].cumsum()
daily_vol = daily.groupby('date')['num_trades'].sum().reset_index()
daily_vol['date'] = pd.to_datetime(daily_vol['date'])
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Chart 7 — Cumulative PnL & Volume vs. Sentiment', fontsize=14, fontweight='bold')
bg = {'Fear/Extreme Fear':'#e74c3c','Neutral':'#f0a500','Greed/Extreme Greed':'#2ecc71'}
prev_d, prev_s = daily_agg.iloc[0]['date'], daily_agg.iloc[0]['sentiment']
for _, row in daily_agg.iterrows():
    if row['sentiment'] != prev_s:
        for ax in axes:
            ax.axvspan(prev_d, row['date'], alpha=0.15, color=bg.get(prev_s,'grey'), linewidth=0)
        prev_d, prev_s = row['date'], row['sentiment']
for ax in axes:
    ax.axvspan(prev_d, daily_agg.iloc[-1]['date'], alpha=0.15, color=bg.get(prev_s,'grey'), linewidth=0)
axes[0].plot(daily_agg['date'], daily_agg['cum_pnl'], color='white', linewidth=2)
axes[0].fill_between(daily_agg['date'], daily_agg['cum_pnl'], alpha=0.25, color='white')
axes[0].set_ylabel('Cumulative PnL (USD)')
axes[0].set_title('All Traders — Cumulative PnL (background = sentiment regime)')
axes[1].bar(daily_vol['date'], daily_vol['num_trades'], color='#3498db', alpha=0.8, width=1)
axes[1].set_ylabel('Total Trades')
axes[1].set_xlabel('Date')
axes[1].set_title('Daily Trade Volume')
from matplotlib.patches import Patch
legend_el = [Patch(facecolor='#e74c3c', alpha=0.5, label='Fear'),
             Patch(facecolor='#f0a500', alpha=0.5, label='Neutral'),
             Patch(facecolor='#2ecc71', alpha=0.5, label='Greed')]
axes[0].legend(handles=legend_el, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('charts/chart7_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 7 saved")

print("\n" + "=" * 60)
print("BONUS — PREDICTIVE MODEL")
print("=" * 60)
daily_m = daily.copy()
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
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Loss Day','Profit Day']))
fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fi.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_title('Chart 8 — Feature Importances\n(Next-Day Profitability Prediction)', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('charts/chart8_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Chart 8 saved")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("\n=== B1: Performance by Sentiment ===")
print(b1[['sentiment','avg_daily_pnl','avg_win_rate','n_trader_days']].to_string(index=False))
print("\n=== B2: Behavior by Sentiment ===")
print(b2.to_string(index=False))
print("\n=== Trader Archetypes ===")
arch_full = trader_stats.groupby('archetype').agg(
    n_traders=('Account','count'),
    avg_pnl=('total_pnl','mean'),
    avg_win_rate=('win_rate','mean'),
    avg_trades=('total_trades','mean'),
).reset_index()
print(arch_full.round(2).to_string(index=False))
print("\n✅ Analysis complete! All 8 charts saved to charts/")
