"""
Module 3: Correlation Analysis and Time Series Analysis
========================================================
Syllabus Topics:
  - Types of analysis: Univariate, Bivariate, Multivariate
  - Time Series Analysis (TSA): Fundamentals & Characteristics
  - Time-based indexing
  - Visualizing time series
  - Grouping time series data
  - Resampling time series data
"""

import os, sys, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utils.data_loader import (load_timeseries, load_country,
                          get_multi_country_pivot, FOCUS_COUNTRIES, PLOTS_DIR)

print("=" * 80)
print("MODULE 3: CORRELATION ANALYSIS AND TIME SERIES ANALYSIS")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3.1 UNIVARIATE ANALYSIS")
print("─" * 60)

india = load_country("India")
cases = india["daily_new_cases"]

print(f"  Variable: India daily_new_cases")
print(f"  Count:      {cases.count()}")
print(f"  Mean:       {cases.mean():>12,.2f}")
print(f"  Median:     {cases.median():>12,.2f}")
print(f"  Std Dev:    {cases.std():>12,.2f}")
print(f"  Skewness:   {cases.skew():>12.4f}")
print(f"  Kurtosis:   {cases.kurtosis():>12.4f}")
print(f"  Min:        {cases.min():>12,.0f}")
print(f"  Max:        {cases.max():>12,.0f}")
print(f"  Range:      {cases.max() - cases.min():>12,.0f}")
print(f"  IQR:        {cases.quantile(0.75) - cases.quantile(0.25):>12,.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Module 3 — Univariate Analysis: India Daily Cases",
             fontsize=14, fontweight="bold")

# Histogram + KDE
axes[0].hist(cases, bins=60, density=True, color="#93c5fd", edgecolor="white",
             alpha=0.7, label="Histogram")
cases.plot.kde(ax=axes[0], color="#1d4ed8", linewidth=2, label="KDE")
axes[0].set_title("Distribution (Histogram + KDE)")
axes[0].set_xlabel("Daily New Cases")
axes[0].legend()

# Box Plot
bp = axes[1].boxplot(cases.dropna(), vert=True, patch_artist=True,
                      boxprops=dict(facecolor="#bfdbfe"))
axes[1].set_title("Box Plot")
axes[1].set_ylabel("Daily New Cases")

# QQ Plot
stats.probplot(cases.dropna(), dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot (Normality Check)")
axes[2].get_lines()[0].set_color("#2563eb")
axes[2].get_lines()[1].set_color("#dc2626")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_univariate_analysis.png"), dpi=200)
plt.close()
print(f"\n  ✓ Saved: plots/m3_univariate_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3.2  BIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3.2 BIVARIATE ANALYSIS")
print("─" * 60)

# Cases vs Deaths correlation
pearson_r, pearson_p = stats.pearsonr(india["daily_new_cases"], india["daily_deaths"])
spearman_r, spearman_p = stats.spearmanr(india["daily_new_cases"], india["daily_deaths"])
kendall_tau, kendall_p = stats.kendalltau(india["daily_new_cases"], india["daily_deaths"])

print(f"  India: daily_new_cases vs daily_deaths")
print(f"    Pearson r:   {pearson_r:.4f}  (p = {pearson_p:.2e})")
print(f"    Spearman ρ:  {spearman_r:.4f}  (p = {spearman_p:.2e})")
print(f"    Kendall τ:   {kendall_tau:.4f}  (p = {kendall_p:.2e})")

# Cross-correlation (lagged relationship between cases and deaths)
max_lag = 30
cross_corr = [india["daily_new_cases"].corr(india["daily_deaths"].shift(lag))
              for lag in range(max_lag + 1)]
optimal_lag = np.argmax(cross_corr)
print(f"\n  Cross-Correlation Analysis (Cases → Deaths lag):")
print(f"    Optimal lag: {optimal_lag} days (r = {cross_corr[optimal_lag]:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Module 3 — Bivariate Analysis: India Cases vs Deaths",
             fontsize=14, fontweight="bold")

# Scatter plot
axes[0].scatter(india["daily_new_cases"], india["daily_deaths"],
                alpha=0.3, s=10, c="#2563eb")
z = np.polyfit(india["daily_new_cases"], india["daily_deaths"], 1)
p = np.poly1d(z)
x_line = np.linspace(0, india["daily_new_cases"].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f"r = {pearson_r:.3f}")
axes[0].set_xlabel("Daily New Cases")
axes[0].set_ylabel("Daily Deaths")
axes[0].set_title("Scatter Plot + Regression Line")
axes[0].legend()

# Hexbin (density scatter)
hb = axes[1].hexbin(india["daily_new_cases"], india["daily_deaths"],
                     gridsize=30, cmap="YlOrRd", mincnt=1)
axes[1].set_xlabel("Daily New Cases")
axes[1].set_ylabel("Daily Deaths")
axes[1].set_title("Hexbin Density Plot")
plt.colorbar(hb, ax=axes[1], label="Count")

# Cross-correlation plot
axes[2].bar(range(max_lag + 1), cross_corr, color="#60a5fa", edgecolor="white")
axes[2].axvline(optimal_lag, color="#dc2626", linestyle="--",
                label=f"Optimal Lag = {optimal_lag} days")
axes[2].set_xlabel("Lag (Days)")
axes[2].set_ylabel("Pearson Correlation")
axes[2].set_title("Cross-Correlation (Cases leads Deaths)")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_bivariate_analysis.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m3_bivariate_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3.3  MULTIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3.3 MULTIVARIATE ANALYSIS")
print("─" * 60)

pivot_cases = get_multi_country_pivot(FOCUS_COUNTRIES, "daily_new_cases")
corr_matrix = pivot_cases.corr()

print("  Correlation Matrix (daily new cases) among 10 countries:")
print(corr_matrix.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Module 3 — Multivariate Correlation Analysis",
             fontsize=14, fontweight="bold")

# Correlation heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=axes[0],
            square=True, cbar_kws={"shrink": 0.8})
axes[0].set_title("Pearson Correlation Heatmap", fontweight="bold")

# Pairwise scatter for top 4 countries
top4 = ["India", "US", "Brazil", "United Kingdom"]
pivot_top4 = pivot_cases[top4]

# Plot pairwise correlations as clustered heatmap
deaths_pivot = get_multi_country_pivot(FOCUS_COUNTRIES, "daily_deaths")
corr_deaths = deaths_pivot.corr()
sns.heatmap(corr_deaths, annot=True, fmt=".2f", cmap="Reds",
            linewidths=0.5, ax=axes[1], square=True,
            cbar_kws={"shrink": 0.8})
axes[1].set_title("Death Correlation Heatmap", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_multivariate_correlation.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m3_multivariate_correlation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3.4  TIME SERIES ANALYSIS — FUNDAMENTALS & CHARACTERISTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3.4 TIME SERIES ANALYSIS — FUNDAMENTALS")
print("─" * 60)

# 3.4.1 — Time-Based Indexing
print("\n  3.4.1 Time-Based Indexing:")
india_ts = india["daily_new_cases"].copy()
print(f"    Index type: {type(india_ts.index).__name__}")
print(f"    Frequency:  {india_ts.index.freq}")
india_ts = india_ts.asfreq("D", fill_value=0)
print(f"    After asfreq('D'): freq = {india_ts.index.freq}")

# Demonstrate time-based slicing
print(f"    Slice '2021-04':'2021-05' → {len(india_ts['2021-04':'2021-05'])} records")
print(f"    Slice '2020' → {len(india_ts['2020'])} records")

# 3.4.2 — Stationarity Test (ADF)
print("\n  3.4.2 Stationarity Test (Augmented Dickey-Fuller):")
adf_result = adfuller(india_ts.dropna(), autolag="AIC")
print(f"    ADF Statistic:  {adf_result[0]:.4f}")
print(f"    p-value:        {adf_result[1]:.6f}")
print(f"    Critical Values: {adf_result[4]}")
print(f"    Stationary?     {'Yes' if adf_result[1] < 0.05 else 'No (needs differencing)'}")

# First-difference for stationarity
india_diff = india_ts.diff().dropna()
adf_diff = adfuller(india_diff, autolag="AIC")
print(f"\n    After 1st differencing:")
print(f"    ADF Statistic: {adf_diff[0]:.4f}")
print(f"    p-value:       {adf_diff[1]:.6f}")
print(f"    Stationary?    {'Yes' if adf_diff[1] < 0.05 else 'No'}")

# 3.4.3 — Time Series Decomposition (Trend + Seasonal + Residual)
print("\n  3.4.3 Time Series Decomposition:")
decomp = seasonal_decompose(india_ts, model="additive", period=30)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle("Module 3 — Time Series Decomposition: India Daily Cases (period=30 days)",
             fontsize=14, fontweight="bold")

axes[0].plot(decomp.observed, color="#1d4ed8", linewidth=0.8)
axes[0].set_ylabel("Observed")
axes[0].set_title("Original Time Series")

axes[1].plot(decomp.trend, color="#dc2626", linewidth=1.5)
axes[1].set_ylabel("Trend")
axes[1].set_title("Trend Component")

axes[2].plot(decomp.seasonal, color="#16a34a", linewidth=0.8)
axes[2].set_ylabel("Seasonal")
axes[2].set_title("Seasonal Component (30-Day Cycle)")

axes[3].plot(decomp.resid, color="#9333ea", linewidth=0.5, alpha=0.7)
axes[3].set_ylabel("Residual")
axes[3].set_title("Residual (Noise)")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_time_series_decomposition.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m3_time_series_decomposition.png")

# 3.4.4 — ACF and PACF
print("\n  3.4.4 Autocorrelation (ACF) & Partial Autocorrelation (PACF):")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Module 3 — ACF & PACF: India Daily Cases",
             fontsize=14, fontweight="bold")

plot_acf(india_ts.dropna(), lags=60, ax=axes[0], color="#2563eb")
axes[0].set_title("Autocorrelation Function (ACF)")

plot_pacf(india_ts.dropna(), lags=60, ax=axes[1], color="#dc2626", method="ywm")
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_acf_pacf.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m3_acf_pacf.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3.5  GROUPING TIME SERIES DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3.5 GROUPING TIME SERIES DATA")
print("─" * 60)

# Monthly aggregation
monthly = india_ts.resample("ME").agg(["sum", "mean", "max"])
monthly.columns = ["Monthly Total", "Daily Avg", "Peak Day"]
print("  Monthly aggregation (top 10 months by total cases):")
print(monthly.nlargest(10, "Monthly Total").to_string())

# Weekly aggregation
weekly = india_ts.resample("W").sum()
print(f"\n  Weekly aggregation: {len(weekly)} weeks")

# Quarterly aggregation
quarterly = india_ts.resample("QE").sum()
print(f"  Quarterly aggregation: {len(quarterly)} quarters")

# Year-over-Year comparison
yearly = india_ts.resample("YE").agg(["sum", "mean", "max"])
yearly.columns = ["Annual Total", "Daily Avg", "Peak Day"]
yearly.index = yearly.index.year
print(f"\n  Year-over-Year Summary:")
print(yearly.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3.6  RESAMPLING TIME SERIES DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3.6 RESAMPLING TIME SERIES DATA")
print("─" * 60)

# Downsampling: daily → weekly (with different aggregation methods)
weekly_sum  = india_ts.resample("W").sum()
weekly_mean = india_ts.resample("W").mean()
weekly_max  = india_ts.resample("W").max()
print(f"  Downsampled daily → weekly: {len(india_ts)} days → {len(weekly_sum)} weeks")

# Upsampling: daily → 12-hourly (with interpolation)
hourly = india_ts.resample("12h").interpolate(method="linear")
print(f"  Upsampled daily → 12-hourly: {len(india_ts)} → {len(hourly)} records")

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Module 3 — Grouping & Resampling: India Daily Cases",
             fontsize=14, fontweight="bold")

# Monthly bar chart
monthly["Monthly Total"].plot(kind="bar", ax=axes[0, 0], color="#60a5fa",
                               edgecolor="white", width=0.8)
axes[0, 0].set_title("Monthly Total Cases")
axes[0, 0].set_ylabel("Total Cases")
axes[0, 0].tick_params(axis="x", rotation=45)
# Show every 3rd label
for i, label in enumerate(axes[0, 0].xaxis.get_ticklabels()):
    if i % 3 != 0:
        label.set_visible(False)

# Weekly comparison
axes[0, 1].plot(weekly_sum.index, weekly_sum, label="Weekly Sum",
                color="#2563eb", linewidth=1)
axes[0, 1].plot(weekly_mean.index, weekly_mean * 7, label="Weekly Mean × 7",
                color="#dc2626", linewidth=1, linestyle="--")
axes[0, 1].set_title("Weekly Downsampling Comparison")
axes[0, 1].legend()
axes[0, 1].set_ylabel("Cases")

# Year-over-year bar chart
yearly["Annual Total"].plot(kind="bar", ax=axes[1, 0],
                             color=["#93c5fd", "#3b82f6", "#1d4ed8", "#1e40af", "#172554"],
                             edgecolor="black")
axes[1, 0].set_title("Year-over-Year Total Cases")
axes[1, 0].set_ylabel("Total Cases")
axes[1, 0].tick_params(axis="x", rotation=0)
for bar in axes[1, 0].patches:
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{bar.get_height()/1e6:.1f}M", ha="center", va="bottom", fontsize=9)

# Quarterly
quarterly.plot(ax=axes[1, 1], color="#059669", linewidth=2, marker="o")
axes[1, 1].set_title("Quarterly Resampled Cases")
axes[1, 1].set_ylabel("Quarterly Total Cases")
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_grouping_resampling.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m3_grouping_resampling.png")

# ── Visualizing multi-country time series ────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
for country in ["India", "US", "Brazil", "United Kingdom", "Italy"]:
    country_data = load_country(country)["daily_new_cases"]
    country_ts = country_data.rolling(7).mean()
    ax.plot(country_ts, linewidth=1.5, label=country)

ax.set_title("Module 3 — Multi-Country 7-Day Avg Time Series Comparison",
             fontsize=14, fontweight="bold")
ax.set_ylabel("7-Day Avg Daily Cases")
ax.set_xlabel("Date")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m3_multi_country_timeseries.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m3_multi_country_timeseries.png")

print("\n" + "=" * 80)
print("MODULE 3 COMPLETE")
print("=" * 80)
