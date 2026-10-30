"""
Module 4: Data Summarization and Visualization
================================================
Syllabus Topics:
  - Statistical summary measures, data elaboration
  - 1-D Statistical data analysis
  - 2-D Statistical data analysis
  - Contingency tables
  - n-D Statistical data analysis
  - Visualization: Scatter plots, Dot charts, Bar plots
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
import seaborn as sns
from scipy import stats as sp_stats

from utils.data_loader import (load_timeseries, load_country, build_country_features,
                          get_multi_country_pivot, FOCUS_COUNTRIES, PLOTS_DIR)

print("=" * 80)
print("MODULE 4: DATA SUMMARIZATION AND VISUALIZATION")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 4.1  STATISTICAL SUMMARY MEASURES & DATA ELABORATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.1 STATISTICAL SUMMARY MEASURES")
print("─" * 60)

df = load_timeseries()
india = load_country("India")

# Central tendency, dispersion, shape
cases = india["daily_new_cases"]
deaths = india["daily_deaths"]

summary = {
    "Metric":     ["daily_new_cases", "daily_deaths"],
    "Count":      [cases.count(), deaths.count()],
    "Mean":       [cases.mean(), deaths.mean()],
    "Median":     [cases.median(), deaths.median()],
    "Mode":       [cases.mode().iloc[0], deaths.mode().iloc[0]],
    "Std Dev":    [cases.std(), deaths.std()],
    "Variance":   [cases.var(), deaths.var()],
    "Min":        [cases.min(), deaths.min()],
    "Max":        [cases.max(), deaths.max()],
    "Range":      [cases.max()-cases.min(), deaths.max()-deaths.min()],
    "Q1 (25%)":   [cases.quantile(0.25), deaths.quantile(0.25)],
    "Q2 (50%)":   [cases.quantile(0.50), deaths.quantile(0.50)],
    "Q3 (75%)":   [cases.quantile(0.75), deaths.quantile(0.75)],
    "IQR":        [cases.quantile(0.75) - cases.quantile(0.25),
                   deaths.quantile(0.75) - deaths.quantile(0.25)],
    "Skewness":   [cases.skew(), deaths.skew()],
    "Kurtosis":   [cases.kurtosis(), deaths.kurtosis()],
    "Coeff Var":  [cases.std()/cases.mean()*100, deaths.std()/deaths.mean()*100],
}
summary_df = pd.DataFrame(summary).set_index("Metric").T
print("\n  India — Statistical Summary:")
print(summary_df.to_string())

# Data Elaboration: Percentiles
print("\n  Percentile Profile (India daily_new_cases):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p:>2d}: {cases.quantile(p/100):>12,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.2  1-D STATISTICAL DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.2  1-D STATISTICAL DATA ANALYSIS")
print("─" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Module 4 — 1-D Statistical Analysis: India",
             fontsize=14, fontweight="bold")

# 4.2.1 Histogram
axes[0, 0].hist(cases, bins=50, color="#60a5fa", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(cases.mean(), color="#dc2626", linestyle="--", label=f"Mean = {cases.mean():,.0f}")
axes[0, 0].axvline(cases.median(), color="#16a34a", linestyle="-.", label=f"Median = {cases.median():,.0f}")
axes[0, 0].set_title("Histogram with Mean & Median")
axes[0, 0].set_xlabel("Daily Cases")
axes[0, 0].legend(fontsize=8)

# 4.2.2 Box Plot (cases + deaths side by side)
data_to_box = [cases.values, deaths.values]
try:
    bp = axes[0, 1].boxplot(data_to_box, tick_labels=["Cases", "Deaths"],
                              patch_artist=True, widths=0.5)
except TypeError:
    bp = axes[0, 1].boxplot(data_to_box, labels=["Cases", "Deaths"],
                              patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor("#bfdbfe")
bp["boxes"][1].set_facecolor("#fecaca")
axes[0, 1].set_title("Box Plot: Cases vs Deaths")
axes[0, 1].set_ylabel("Count")

# 4.2.3 Violin Plot
parts = axes[0, 2].violinplot([cases.values], positions=[1], showmeans=True,
                                showmedians=True)
parts["bodies"][0].set_facecolor("#c4b5fd")
axes[0, 2].set_title("Violin Plot: Daily Cases")
axes[0, 2].set_ylabel("Daily Cases")
axes[0, 2].set_xticks([1])
axes[0, 2].set_xticklabels(["India"])

# 4.2.4 CDF (Cumulative Distribution Function)
sorted_cases = np.sort(cases.values)
cdf = np.arange(1, len(sorted_cases) + 1) / len(sorted_cases)
axes[1, 0].plot(sorted_cases, cdf, color="#1d4ed8", linewidth=2)
axes[1, 0].axhline(0.5, color="#dc2626", linestyle="--", alpha=0.5, label="Median (50th %ile)")
axes[1, 0].set_title("Empirical CDF")
axes[1, 0].set_xlabel("Daily Cases")
axes[1, 0].set_ylabel("Cumulative Probability")
axes[1, 0].legend()

# 4.2.5 Stem Plot (weekly max)
weekly_max = cases.resample("W").max().iloc[-20:]
axes[1, 1].stem(weekly_max.index, weekly_max.values, linefmt="#2563eb",
                markerfmt="o", basefmt="gray")
axes[1, 1].set_title("Stem Plot: Weekly Peak Cases (Last 20 Weeks)")
axes[1, 1].set_ylabel("Peak Daily Cases")
axes[1, 1].tick_params(axis="x", rotation=30)

# 4.2.6 Dot chart (top 10 countries total cases)
features = build_country_features()
top10 = features.nlargest(10, "total_cases").sort_values("total_cases")
axes[1, 2].scatter(top10["total_cases"], top10["country"], color="#2563eb", s=80, zorder=5)
axes[1, 2].hlines(top10["country"], 0, top10["total_cases"], color="#94a3b8", linewidth=1)
axes[1, 2].set_title("Dot Chart: Top 10 Countries by Total Cases")
axes[1, 2].set_xlabel("Total Confirmed Cases")
for i, (_, row) in enumerate(top10.iterrows()):
    axes[1, 2].text(row["total_cases"], i, f"  {row['total_cases']/1e6:.1f}M", va="center", fontsize=8)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m4_1d_analysis.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m4_1d_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.3  2-D STATISTICAL DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.3  2-D STATISTICAL DATA ANALYSIS")
print("─" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Module 4 — 2-D Statistical Analysis",
             fontsize=14, fontweight="bold")

# 4.3.1 Scatter Plot: Total Cases vs Total Deaths by country
axes[0, 0].scatter(features["total_cases"], features["total_deaths"],
                    c=features["cfr_percent"], cmap="Reds", s=40, alpha=0.7,
                    edgecolors="black", linewidths=0.3)
axes[0, 0].set_xlabel("Total Cases")
axes[0, 0].set_ylabel("Total Deaths")
axes[0, 0].set_title("Scatter: Total Cases vs Deaths (color = CFR %)")
# Annotate top 5
for _, row in features.nlargest(5, "total_cases").iterrows():
    axes[0, 0].annotate(row["country"], (row["total_cases"], row["total_deaths"]),
                          fontsize=7, ha="left")

# 4.3.2 Joint Distribution (2D Density)
axes[0, 1].hexbin(india["daily_new_cases"], india["daily_deaths"],
                   gridsize=25, cmap="YlGnBu", mincnt=1)
axes[0, 1].set_xlabel("Daily Cases")
axes[0, 1].set_ylabel("Daily Deaths")
axes[0, 1].set_title("2D Density: India Cases vs Deaths")

# 4.3.3 Bubble Chart: Peak cases vs CFR (bubble size = total deaths)
top20 = features.nlargest(20, "total_cases")
scatter = axes[1, 0].scatter(
    top20["peak_daily_cases"], top20["cfr_percent"],
    s=top20["total_deaths"] / 5000, alpha=0.6, c="#3b82f6",
    edgecolors="black", linewidths=0.5
)
axes[1, 0].set_xlabel("Peak Daily Cases")
axes[1, 0].set_ylabel("Case Fatality Rate (%)")
axes[1, 0].set_title("Bubble Chart: Peak Cases vs CFR (size = Total Deaths)")
for _, row in top20.iterrows():
    axes[1, 0].annotate(row["country"], (row["peak_daily_cases"], row["cfr_percent"]),
                          fontsize=6)

# 4.3.4 Regression plot
x = features["total_cases"].values.reshape(-1)
y = features["total_deaths"].values.reshape(-1)
mask_valid = (x > 0) & (y > 0)
slope, intercept, r_val, p_val, std_err = sp_stats.linregress(np.log10(x[mask_valid]),
                                                                 np.log10(y[mask_valid]))
axes[1, 1].scatter(np.log10(x[mask_valid]), np.log10(y[mask_valid]),
                    alpha=0.5, s=15, c="#2563eb")
xfit = np.linspace(np.log10(x[mask_valid]).min(), np.log10(x[mask_valid]).max(), 100)
axes[1, 1].plot(xfit, slope * xfit + intercept, "r--", linewidth=2,
                label=f"R² = {r_val**2:.3f}")
axes[1, 1].set_xlabel("log₁₀(Total Cases)")
axes[1, 1].set_ylabel("log₁₀(Total Deaths)")
axes[1, 1].set_title("Log-Log Regression: Cases → Deaths")
axes[1, 1].legend()

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m4_2d_analysis.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m4_2d_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.4  CONTINGENCY TABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.4  CONTINGENCY TABLES")
print("─" * 60)

# Categorize daily cases and deaths into severity bins
india_ct = india.copy()
india_ct["case_severity"] = pd.cut(
    india_ct["daily_new_cases"],
    bins=[-1, 500, 10000, 100000, float("inf")],
    labels=["Minimal", "Moderate", "High", "Critical"]
)
india_ct["death_severity"] = pd.cut(
    india_ct["daily_deaths"],
    bins=[-1, 10, 100, 1000, float("inf")],
    labels=["Negligible", "Low", "Moderate", "Severe"]
)

contingency = pd.crosstab(india_ct["case_severity"], india_ct["death_severity"],
                            margins=True, margins_name="Total")
print("  Contingency Table: Case Severity × Death Severity (India)")
print(contingency.to_string())

# Chi-Square Test of Independence
ct_no_margins = pd.crosstab(india_ct["case_severity"], india_ct["death_severity"])
chi2, p_val, dof, expected = sp_stats.chi2_contingency(ct_no_margins)
print(f"\n  Chi-Square Test of Independence:")
print(f"    χ² = {chi2:,.2f}")
print(f"    p-value = {p_val:.2e}")
print(f"    DoF = {dof}")
print(f"    Result: {'Dependent (reject H₀)' if p_val < 0.05 else 'Independent (fail to reject H₀)'}")

# Cramér's V (effect size)
n = ct_no_margins.sum().sum()
min_dim = min(ct_no_margins.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"    Cramér's V = {cramers_v:.4f} (effect size)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Module 4 — Contingency Tables", fontsize=14, fontweight="bold")

sns.heatmap(ct_no_margins, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=axes[0])
axes[0].set_title("Observed Frequencies")
axes[0].set_ylabel("Case Severity")
axes[0].set_xlabel("Death Severity")

expected_df = pd.DataFrame(expected, index=ct_no_margins.index,
                             columns=ct_no_margins.columns)
sns.heatmap(expected_df, annot=True, fmt=".1f", cmap="YlGnBu",
            linewidths=0.5, ax=axes[1])
axes[1].set_title(f"Expected Frequencies (χ² = {chi2:,.1f}, p = {p_val:.2e})")
axes[1].set_ylabel("Case Severity")
axes[1].set_xlabel("Death Severity")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m4_contingency_tables.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m4_contingency_tables.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.5  n-D STATISTICAL DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.5  n-D STATISTICAL DATA ANALYSIS")
print("─" * 60)

top10_features = features.nlargest(10, "total_cases")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Module 4 — n-D Statistical Analysis",
             fontsize=14, fontweight="bold")

# 4.5.1 Parallel Coordinates Plot
from pandas.plotting import parallel_coordinates

norm_cols = ["total_cases", "total_deaths", "peak_daily_cases",
             "mean_daily_cases", "cfr_percent", "case_volatility"]
plot_data = top10_features[["country"] + norm_cols].copy()
for c in norm_cols:
    col_min = plot_data[c].min()
    col_max = plot_data[c].max()
    if col_max > col_min:
        plot_data[c] = (plot_data[c] - col_min) / (col_max - col_min)
    else:
        plot_data[c] = 0.0

parallel_coordinates(plot_data, "country", ax=axes[0], colormap="tab10", linewidth=2)
axes[0].set_title("Parallel Coordinates (Normalized Features)", fontweight="bold")
axes[0].set_ylabel("Normalized Value")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(loc="upper right", fontsize=6, ncol=2)

# 4.5.2 Radar / Spider Chart for top 5
categories = norm_cols
top5 = plot_data.head(5)
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax_radar = fig.add_subplot(122, polar=True)
for _, row in top5.iterrows():
    values = row[categories].values.tolist()
    values += values[:1]
    ax_radar.plot(angles, values, linewidth=2, label=row["country"])
    ax_radar.fill(angles, values, alpha=0.1)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=7)
ax_radar.set_title("Radar Chart: Top 5 Countries", fontweight="bold", pad=20)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

# Remove the second subplot from axes since we used add_subplot
axes[1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m4_nd_analysis.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m4_nd_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.6  BAR PLOTS & DOT CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.6  BAR PLOTS & DOT CHARTS")
print("─" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Module 4 — Bar Plots & Dot Charts",
             fontsize=14, fontweight="bold")

# 4.6.1 Grouped bar chart: Cases & Deaths by top 10 countries
top10_raw = features.nlargest(10, "total_cases").sort_values("total_cases", ascending=True)
y_pos = np.arange(len(top10_raw))
bar_height = 0.35

axes[0, 0].barh(y_pos - bar_height/2, top10_raw["total_cases"],
                 bar_height, label="Cases", color="#60a5fa")
axes[0, 0].barh(y_pos + bar_height/2, top10_raw["total_deaths"] * 50,
                 bar_height, label="Deaths × 50", color="#f87171")
axes[0, 0].set_yticks(y_pos)
axes[0, 0].set_yticklabels(top10_raw["country"])
axes[0, 0].set_title("Grouped Bar: Cases & Deaths (Top 10)")
axes[0, 0].legend()

# 4.6.2 Stacked Bar: Monthly cases and deaths for India
india_monthly = india[["daily_new_cases", "daily_deaths"]].resample("ME").sum()
india_monthly = india_monthly.iloc[-24:]  # Last 24 months
axes[0, 1].bar(india_monthly.index, india_monthly["daily_new_cases"],
                label="Cases", color="#60a5fa", width=20)
axes[0, 1].bar(india_monthly.index, india_monthly["daily_deaths"],
                bottom=india_monthly["daily_new_cases"],
                label="Deaths", color="#f87171", width=20)
axes[0, 1].set_title("Stacked Bar: Monthly Cases & Deaths (India)")
axes[0, 1].legend()
axes[0, 1].tick_params(axis="x", rotation=30)

# 4.6.3 CFR% Dot chart (Cleveland dot plot)
top15_cfr = features[features["total_cases"] > 100000].nlargest(15, "cfr_percent")
top15_cfr = top15_cfr.sort_values("cfr_percent")
axes[1, 0].scatter(top15_cfr["cfr_percent"], top15_cfr["country"],
                    color="#dc2626", s=80, zorder=5)
axes[1, 0].hlines(top15_cfr["country"], 0, top15_cfr["cfr_percent"],
                    color="#fca5a5", linewidth=2)
axes[1, 0].set_title("Dot Chart: Highest CFR% (Countries with >100K cases)")
axes[1, 0].set_xlabel("Case Fatality Rate (%)")
for _, row in top15_cfr.iterrows():
    axes[1, 0].text(row["cfr_percent"] + 0.1, row["country"],
                      f"{row['cfr_percent']:.1f}%", va="center", fontsize=8)

# 4.6.4 Comparative Bar: Peak daily cases
top10_peak = features.nlargest(10, "peak_daily_cases").sort_values("peak_daily_cases")
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top10_peak)))
axes[1, 1].barh(top10_peak["country"], top10_peak["peak_daily_cases"],
                 color=colors, edgecolor="black", linewidth=0.3)
axes[1, 1].set_title("Bar Chart: Highest Single-Day Case Count")
axes[1, 1].set_xlabel("Peak Daily Cases")
for i, (_, row) in enumerate(top10_peak.iterrows()):
    axes[1, 1].text(row["peak_daily_cases"], i,
                      f"  {row['peak_daily_cases']:,.0f}", va="center", fontsize=8)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m4_bar_dot_charts.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m4_bar_dot_charts.png")

print("\n" + "=" * 80)
print("MODULE 4 COMPLETE")
print("=" * 80)
