"""
Module 2: Data Transformation
==============================
Syllabus Topics:
  - Performing data deduplication
  - Replacing values
  - Discretization and binning
  - Introduction to missing data & handling missing data
  - Traditional methods (forward fill, interpolation, mean imputation)
  - Maximum Likelihood Estimation (EM-based imputation)
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
from scipy.stats import norm
from sklearn.impute import SimpleImputer

from utils.data_loader import load_timeseries, load_country, PLOTS_DIR

print("=" * 80)
print("MODULE 2: DATA TRANSFORMATION")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  DATA DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2.1 DATA DEDUPLICATION")
print("─" * 60)

df = load_timeseries()
print(f"  Original dataset shape: {df.shape}")

# Check for exact duplicate rows
dup_count = df.duplicated().sum()
print(f"  Exact duplicate rows found: {dup_count}")

# Introduce artificial duplicates for demonstration
df_with_dups = pd.concat([df, df.sample(n=500, random_state=42)], ignore_index=True)
print(f"  After injecting 500 duplicates: {df_with_dups.shape}")
dup_count_after = df_with_dups.duplicated().sum()
print(f"  Duplicates detected: {dup_count_after}")

# Remove duplicates
df_deduped = df_with_dups.drop_duplicates().reset_index(drop=True)
print(f"  After deduplication: {df_deduped.shape}")

# Check for duplicate (country, date) pairs (logical duplicates)
logical_dups = df.duplicated(subset=["country", "Date"]).sum()
print(f"  Logical duplicates (same country+date): {logical_dups}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  REPLACING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2.2 REPLACING VALUES")
print("─" * 60)

india = load_country("India").copy()

# 2.2.1 — Replace zero-case early days with NaN (no reporting)
early_zeros = (india["daily_new_cases"] == 0).sum()
print(f"  India: {early_zeros} days with zero new cases")

# 2.2.2 — Replace extreme outlier values (> 3 standard deviations) with median
cases = india["daily_new_cases"].copy()
mean_val = cases.mean()
std_val = cases.std()
upper_threshold = mean_val + 3 * std_val
outliers_mask = cases > upper_threshold
outlier_count = outliers_mask.sum()
print(f"  Outliers (> 3σ = {upper_threshold:,.0f}): {outlier_count} days")

cases_cleaned = cases.copy()
cases_cleaned[outliers_mask] = cases.median()
print(f"  Replaced {outlier_count} outlier values with median = {cases.median():,.0f}")

# 2.2.3 — Categorical replacement
india["severity"] = pd.cut(
    india["daily_new_cases"],
    bins=[-1, 100, 10000, 100000, float("inf")],
    labels=["Negligible", "Low", "Moderate", "Severe"]
)
severity_counts = india["severity"].value_counts()
print(f"\n  Severity categories after value replacement:")
for cat, count in severity_counts.items():
    print(f"    {cat}: {count} days")

# ─────────────────────────────────────────────────────────────────────────────
# 2.3  DISCRETIZATION AND BINNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2.3 DISCRETIZATION AND BINNING")
print("─" * 60)

# 2.3.1 — Equal-Width Binning
india["cases_eq_width"] = pd.cut(india["daily_new_cases"], bins=5,
                                  labels=["Very Low", "Low", "Medium", "High", "Very High"])
print("  Equal-Width Binning (5 bins):")
print(india["cases_eq_width"].value_counts().to_string())

# 2.3.2 — Equal-Frequency (Quantile) Binning
india["cases_eq_freq"] = pd.qcut(india["daily_new_cases"].rank(method="first"),
                                  q=4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\n  Equal-Frequency Binning (4 quantiles):")
print(india["cases_eq_freq"].value_counts().to_string())

# 2.3.3 — Custom Domain-Specific Binning
custom_bins = [0, 500, 5000, 50000, 200000, float("inf")]
custom_labels = ["Minimal", "Controlled", "Spreading", "Surge", "Crisis"]
india["cases_custom"] = pd.cut(india["daily_new_cases"], bins=custom_bins,
                                labels=custom_labels, include_lowest=True)
print("\n  Domain-Specific Binning:")
print(india["cases_custom"].value_counts().to_string())

# Plot binning distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Module 2: Discretization & Binning Distributions — India",
             fontsize=14, fontweight="bold")

for ax, col, title in zip(
    axes,
    ["cases_eq_width", "cases_eq_freq", "cases_custom"],
    ["Equal-Width (5 bins)", "Equal-Frequency (4 quantiles)", "Domain-Specific (5 bins)"]
):
    counts = india[col].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(counts)),
                edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Number of Days")
    ax.tick_params(axis="x", rotation=30)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                int(bar.get_height()), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m2_binning_distributions.png"), dpi=200)
plt.close()
print(f"\n  ✓ Saved: plots/m2_binning_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2.4  HANDLING MISSING DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2.4 HANDLING MISSING DATA")
print("─" * 60)

# Create a copy with artificially introduced missing values
india_miss = india[["daily_new_cases", "daily_deaths"]].copy()
np.random.seed(42)
mask = np.random.random(india_miss.shape) < 0.10  # 10% missing
india_miss = india_miss.mask(mask)
missing_pct = india_miss.isnull().mean() * 100
print(f"  Artificially introduced ~10% missing data:")
print(f"    daily_new_cases missing: {missing_pct['daily_new_cases']:.1f}%")
print(f"    daily_deaths missing:    {missing_pct['daily_deaths']:.1f}%")

# 2.4.1 — Traditional Method: Mean/Median Imputation
mean_imputed = india_miss.fillna(india_miss.mean())
median_imputed = india_miss.fillna(india_miss.median())
print(f"\n  Mean Imputation — fill value for cases: {india_miss['daily_new_cases'].mean():,.0f}")
print(f"  Median Imputation — fill value for cases: {india_miss['daily_new_cases'].median():,.0f}")

# 2.4.2 — Forward Fill / Backward Fill
ffill_imputed = india_miss.ffill()
bfill_imputed = india_miss.bfill()
fbfill_imputed = india_miss.ffill().bfill()  # Handle leading NaNs
print(f"  Forward Fill — remaining NaNs: {ffill_imputed.isnull().sum().sum()}")
print(f"  Forward+Backward Fill — remaining NaNs: {fbfill_imputed.isnull().sum().sum()}")

# 2.4.3 — Linear Interpolation
interp_imputed = india_miss.interpolate(method="linear").bfill()
print(f"  Linear Interpolation — remaining NaNs: {interp_imputed.isnull().sum().sum()}")

# 2.4.4 — Maximum Likelihood Estimation (EM-style)
print("\n  2.4.4 Maximum Likelihood Estimation (EM-based Imputation):")
def em_impute(data, max_iter=50, tol=1e-4):
    """Simplified EM imputation for univariate normal data.
    E-step: fill missing with current mean estimate.
    M-step: re-estimate mean and variance from filled data."""
    col = data.copy()
    observed = col.dropna()
    mu = observed.mean()
    sigma = observed.std()

    for iteration in range(max_iter):
        # E-step: impute missing values with current mean
        filled = col.fillna(mu)
        # M-step: re-estimate parameters from complete data
        mu_new = filled.mean()
        sigma_new = filled.std()
        # Check convergence
        if abs(mu_new - mu) < tol and abs(sigma_new - sigma) < tol:
            print(f"    EM converged at iteration {iteration + 1}: "
                  f"μ = {mu_new:,.2f}, σ = {sigma_new:,.2f}")
            return filled, mu_new, sigma_new
        mu, sigma = mu_new, sigma_new

    print(f"    EM stopped at max iterations: μ = {mu:,.2f}, σ = {sigma:,.2f}")
    return col.fillna(mu), mu, sigma

mle_cases, mu_mle, sigma_mle = em_impute(india_miss["daily_new_cases"])
print(f"    MLE estimated mean: {mu_mle:,.2f}")
print(f"    MLE estimated std:  {sigma_mle:,.2f}")

# Comparison plot: original vs imputation methods
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("Module 2: Missing Data Imputation Methods — India Daily Cases",
             fontsize=14, fontweight="bold")

original = india["daily_new_cases"]
date_idx = india.index

methods = {
    "Original (Ground Truth)":   original,
    "Data with 10% Missing":     india_miss["daily_new_cases"],
    "Mean Imputation":           mean_imputed["daily_new_cases"],
    "Forward Fill":              fbfill_imputed["daily_new_cases"],
    "Linear Interpolation":      interp_imputed["daily_new_cases"],
    "MLE (EM) Imputation":       mle_cases,
}

for ax, (title, series) in zip(axes.flatten(), methods.items()):
    ax.plot(date_idx, series, linewidth=0.8, color="#2563eb" if "Original" in title
            else "#dc2626" if "Missing" in title else "#16a34a")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("Daily Cases")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m2_missing_data_imputation.png"), dpi=200)
plt.close()
print(f"\n  ✓ Saved: plots/m2_missing_data_imputation.png")

# ── Imputation Quality Metrics ───────────────────────────────────────────────
print("\n  Imputation Quality (RMSE vs ground truth on missing positions):")
missing_idx = india_miss["daily_new_cases"].isnull()
true_vals = original[missing_idx]
for name, imputed in [("Mean", mean_imputed["daily_new_cases"]),
                       ("Forward Fill", fbfill_imputed["daily_new_cases"]),
                       ("Interpolation", interp_imputed["daily_new_cases"]),
                       ("MLE (EM)", mle_cases)]:
    rmse = np.sqrt(((true_vals - imputed[missing_idx]) ** 2).mean())
    print(f"    {name:20s} RMSE = {rmse:>12,.2f}")

print("\n" + "=" * 80)
print("MODULE 2 COMPLETE")
print("=" * 80)
