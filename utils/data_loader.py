"""
data_loader.py – Shared Data Loading Utility
=============================================
Loads the JHU CSSE COVID-19 time-series CSVs, reshapes them into a tidy
chronological DataFrame, and provides helper functions consumed by every
module script (Modules 2-7).
"""

import os
import sys

# Ensure UTF-8 output across Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

# Resolve project root regardless of where this module is imported from
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_CUR_DIR) in ("utils", "modules", "dashboard"):
    PROJECT_ROOT = os.path.dirname(_CUR_DIR)
else:
    PROJECT_ROOT = _CUR_DIR

BASE_DIR = PROJECT_ROOT
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Core Data Loading ────────────────────────────────────────────────────────

def _resolve_data_path(filename: str) -> str:
    """Find CSV file in data/ or root directory."""
    candidates = [
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(os.path.dirname(BASE_DIR), "data", filename),
        os.path.join(os.path.dirname(BASE_DIR), filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_timeseries() -> pd.DataFrame:
    """Load and merge the confirmed-cases and deaths CSVs into a single
    tidy DataFrame with daily and cumulative metrics per country per date."""

    confirmed_path = _resolve_data_path("time_series_covid19_confirmed_global.csv")
    deaths_path    = _resolve_data_path("time_series_covid19_deaths_global.csv")

    confirmed_df = pd.read_csv(confirmed_path)
    deaths_df    = pd.read_csv(deaths_path)

    drop_cols = ["Province/State", "Lat", "Long"]
    confirmed_country = confirmed_df.drop(columns=drop_cols).groupby("Country/Region", as_index=False).sum()
    deaths_country    = deaths_df.drop(columns=drop_cols).groupby("Country/Region", as_index=False).sum()

    confirmed_melted = confirmed_country.melt(
        id_vars=["Country/Region"], var_name="Date", value_name="cumulative_cases"
    )
    deaths_melted = deaths_country.melt(
        id_vars=["Country/Region"], var_name="Date", value_name="cumulative_deaths"
    )

    merged = pd.merge(confirmed_melted, deaths_melted, on=["Country/Region", "Date"])
    merged["Date"] = pd.to_datetime(merged["Date"], format="%m/%d/%y")
    merged = merged.sort_values(["Country/Region", "Date"]).reset_index(drop=True)

    # Daily increments (clip negative reporting corrections to 0)
    merged["daily_new_cases"] = (
        merged.groupby("Country/Region")["cumulative_cases"]
        .diff().fillna(merged["cumulative_cases"]).clip(lower=0)
    )
    merged["daily_deaths"] = (
        merged.groupby("Country/Region")["cumulative_deaths"]
        .diff().fillna(merged["cumulative_deaths"]).clip(lower=0)
    )

    merged.rename(columns={"Country/Region": "country"}, inplace=True)
    return merged


def load_country(country_name: str = "India") -> pd.DataFrame:
    """Return time-series data for a single country with date-indexed rows."""
    df = load_timeseries()
    cdf = df[df["country"] == country_name].copy()
    cdf = cdf.set_index("Date").sort_index()
    return cdf


def build_country_features() -> pd.DataFrame:
    """Build a feature matrix with one row per country and aggregate
    epidemiological metrics (for clustering & dimensionality reduction)."""
    df = load_timeseries()

    features = df.groupby("country").agg(
        total_cases      = ("cumulative_cases", "max"),
        total_deaths     = ("cumulative_deaths", "max"),
        peak_daily_cases = ("daily_new_cases", "max"),
        peak_daily_deaths= ("daily_deaths", "max"),
        mean_daily_cases = ("daily_new_cases", "mean"),
        mean_daily_deaths= ("daily_deaths", "mean"),
        std_daily_cases  = ("daily_new_cases", "std"),
        std_daily_deaths = ("daily_deaths", "std"),
        days_recorded    = ("daily_new_cases", "count"),
    ).reset_index()

    features["cfr_percent"] = np.where(
        features["total_cases"] > 0,
        (features["total_deaths"] / features["total_cases"]) * 100,
        0.0
    )
    features["case_volatility"] = np.where(
        features["mean_daily_cases"] > 0,
        features["std_daily_cases"] / features["mean_daily_cases"],
        0.0
    )

    return features


# ── Utility ──────────────────────────────────────────────────────────────────

FOCUS_COUNTRIES = ["India", "US", "Brazil", "United Kingdom", "Italy",
                   "France", "Germany", "Russia", "Japan", "South Africa"]

def get_multi_country_pivot(countries=None, metric="daily_new_cases"):
    """Pivot table: rows = dates, columns = countries, values = metric."""
    if countries is None:
        countries = FOCUS_COUNTRIES
    df = load_timeseries()
    sub = df[df["country"].isin(countries)]
    pivot = sub.pivot_table(index="Date", columns="country", values=metric)
    return pivot.fillna(0)


if __name__ == "__main__":
    df = load_timeseries()
    print(f"Total records: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nSample:\n{df.head()}")
