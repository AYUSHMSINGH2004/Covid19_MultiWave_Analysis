import os
import pandas as pd
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
confirmed_path = os.path.join(BASE_DIR, "data", "time_series_covid19_confirmed_global.csv")
if not os.path.exists(confirmed_path):
    confirmed_path = "time_series_covid19_confirmed_global.csv"

deaths_path = os.path.join(BASE_DIR, "data", "time_series_covid19_deaths_global.csv")
if not os.path.exists(deaths_path):
    deaths_path = "time_series_covid19_deaths_global.csv"

print("1. Loading raw CSV datasets...")
confirmed_df = pd.read_csv(confirmed_path)
deaths_df = pd.read_csv(deaths_path)

# Drop geographic sub-columns and sum provinces/states by Country/Region
drop_cols = ["Province/State", "Lat", "Long"]
confirmed_country = confirmed_df.drop(columns=drop_cols).groupby("Country/Region", as_index=False).sum()
deaths_country = deaths_df.drop(columns=drop_cols).groupby("Country/Region", as_index=False).sum()

print("2. Reshaping from wide format to chronological time-series...")
confirmed_melted = confirmed_country.melt(
    id_vars=["Country/Region"], var_name="Date", value_name="cumulative_cases"
)
deaths_melted = deaths_country.melt(
    id_vars=["Country/Region"], var_name="Date", value_name="cumulative_deaths"
)

# Merge confirmed cases and deaths
merged = pd.merge(confirmed_melted, deaths_melted, on=["Country/Region", "Date"])
merged["Date"] = pd.to_datetime(merged["Date"], format="%m/%d/%y")
merged = merged.sort_values(["Country/Region", "Date"]).reset_index(drop=True)

print("3. Calculating daily new cases and deaths...")
# Subtract previous day's cumulative total per country and clip negative reporting revisions at 0
merged["daily_new_cases"] = (
    merged.groupby("Country/Region")["cumulative_cases"].diff().fillna(merged["cumulative_cases"]).clip(lower=0)
)
merged["daily_deaths"] = (
    merged.groupby("Country/Region")["cumulative_deaths"].diff().fillna(merged["cumulative_deaths"]).clip(lower=0)
)

print("4. Connecting to local MongoDB...")
client = MongoClient("mongodb://localhost:27017/")
db = client["covid_wave_db"]

# Drop existing collection if re-running the script
if "daily_metrics" in db.list_collection_names():
    db.drop_collection("daily_metrics")

# Create optimized MongoDB Time Series Collection
db.create_collection(
    "daily_metrics",
    timeseries={
        "timeField": "timestamp",
        "metaField": "metadata",
        "granularity": "hours"
    }
)
collection = db["daily_metrics"]

print("5. Formatting BSON documents and bulk inserting...")
documents = [
    {
        "timestamp": dt.to_pydatetime(),
        "metadata": {"country": country},
        "metrics": {
            "daily_new_cases": int(new_c),
            "daily_deaths": int(new_d),
            "cumulative_cases": int(cum_c),
            "cumulative_deaths": int(cum_d),
        },
    }
    for country, dt, new_c, new_d, cum_c, cum_d in zip(
        merged["Country/Region"],
        merged["Date"],
        merged["daily_new_cases"],
        merged["daily_deaths"],
        merged["cumulative_cases"],
        merged["cumulative_deaths"],
    )
]

# Insert in batches of 50,000 for optimal performance
batch_size = 50000
for i in range(0, len(documents), batch_size):
    collection.insert_many(documents[i : i + batch_size])
    print(f"   Inserted {min(i + batch_size, len(documents))} / {len(documents)} records...")

# Create a compound index on metadata.country and timestamp to speed up wave queries
collection.create_index([("metadata.country", 1), ("timestamp", 1)])

print("Done! Total documents in MongoDB:", collection.count_documents({}))