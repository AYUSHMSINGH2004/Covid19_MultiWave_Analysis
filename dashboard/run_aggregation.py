import pandas as pd
from pymongo import MongoClient

def get_smoothed_country_data(country_name="India"):
    client = MongoClient("mongodb://localhost:27017/")
    collection = client["covid_wave_db"]["daily_metrics"]

    pipeline = [
        # Stage 1: Filter by country
        {"$match": {"metadata.country": country_name}},

        # Stage 2: 7-day rolling average (-6 days to current day)
        {
            "$setWindowFields": {
                "partitionBy": "$metadata.country",
                "sortBy": {"timestamp": 1},
                "output": {
                    "smoothed_cases": {
                        "$avg": "$metrics.daily_new_cases",
                        "window": {"documents": [-6, 0]},
                    },
                    "smoothed_deaths": {
                        "$avg": "$metrics.daily_deaths",
                        "window": {"documents": [-6, 0]},
                    },
                },
            }
        },

        # Stage 3: Format clean output fields
        {
            "$project": {
                "_id": 0,
                "date": "$timestamp",
                "country": "$metadata.country",
                "daily_new_cases": "$metrics.daily_new_cases",
                "smoothed_cases": {"$round": ["$smoothed_cases", 2]},
                "daily_deaths": "$metrics.daily_deaths",
                "smoothed_deaths": {"$round": ["$smoothed_deaths", 2]},
            }
        },
    ]

    cursor = collection.aggregate(pipeline)
    return pd.DataFrame(list(cursor))

if __name__ == "__main__":
    country = "India"
    df = get_smoothed_country_data(country)
    print(f"Aggregated records retrieved for {country}: {len(df)}")
    print("\nSample during peak wave period (May 2021):")
    sample = df[(df["date"] >= "2021-05-01") & (df["date"] <= "2021-05-10")]
    print(sample[["date", "daily_new_cases", "smoothed_cases", "daily_deaths", "smoothed_deaths"]].to_string(index=False))