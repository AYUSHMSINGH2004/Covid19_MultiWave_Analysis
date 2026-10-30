import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from pymongo import MongoClient
from run_aggregation import get_smoothed_country_data

def analyze_and_plot_waves(country_name="India"):
    print(f"1. Fetching smoothed time-series from MongoDB for {country_name}...")
    df = get_smoothed_country_data(country_name)
    if df.empty:
        print(f"No records found for '{country_name}'. Check the country spelling.")
        return

    # Detect distinct wave peaks on the 7-day smoothed cases curve
    # distance=60 ensures peaks are at least 60 days apart
    # prominence=4% of max peak filters out minor reporting bumps
    max_smoothed = df["smoothed_cases"].max()
    peaks, properties = find_peaks(
        df["smoothed_cases"].to_numpy(),
        distance=60,
        prominence=max_smoothed * 0.04
    )

    # Detect corresponding peaks on the 7-day smoothed deaths curve
    max_deaths = df["smoothed_deaths"].max()
    death_peaks, _ = find_peaks(
        df["smoothed_deaths"].to_numpy(),
        distance=60,
        prominence=max_deaths * 0.04
    )

    print(f"\n2. Detected {len(peaks)} Major Epidemiological Waves for {country_name}:")
    print("-" * 72)
    print(f"{'Wave':<8} | {'Peak Date':<12} | {'7-Day Avg Cases':<18} | {'Raw Daily Cases':<16}")
    print("-" * 72)

    wave_documents = []
    for idx, peak_idx in enumerate(peaks, start=1):
        row = df.iloc[peak_idx]
        peak_date_str = row["date"].strftime("%Y-%m-%d")
        print(
            f"Wave {idx:<3} | {peak_date_str:<12} | "
            f"{row['smoothed_cases']:>18,.2f} | {row['daily_new_cases']:>16,}"
        )
        wave_documents.append({
            "country": country_name,
            "wave_number": idx,
            "peak_date": row["date"].to_pydatetime(),
            "peak_smoothed_cases": float(row["smoothed_cases"]),
            "peak_daily_cases": int(row["daily_new_cases"]),
            "prominence": round(float(properties["prominences"][idx - 1]), 2)
        })
    print("-" * 72)

    # Save the detected waves back to MongoDB for reporting
    client = MongoClient("mongodb://localhost:27017/")
    waves_col = client["covid_wave_db"]["detected_waves"]
    waves_col.delete_many({"country": country_name})
    if wave_documents:
        waves_col.insert_many(wave_documents)
        print(f"3. Saved {len(wave_documents)} wave summary records to 'covid_wave_db.detected_waves' in MongoDB.")

    # Plot the Multi-Wave Analysis Chart
    print("4. Generating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"COVID-19 Multi-Wave Epidemiological Analysis: {country_name}", fontsize=15, fontweight="bold")

    # Top Panel: Daily Cases & Wave Peaks
    ax1.bar(df["date"], df["daily_new_cases"], color="#cbd5e1", alpha=0.6, label="Raw Daily Cases")
    ax1.plot(df["date"], df["smoothed_cases"], color="#2563eb", linewidth=2.2, label="7-Day Moving Avg (MongoDB)")
    ax1.scatter(df["date"].iloc[peaks], df["smoothed_cases"].iloc[peaks], color="#dc2626", s=70, zorder=5, label="Detected Wave Peak")

    for idx, peak_idx in enumerate(peaks, start=1):
        peak_date = df["date"].iloc[peak_idx]
        peak_val = df["smoothed_cases"].iloc[peak_idx]
        ax1.annotate(
            f"Wave {idx}\n({peak_date.strftime('%b %Y')})\n{int(peak_val):,}/day",
            xy=(peak_date, peak_val),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#dc2626", lw=1)
        )

    ax1.set_ylabel("Daily Confirmed Cases", fontweight="bold")
    ax1.set_ylim(0, max_smoothed * 1.28)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Bottom Panel: Daily Deaths & Mortality Peaks
    ax2.bar(df["date"], df["daily_deaths"], color="#fecaca", alpha=0.6, label="Raw Daily Deaths")
    ax2.plot(df["date"], df["smoothed_deaths"], color="#991b1b", linewidth=2, label="7-Day Moving Avg Deaths")
    ax2.scatter(df["date"].iloc[death_peaks], df["smoothed_deaths"].iloc[death_peaks], color="#000000", s=55, zorder=5, label="Mortality Peak")

    ax2.set_ylabel("Daily Reported Deaths", fontweight="bold")
    ax2.set_xlabel("Timeline", fontweight="bold")
    ax2.set_ylim(0, max_deaths * 1.25)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE_DIR, "plots", "legacy_waves")
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"covid_waves_{country_name.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300)
    print(f"5. Saved high-resolution chart to '{output_file}'. Opening plot window...")
    plt.show()

if __name__ == "__main__":
    target_country = sys.argv[1] if len(sys.argv) > 1 else "India"
    analyze_and_plot_waves(target_country)