import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, peak_widths
from pymongo import MongoClient

# Page Configuration
st.set_page_config(
    page_title="COVID-19 Multi-Wave & CFR Analytics (MongoDB)",
    layout="wide"
)

@st.cache_resource
def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    return client["covid_wave_db"]

db = get_db()
metrics_col = db["daily_metrics"]
waves_col = db["detected_waves"]

@st.cache_data
def get_country_list():
    countries = metrics_col.distinct("metadata.country")
    return sorted(countries)

def fetch_smoothed_data(country_name, window_days):
    pipeline = [
        {"$match": {"metadata.country": country_name}},
        {
            "$setWindowFields": {
                "partitionBy": "$metadata.country",
                "sortBy": {"timestamp": 1},
                "output": {
                    "smoothed_cases": {
                        "$avg": "$metrics.daily_new_cases",
                        "window": {"documents": [-(window_days - 1), 0]},
                    },
                    "smoothed_deaths": {
                        "$avg": "$metrics.daily_deaths",
                        "window": {"documents": [-(window_days - 1), 0]},
                    },
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": "$timestamp",
                "country": "$metadata.country",
                "daily_new_cases": "$metrics.daily_new_cases",
                "smoothed_cases": {"$round": ["$smoothed_cases", 2]},
                "daily_deaths": "$metrics.daily_deaths",
                "smoothed_deaths": {"$round": ["$smoothed_deaths", 2]},
                "cumulative_cases": "$metrics.cumulative_cases",
                "cumulative_deaths": "$metrics.cumulative_deaths",
            }
        },
    ]
    return pd.DataFrame(list(metrics_col.aggregate(pipeline)))

# Sidebar Controls
st.sidebar.header("MongoDB & Wave Controls")
countries = get_country_list()
default_idx = countries.index("India") if "India" in countries else 0

selected_country = st.sidebar.selectbox("Select Country / Region", countries, index=default_idx)
window_days = st.sidebar.slider("MongoDB Rolling Avg Window (Days)", min_value=3, max_value=21, value=7)
min_distance = st.sidebar.slider("Min Days Between Wave Peaks", min_value=30, max_value=120, value=60)
prominence_pct = st.sidebar.slider("Wave Prominence (% of Max Peak)", min_value=1, max_value=20, value=4)
rel_height = st.sidebar.slider("Wave Base Cutoff (% of Peak Height)", min_value=70, max_value=95, value=85) / 100.0

# Fetch Data from MongoDB
df = fetch_smoothed_data(selected_country, window_days)
y_cases = df["smoothed_cases"].to_numpy()
max_smoothed = y_cases.max()

# 1. Detect Wave Peaks
peaks, properties = find_peaks(
    y_cases,
    distance=min_distance,
    prominence=max(max_smoothed * (prominence_pct / 100.0), 1)
)

# 2. Calculate Wave Start & End Boundaries via Peak Widths
widths, width_heights, left_ips, right_ips = peak_widths(
    y_cases, peaks, rel_height=rel_height
)

max_deaths = df["smoothed_deaths"].max()
death_peaks, _ = find_peaks(
    df["smoothed_deaths"].to_numpy(),
    distance=min_distance,
    prominence=max(max_deaths * (prominence_pct / 100.0), 1)
)

# 3. Build Enriched Wave Metrics (Duration, Total Cases/Deaths, CFR %)
wave_records = []
n_rows = len(df)

for idx, peak_idx in enumerate(peaks):
    start_idx = max(0, int(np.floor(left_ips[idx])))
    end_idx = min(n_rows - 1, int(np.ceil(right_ips[idx])))

    # Prevent overlapping boundaries between consecutive waves using the lowest trough between peaks
    if idx > 0:
        prev_peak = peaks[idx - 1]
        trough_left = prev_peak + int(np.argmin(y_cases[prev_peak:peak_idx]))
        start_idx = max(start_idx, trough_left)
    if idx < len(peaks) - 1:
        next_peak = peaks[idx + 1]
        trough_right = peak_idx + int(np.argmin(y_cases[peak_idx:next_peak]))
        end_idx = min(end_idx, trough_right)

    wave_slice = df.iloc[start_idx : end_idx + 1]
    peak_row = df.iloc[peak_idx]

    start_date = df["date"].iloc[start_idx]
    end_date = df["date"].iloc[end_idx]
    duration_days = int((end_date - start_date).days) + 1

    wave_cases = int(wave_slice["daily_new_cases"].sum())
    wave_deaths = int(wave_slice["daily_deaths"].sum())
    cfr_pct = round((wave_deaths / wave_cases) * 100, 2) if wave_cases > 0 else 0.0

    wave_records.append({
        "country": selected_country,
        "wave_number": idx + 1,
        "start_date": start_date.to_pydatetime(),
        "peak_date": peak_row["date"].to_pydatetime(),
        "end_date": end_date.to_pydatetime(),
        "duration_days": duration_days,
        "peak_smoothed_cases": float(peak_row["smoothed_cases"]),
        "peak_daily_cases": int(peak_row["daily_new_cases"]),
        "total_wave_cases": wave_cases,
        "total_wave_deaths": wave_deaths,
        "cfr_percent": cfr_pct,
    })

# Save Enriched Wave Records to MongoDB
waves_col.delete_many({"country": selected_country})
if wave_records:
    waves_col.insert_many(wave_records)

# Top Header & Summary KPI Cards
st.title(f"COVID-19 Multi-Wave & Fatality Analysis: {selected_country}")
st.caption("Powered by MongoDB Time Series (`$setWindowFields`) & SciPy Peak/Width Detection")

total_cases = int(df["cumulative_cases"].max())
total_deaths = int(df["cumulative_deaths"].max())
overall_cfr = round((total_deaths / total_cases) * 100, 2) if total_cases > 0 else 0.0
avg_duration = int(np.mean([w["duration_days"] for w in wave_records])) if wave_records else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Confirmed Cases", f"{total_cases:,}")
col2.metric("Total Reported Deaths", f"{total_deaths:,}")
col3.metric("Overall Country CFR", f"{overall_cfr}%")
col4.metric("Detected Waves", f"{len(peaks)}")
col5.metric("Avg Wave Duration", f"{avg_duration} days")

# Interactive Dual-Panel Timeline with Shaded Wave Spans
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        f"Daily Cases, {window_days}-Day Moving Avg & Shaded Wave Spans",
        f"Daily Deaths & {window_days}-Day Moving Avg"
    )
)

# Panel 1: Cases
fig.add_trace(
    go.Bar(x=df["date"], y=df["daily_new_cases"], name="Raw Daily Cases", marker_color="#cbd5e1", opacity=0.55),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df["date"], y=df["smoothed_cases"], name=f"{window_days}-Day Avg Cases", line=dict(color="#2563eb", width=2.5)),
    row=1, col=1
)
if len(peaks) > 0:
    fig.add_trace(
        go.Scatter(
            x=df["date"].iloc[peaks],
            y=df["smoothed_cases"].iloc[peaks],
            mode="markers+text",
            name="Wave Peak",
            text=[f"Wave {w['wave_number']} ({w['cfr_percent']}% CFR)" for w in wave_records],
            textposition="top center",
            marker=dict(color="#dc2626", size=11)
        ),
        row=1, col=1
    )

# Shade each detected wave's Start-to-End window
for w in wave_records:
    fig.add_vrect(
        x0=w["start_date"],
        x1=w["end_date"],
        fillcolor="#3b82f6",
        opacity=0.10,
        layer="below",
        line_width=1,
        line_dash="dot",
        line_color="#1d4ed8",
    )

# Panel 2: Deaths
fig.add_trace(
    go.Bar(x=df["date"], y=df["daily_deaths"], name="Raw Daily Deaths", marker_color="#fecaca", opacity=0.55),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df["date"], y=df["smoothed_deaths"], name=f"{window_days}-Day Avg Deaths", line=dict(color="#991b1b", width=2.5)),
    row=2, col=1
)
if len(death_peaks) > 0:
    fig.add_trace(
        go.Scatter(
            x=df["date"].iloc[death_peaks],
            y=df["smoothed_deaths"].iloc[death_peaks],
            mode="markers",
            name="Mortality Peak",
            marker=dict(color="#000000", size=9)
        ),
        row=2, col=1
    )

fig.update_layout(height=680, hovermode="x unified", template="plotly_white", barmode="overlay")
st.plotly_chart(fig, use_container_width=True)

# Wave Comparison Charts & Tables
if wave_records:
    wave_df = pd.DataFrame(wave_records).drop(columns=["_id"], errors="ignore")
    for col in ["start_date", "peak_date", "end_date"]:
        wave_df[col] = pd.to_datetime(wave_df[col]).dt.strftime("%Y-%m-%d")

    wave_df["Wave Label"] = "Wave " + wave_df["wave_number"].astype(str)

    c1, c2 = st.columns(2)
    with c1:
        cfr_fig = px.bar(
            wave_df,
            x="Wave Label",
            y="cfr_percent",
            text="cfr_percent",
            color="cfr_percent",
            color_continuous_scale="Reds",
            title=f"Case Fatality Rate (CFR %) by Wave in {selected_country}",
            labels={"cfr_percent": "CFR (%)"}
        )
        cfr_fig.update_traces(texttemplate="%{text}%", textposition="outside")
        cfr_fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(cfr_fig, use_container_width=True)

    with c2:
        dur_fig = px.bar(
            wave_df,
            x="Wave Label",
            y="duration_days",
            text="duration_days",
            color="total_wave_cases",
            color_continuous_scale="Blues",
            title=f"Wave Duration (Days) & Total Cases in {selected_country}",
            labels={"duration_days": "Duration (Days)", "total_wave_cases": "Total Cases"}
        )
        dur_fig.update_traces(texttemplate="%{text} days", textposition="outside")
        dur_fig.update_layout(height=360)
        st.plotly_chart(dur_fig, use_container_width=True)

# Data Tables & Export
tab1, tab2 = st.tabs([f"{selected_country} Wave Metrics & CFR", "All Countries Stored in MongoDB"])

with tab1:
    if wave_records:
        display_df = wave_df.drop(columns=["Wave Label"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {selected_country} Wave Report (CSV)",
            data=csv_data,
            file_name=f"{selected_country}_covid_wave_cfr_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No waves detected with the current slider thresholds.")

with tab2:
    all_waves = list(waves_col.find({}, {"_id": 0}).sort([("country", 1), ("wave_number", 1)]))
    if all_waves:
        all_waves_df = pd.DataFrame(all_waves)
        for col in ["start_date", "peak_date", "end_date"]:
            if col in all_waves_df.columns:
                all_waves_df[col] = pd.to_datetime(all_waves_df[col]).dt.strftime("%Y-%m-%d")
        st.dataframe(all_waves_df, use_container_width=True, hide_index=True)