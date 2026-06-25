"""
VayuDrishti — Home page.
City selector, current AQI gauge, and quick stats.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from config.settings import CITIES, AQI_CATEGORIES
from src.data_ingestion.aqi_api import fetch_current_aqi
from src.utils.aqi_categories import compute_aqi


st.set_page_config(
    page_title="VayuDrishti",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ VayuDrishti")
st.caption("AI-Powered Urban Air Quality Intelligence")

# City selector — persists across pages via session state
city = st.selectbox("Select city", list(CITIES.keys()), key="selected_city")
city_info = CITIES[city]


@st.cache_data(ttl=1800)  # cache 30 min
def get_current_data(lat, lon):
    df = fetch_current_aqi(lat, lon)
    return df


try:
    current_df = get_current_data(city_info["lat"], city_info["lon"])
    now = current_df.iloc[0]

    concentrations = {
        "pm2_5": now["pm2_5"],
        "pm10": now["pm10"],
        "no2": now["no2"],
        "so2": now["so2"],
        "o3": now["o3"],
        "co": now["co"] / 1000,  # μg/m³ → mg/m³
    }
    aqi_result = compute_aqi(concentrations)
    aqi_val = aqi_result["aqi"] or 0

    # --- AQI Gauge ---
    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            title={"text": f"{city} — Current AQI"},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": aqi_result["color"]},
                "steps": [
                    {"range": [r["range"][0], r["range"][1]], "color": r["color"]}
                    for r in AQI_CATEGORIES
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": aqi_val,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("AQI Category", aqi_result["category"])
        st.metric("Dominant Pollutant", aqi_result["dominant_pollutant"].upper())

        st.markdown("**Pollutant readings (μg/m³)**")
        readings = {
            "PM2.5": now["pm2_5"],
            "PM10": now["pm10"],
            "NO₂": now["no2"],
            "SO₂": now["so2"],
            "O₃": now["o3"],
            "CO": now["co"],
        }
        for name, val in readings.items():
            if pd.notna(val):
                st.text(f"  {name}: {val:.1f}")

    # --- 5-day sparkline ---
    st.subheader("5-day AQI trend")

    aqi_series = []
    for _, row in current_df.iterrows():
        conc = {
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "no2": row["no2"],
            "co": row["co"] / 1000 if pd.notna(row["co"]) else None,
        }
        r = compute_aqi(conc)
        aqi_series.append(r["aqi"] or 0)

    trend_df = pd.DataFrame({
        "timestamp": current_df["timestamp"],
        "AQI": aqi_series,
    })

    st.line_chart(trend_df, x="timestamp", y="AQI", height=250)

    # --- Multi-city quick comparison ---
    st.subheader("All cities — current snapshot")

    city_rows = []
    for cname, cinfo in CITIES.items():
        try:
            cdf = get_current_data(cinfo["lat"], cinfo["lon"])
            crow = cdf.iloc[0]
            cconc = {
                "pm2_5": crow["pm2_5"],
                "pm10": crow["pm10"],
                "no2": crow["no2"],
                "co": crow["co"] / 1000 if pd.notna(crow["co"]) else None,
            }
            cr = compute_aqi(cconc)
            city_rows.append({
                "City": cname,
                "AQI": cr["aqi"] or 0,
                "Category": cr["category"],
                "Dominant": (cr["dominant_pollutant"] or "").upper(),
                "PM2.5": f"{crow['pm2_5']:.1f}" if pd.notna(crow["pm2_5"]) else "—",
            })
        except Exception:
            city_rows.append({"City": cname, "AQI": "—", "Category": "Error", "Dominant": "—", "PM2.5": "—"})

    st.dataframe(pd.DataFrame(city_rows), hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.info("Check your internet connection. The dashboard needs Open-Meteo API access.")
