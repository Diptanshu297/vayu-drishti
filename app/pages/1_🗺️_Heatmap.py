"""
VayuDrishti — AQI Heatmap page.
Folium map with color-coded AQI markers and pollution heatmap overlay.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd

from config.settings import CITIES
from src.data_ingestion.aqi_api import fetch_current_aqi
from src.utils.aqi_categories import compute_aqi


st.set_page_config(page_title="VayuDrishti — Heatmap", page_icon="🗺️", layout="wide")
st.title("🗺️ AQI Heatmap")

city = st.selectbox("Select city", list(CITIES.keys()), key="map_city")
city_info = CITIES[city]


def aqi_color(aqi_val):
    if aqi_val <= 50: return "green"
    if aqi_val <= 100: return "lightgreen"
    if aqi_val <= 200: return "orange"
    if aqi_val <= 300: return "red"
    if aqi_val <= 400: return "darkred"
    return "black"


@st.cache_data(ttl=1800)
def get_grid_data(lat, lon):
    """Generate a grid of AQI readings around the city center."""
    # Create a grid of points around city center
    offsets = np.linspace(-0.08, 0.08, 5)
    points = []

    for dlat in offsets:
        for dlon in offsets:
            plat = lat + dlat
            plon = lon + dlon

            try:
                df = fetch_current_aqi(plat, plon)
                row = df.iloc[0]
                conc = {
                    "pm2_5": row["pm2_5"],
                    "pm10": row["pm10"],
                    "no2": row["no2"],
                    "co": row["co"] / 1000 if pd.notna(row["co"]) else None,
                }
                result = compute_aqi(conc)
                points.append({
                    "lat": plat,
                    "lon": plon,
                    "aqi": result["aqi"] or 0,
                    "category": result["category"],
                    "dominant": result["dominant_pollutant"] or "unknown",
                    "pm2_5": row["pm2_5"],
                })
            except Exception:
                continue

    return points


with st.spinner("Loading AQI grid data..."):
    points = get_grid_data(city_info["lat"], city_info["lon"])

if points:
    # Create folium map
    m = folium.Map(
        location=[city_info["lat"], city_info["lon"]],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    # Add markers for each grid point
    for p in points:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=18,
            color=aqi_color(p["aqi"]),
            fill=True,
            fill_color=aqi_color(p["aqi"]),
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>AQI: {p['aqi']}</b><br>"
                f"Category: {p['category']}<br>"
                f"Dominant: {p['dominant'].upper()}<br>"
                f"PM2.5: {p['pm2_5']:.1f} μg/m³",
                max_width=200,
            ),
            tooltip=f"AQI: {p['aqi']} ({p['category']})",
        ).add_to(m)

    # Add heatmap layer
    from folium.plugins import HeatMap
    heat_data = [[p["lat"], p["lon"], p["aqi"]] for p in points]
    HeatMap(
        heat_data,
        radius=40,
        blur=30,
        max_zoom=13,
        gradient={
            0.2: "green",
            0.4: "yellow",
            0.6: "orange",
            0.8: "red",
            1.0: "darkred",
        },
    ).add_to(m)

    # Legend
    st.markdown(
        """
        <div style="display: flex; gap: 12px; margin-bottom: 8px; flex-wrap: wrap;">
            <span>🟢 Good (0-50)</span>
            <span>🟡 Moderate (101-200)</span>
            <span>🟠 Poor (201-300)</span>
            <span>🔴 Very Poor (301-400)</span>
            <span>⚫ Severe (401+)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st_folium(m, width=None, height=550, use_container_width=True)

    # Data table below map
    with st.expander("Grid data"):
        grid_df = pd.DataFrame(points)
        st.dataframe(grid_df, hide_index=True, use_container_width=True)

else:
    st.error("Could not load grid data. Check internet connection.")
