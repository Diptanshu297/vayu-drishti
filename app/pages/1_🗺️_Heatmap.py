"""
VayuDrishti — AQI Heatmap page.
Folium map with satellite-like spatial pollution overlay.
Loads pre-generated grid data (from GEE or fallback).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import json
import pandas as pd

from config.settings import CITIES, SATELLITE_DIR
from src.data_ingestion.aqi_api import fetch_current_aqi
from src.utils.aqi_categories import compute_aqi


st.set_page_config(page_title="VayuDrishti — Heatmap", page_icon="🗺️", layout="wide")
st.title("🗺️ AQI Heatmap")

city = st.selectbox("Select city", list(CITIES.keys()), key="map_city")
city_info = CITIES[city]
city_lower = city.lower()


def aqi_color(aqi_val):
    if aqi_val is None:
        return "gray"
    if aqi_val <= 50: return "green"
    if aqi_val <= 100: return "lightgreen"
    if aqi_val <= 200: return "orange"
    if aqi_val <= 300: return "red"
    if aqi_val <= 400: return "darkred"
    return "black"


def load_spatial_data(city_lower):
    """Load spatial grid data, merging GEE (NO2) and fallback (PM2.5/PM10/SO2)."""
    gee_path = os.path.join(SATELLITE_DIR, f"{city_lower}_no2.json")
    fallback_path = os.path.join(SATELLITE_DIR, f"{city_lower}_spatial.json")

    gee_grid = None
    fallback_grid = None
    sources = []

    # Load GEE data (has NO2)
    if os.path.exists(gee_path):
        with open(gee_path) as f:
            data = json.load(f)
        if data.get("months"):
            gee_grid = data["months"][-1]["grid"]
            sources.append("Sentinel-5P (NO₂)")

    # Load fallback data (has PM2.5, PM10, SO2)
    if os.path.exists(fallback_path):
        with open(fallback_path) as f:
            data = json.load(f)
        fallback_grid = data.get("grid", [])
        sources.append("CAMS reanalysis (PM2.5/PM10/SO₂)")

    # Merge: use fallback as base, overlay GEE NO2
    if fallback_grid and gee_grid:
        # Build lookup from GEE grid by lat/lon
        gee_lookup = {}
        for p in gee_grid:
            key = (round(p["lat"], 3), round(p["lon"], 3))
            gee_lookup[key] = p.get("no2")

        # Merge: fallback has all pollutants, override NO2 from GEE where available
        merged = []
        for p in fallback_grid:
            entry = dict(p)
            key = None
            # Find closest GEE point
            min_dist = float("inf")
            best_no2 = None
            for gp in gee_grid:
                dist = abs(gp["lat"] - p["lat"]) + abs(gp["lon"] - p["lon"])
                if dist < min_dist:
                    min_dist = dist
                    best_no2 = gp.get("no2")
            if best_no2 is not None and min_dist < 0.05:
                entry["no2"] = best_no2
            merged.append(entry)

        return merged, " + ".join(sources)

    elif fallback_grid:
        return fallback_grid, "CAMS reanalysis"

    elif gee_grid:
        return gee_grid, "Sentinel-5P"

    return None, None


# Load spatial data
grid, source = load_spatial_data(city_lower)

# Pollutant selector
pollutant = st.radio(
    "Overlay pollutant",
    ["no2", "pm2_5", "pm10", "so2"],
    format_func=lambda x: {"no2": "NO₂", "pm2_5": "PM2.5", "pm10": "PM10", "so2": "SO₂"}[x],
    horizontal=True,
)

if grid:
    st.caption(f"Data source: {source} · {len(grid)} grid points")

    # Create map
    m = folium.Map(
        location=[city_info["lat"], city_info["lon"]],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    # Add grid point markers
    for p in grid:
        val = p.get(pollutant)
        if val is None:
            continue

        # Compute AQI for this point if we have PM2.5
        if p.get("pm2_5") is not None:
            conc = {"pm2_5": p["pm2_5"]}
            if p.get("no2") is not None:
                conc["no2"] = p["no2"]
            aqi_result = compute_aqi(conc)
            aqi_val = aqi_result["aqi"]
            cat = aqi_result["category"]
        else:
            aqi_val = None
            cat = "—"

        popup_text = f"<b>AQI: {aqi_val}</b> ({cat})<br>"
        for key in ["no2", "pm2_5", "pm10", "so2"]:
            if p.get(key) is not None:
                label = {"no2": "NO₂", "pm2_5": "PM2.5", "pm10": "PM10", "so2": "SO₂"}[key]
                popup_text += f"{label}: {p[key]:.1f} μg/m³<br>"

        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=14,
            color=aqi_color(aqi_val),
            fill=True,
            fill_color=aqi_color(aqi_val),
            fill_opacity=0.5,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=f"AQI: {aqi_val} ({cat})" if aqi_val else f"{pollutant}: {val:.1f}",
        ).add_to(m)

    # Heatmap overlay
    heat_data = [
        [p["lat"], p["lon"], p[pollutant]]
        for p in grid
        if p.get(pollutant) is not None
    ]

    if heat_data:
        HeatMap(
            heat_data,
            radius=35,
            blur=25,
            max_zoom=13,
            gradient={"0.2": "blue", "0.4": "cyan", "0.6": "yellow", "0.8": "orange", "1.0": "red"},
        ).add_to(m)

    # Legend
    st.markdown(
        """
        <div style="display: flex; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; font-size: 14px;">
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

    # Data table
    with st.expander("Raw grid data"):
        st.dataframe(pd.DataFrame(grid), hide_index=True, use_container_width=True)

else:
    st.warning(f"No spatial data found for {city}. Generate it first:")
    st.code(f"uv run python scripts/generate_satellite_fallback.py --city {city}", language="bash")

    st.info("Or if you have GEE access:")
    st.code(f"uv run python scripts/fetch_satellite.py --city {city}", language="bash")