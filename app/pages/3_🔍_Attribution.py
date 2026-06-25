"""
VayuDrishti — Source Attribution page.
Gaussian plume dispersion + wind direction + source contribution pie chart.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from config.settings import CITIES
from src.attribution.dispersion_model import compute_source_attribution, generate_plume_grid
from src.utils.geo_utils import classify_stability
from src.data_ingestion.weather_api import fetch_weather
from datetime import datetime, timedelta


st.set_page_config(page_title="VayuDrishti — Attribution", page_icon="🔍", layout="wide")
st.title("🔍 Pollution Source Attribution")

city = st.selectbox("Select city", list(CITIES.keys()), key="attr_city")
city_info = CITIES[city]

# --- Known pollution sources (demo data) ---
# In production, these come from OSM via land_use.py
DEMO_SOURCES = {
    "Kolkata": [
        {"name": "Taratala Industrial Zone", "lat": 22.4953, "lon": 88.3210, "emission_rate": 80, "stack_height": 35, "source_type": "industrial"},
        {"name": "Howrah Industrial Belt", "lat": 22.5958, "lon": 88.2636, "emission_rate": 100, "stack_height": 40, "source_type": "industrial"},
        {"name": "Kasba Industrial Area", "lat": 22.5100, "lon": 88.3800, "emission_rate": 50, "stack_height": 25, "source_type": "industrial"},
        {"name": "EM Bypass Traffic Corridor", "lat": 22.5400, "lon": 88.3950, "emission_rate": 40, "stack_height": 5, "source_type": "traffic"},
        {"name": "Kolkata Port Area", "lat": 22.5500, "lon": 88.3300, "emission_rate": 60, "stack_height": 30, "source_type": "industrial"},
    ],
    "Delhi": [
        {"name": "Anand Vihar Industrial", "lat": 28.6469, "lon": 77.3164, "emission_rate": 120, "stack_height": 40, "source_type": "industrial"},
        {"name": "Wazirpur Industrial Area", "lat": 28.6996, "lon": 77.1658, "emission_rate": 90, "stack_height": 35, "source_type": "industrial"},
        {"name": "Okhla Industrial Estate", "lat": 28.5355, "lon": 77.2712, "emission_rate": 80, "stack_height": 30, "source_type": "industrial"},
        {"name": "Ring Road Traffic", "lat": 28.6200, "lon": 77.2400, "emission_rate": 60, "stack_height": 5, "source_type": "traffic"},
    ],
}

# Fallback sources for cities without specific data
DEFAULT_SOURCES = [
    {"name": "Industrial Zone 1", "lat": city_info["lat"] + 0.04, "lon": city_info["lon"] - 0.03, "emission_rate": 70, "stack_height": 30, "source_type": "industrial"},
    {"name": "Industrial Zone 2", "lat": city_info["lat"] - 0.05, "lon": city_info["lon"] + 0.02, "emission_rate": 50, "stack_height": 25, "source_type": "industrial"},
    {"name": "Traffic Corridor", "lat": city_info["lat"] + 0.01, "lon": city_info["lon"] + 0.04, "emission_rate": 40, "stack_height": 5, "source_type": "traffic"},
]

sources = DEMO_SOURCES.get(city, DEFAULT_SOURCES)

# --- Controls ---
st.sidebar.header("Atmospheric conditions")

wind_direction = st.sidebar.slider(
    "Wind direction (°, from)",
    0, 360, 270,
    help="Meteorological convention: 0=N, 90=E, 180=S, 270=W"
)

wind_speed = st.sidebar.slider("Wind speed (m/s)", 0.5, 15.0, 3.0, 0.5)

hour = st.sidebar.slider("Hour of day", 0, 23, 14)

cloud_cover = st.sidebar.slider("Cloud cover", 0.0, 1.0, 0.3, 0.1)

stability = classify_stability(wind_speed, hour, cloud_cover)
st.sidebar.metric("Stability class", stability)
stability_labels = {
    "A": "Very unstable", "B": "Moderately unstable", "C": "Slightly unstable",
    "D": "Neutral", "E": "Slightly stable", "F": "Very stable"
}
st.sidebar.caption(stability_labels.get(stability, ""))

# --- Receptor (monitoring station) ---
receptor_lat = city_info["lat"]
receptor_lon = city_info["lon"]

# --- Compute attribution ---
result = compute_source_attribution(
    sources=sources,
    receptor_lat=receptor_lat,
    receptor_lon=receptor_lon,
    wind_speed=wind_speed,
    wind_direction=wind_direction,
    stability_class=stability,
)

# --- Layout ---
col1, col2 = st.columns([3, 2])

with col1:
    # Map with sources and plume
    m = folium.Map(
        location=[receptor_lat, receptor_lon],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    # Add receptor marker
    folium.Marker(
        [receptor_lat, receptor_lon],
        popup="Monitoring Station",
        tooltip="Receptor",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    # Add source markers
    source_colors = {"industrial": "red", "traffic": "orange", "construction": "gray"}

    for src in sources:
        attr = next((a for a in result["attributions"] if a["name"] == src["name"]), None)
        pct = attr["percentage"] if attr else 0

        folium.Marker(
            [src["lat"], src["lon"]],
            popup=f"<b>{src['name']}</b><br>Type: {src['source_type']}<br>Contribution: {pct:.1f}%",
            tooltip=f"{src['name']}: {pct:.1f}%",
            icon=folium.Icon(
                color=source_colors.get(src["source_type"], "gray"),
                icon="industry" if src["source_type"] == "industrial" else "car",
                prefix="fa",
            ),
        ).add_to(m)

    # Add wind direction arrow
    wind_toward = (wind_direction + 180) % 360
    arrow_len = 0.03
    end_lat = receptor_lat + arrow_len * np.cos(np.radians(wind_toward))
    end_lon = receptor_lon + arrow_len * np.sin(np.radians(wind_toward))

    folium.PolyLine(
        [[receptor_lat, receptor_lon], [end_lat, end_lon]],
        color="#3b82f6",
        weight=3,
        opacity=0.8,
        tooltip=f"Wind from {wind_direction}°",
    ).add_to(m)

    # Add plume overlay for the top contributing source
    top_source = None
    for a in result["attributions"]:
        if a["percentage"] > 0:
            top_source = next(s for s in sources if s["name"] == a["name"])
            break

    if top_source:
        plume = generate_plume_grid(
            source=top_source,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            stability_class=stability,
            grid_size=30,
            max_distance=8000,
        )

        # Normalize and add as heatmap
        conc = plume["concentrations"]
        if conc.max() > 0:
            from folium.plugins import HeatMap
            heat_data = []
            for i, lat in enumerate(plume["lats"]):
                for j, lon in enumerate(plume["lons"]):
                    if conc[i, j] > conc.max() * 0.01:
                        heat_data.append([lat, lon, float(conc[i, j] / conc.max())])

            if heat_data:
                HeatMap(
                    heat_data,
                    radius=25,
                    blur=20,
                    gradient={"0.2": "yellow", "0.5": "orange", "0.8": "red", "1.0": "darkred"},
                ).add_to(m)

    st_folium(m, width=None, height=500, use_container_width=True)

with col2:
    # Attribution pie chart
    st.subheader("Source attribution")

    active = [a for a in result["attributions"] if a["percentage"] > 0.1]

    if active:
        fig = go.Figure(go.Pie(
            labels=[a["name"] for a in active],
            values=[a["percentage"] for a in active],
            marker_colors=["#ef4444", "#f59e0b", "#3b82f6", "#22c55e", "#8b5cf6"][:len(active)],
            textinfo="label+percent",
            hole=0.3,
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Details table
        for a in active:
            st.markdown(f"**{a['name']}** — {a['percentage']:.1f}%")
            st.caption(f"Type: {a['source_type']} · Distance: {abs(a['downwind_distance']):.0f}m downwind")
    else:
        st.info("No sources are upwind at the current wind direction. Try adjusting the wind direction slider.")

    # Stability explanation
    st.divider()
    st.subheader("Atmospheric conditions")
    st.markdown(f"""
    **Wind:** {wind_speed} m/s from {wind_direction}°  
    **Stability:** Class {stability} ({stability_labels[stability]})  
    **Hour:** {hour}:00 · Cloud cover: {cloud_cover:.0%}
    """)

    if stability in ("E", "F"):
        st.warning("Stable atmosphere — pollution disperses slowly. Concentrations will be elevated near sources.")
    elif stability in ("A", "B"):
        st.success("Unstable atmosphere — rapid dispersion. Pollution spreads quickly but dilutes fast.")
