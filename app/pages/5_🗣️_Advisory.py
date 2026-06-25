"""
VayuDrishti — Multilingual Advisory page.
Generate health advisories in Bengali, Hindi, and Kannada using Claude API.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from config.settings import CITIES
from src.advisory.advisory_generator import generate_advisory
from src.data_ingestion.aqi_api import fetch_current_aqi
from src.utils.aqi_categories import compute_aqi


st.set_page_config(page_title="VayuDrishti — Advisory", page_icon="🗣️", layout="wide")
st.title("🗣️ Multilingual Health Advisory")

city = st.selectbox("Select city", list(CITIES.keys()), key="adv_city")
city_info = CITIES[city]

# Language mapping
CITY_LANGUAGES = {
    "Kolkata": "bengali",
    "Delhi": "hindi",
    "Bengaluru": "kannada",
    "Mumbai": "hindi",
    "Chennai": "hindi",
    "Lucknow": "hindi",
}

col1, col2, col3 = st.columns(3)

with col1:
    ward_name = st.text_input("Ward / Area name", value=f"Central {city}")

with col2:
    audience = st.selectbox("Audience", ["citizen", "school", "administrator", "worker"])

with col3:
    default_lang = CITY_LANGUAGES.get(city, "hindi")
    language = st.selectbox(
        "Language",
        ["bengali", "hindi", "kannada", "english"],
        index=["bengali", "hindi", "kannada", "english"].index(default_lang),
    )

# Fetch current data for the advisory
@st.cache_data(ttl=1800)
def get_city_aqi(lat, lon):
    df = fetch_current_aqi(lat, lon)
    row = df.iloc[0]
    conc = {
        "pm2_5": row["pm2_5"],
        "pm10": row["pm10"],
        "no2": row["no2"],
        "co": row["co"] / 1000 if pd.notna(row["co"]) else None,
    }
    result = compute_aqi(conc)
    return {
        "aqi": result["aqi"],
        "category": result["category"],
        "dominant_pollutant": result["dominant_pollutant"],
        "pm2_5": row["pm2_5"],
        "duration_hours": 24,
    }


try:
    forecast_data = get_city_aqi(city_info["lat"], city_info["lon"])
except Exception as e:
    st.error(f"Failed to fetch current data: {e}")
    st.stop()

# Show current conditions
st.divider()
st.subheader("Current conditions")

m1, m2, m3 = st.columns(3)
m1.metric("AQI", forecast_data["aqi"])
m2.metric("Category", forecast_data["category"])
m3.metric("Dominant pollutant", (forecast_data["dominant_pollutant"] or "").upper())

# Generate advisory
st.divider()

lang_display = {"bengali": "বাংলা", "hindi": "हिन्दी", "kannada": "ಕನ್ನಡ", "english": "English"}
audience_display = {"citizen": "👤 Citizen", "school": "🏫 School", "administrator": "🏛️ Administrator", "worker": "👷 Worker"}

if st.button(f"Generate advisory — {audience_display[audience]} · {lang_display[language]}", type="primary"):

    with st.spinner("Generating advisory..."):
        result = generate_advisory(
            forecast_data=forecast_data,
            attribution_data=None,
            audience=audience,
            language=language,
            ward_name=ward_name,
            city=city,
        )

    # Display advisory
    st.divider()

    source_badge = "🤖 AI-generated" if result["source"] == "llm" else "📋 Template"

    st.markdown(f"**{audience_display[audience]}** · {lang_display[language]} · {source_badge}")

    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            font-size: 17px;
            line-height: 1.8;
            margin: 12px 0;
        ">{result['advisory']}</div>""",
        unsafe_allow_html=True,
    )

    # Show verification if available
    if result.get("verification"):
        with st.expander("🔍 Back-translation (quality verification)"):
            st.markdown(result["verification"])

    # Show data used
    with st.expander("📊 Data used for this advisory"):
        st.json(forecast_data)

# --- Side-by-side comparison ---
st.divider()
st.subheader("Quick comparison: same data, different audiences")
st.caption("See how the same AQI data produces different advisories for different audiences")

if st.button("Generate all 3 audiences in Bengali"):
    cols = st.columns(3)
    audiences = ["citizen", "school", "administrator"]
    labels = ["👤 Citizen", "🏫 School", "🏛️ Admin"]

    for i, (aud, label) in enumerate(zip(audiences, labels)):
        with cols[i]:
            st.markdown(f"**{label}**")
            with st.spinner(f"Generating {aud}..."):
                r = generate_advisory(
                    forecast_data=forecast_data,
                    attribution_data=None,
                    audience=aud,
                    language="bengali",
                    ward_name=ward_name,
                    city=city,
                )
            st.markdown(
                f"""<div style="
                    background: #12121a;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    padding: 16px;
                    font-size: 14px;
                    line-height: 1.7;
                    min-height: 200px;
                ">{r['advisory']}</div>""",
                unsafe_allow_html=True,
            )
            source = "🤖 AI" if r["source"] == "llm" else "📋 Template"
            st.caption(source)
