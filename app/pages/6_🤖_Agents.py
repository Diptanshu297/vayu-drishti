"""
VayuDrishti — Agent Activity page.
Run and display the multi-agent pipeline (Data Analyst → Forecaster → Attribution → Advisory).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import json

from config.settings import CITIES
from src.data_ingestion.aqi_api import fetch_current_aqi
from src.utils.aqi_categories import compute_aqi


st.set_page_config(page_title="VayuDrishti — Agents", page_icon="🤖", layout="wide")
st.title("🤖 Agent Activity")
st.caption("Multi-agent pipeline: Data Analyst → Forecaster → Attribution → Health Advisory")

city = st.selectbox("Select city", list(CITIES.keys()), key="agent_city")
city_info = CITIES[city]

# Get current data to feed to agents
@st.cache_data(ttl=1800)
def get_agent_input(lat, lon, city_name):
    df = fetch_current_aqi(lat, lon)
    rows = []
    for _, row in df.head(24).iterrows():
        conc = {
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "no2": row["no2"],
            "co": row["co"] / 1000 if pd.notna(row["co"]) else None,
        }
        result = compute_aqi(conc)
        rows.append({
            "timestamp": str(row["timestamp"]),
            "aqi": result["aqi"],
            "category": result["category"],
            "dominant_pollutant": result["dominant_pollutant"],
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "wind_speed": 3.0,  # placeholder
        })
    return {
        "city": city_name,
        "station_readings": rows[:6],  # keep input small for faster agent response
        "current_conditions": rows[0] if rows else {},
    }


# Show what the agents will receive
with st.expander("📥 Agent input data"):
    try:
        input_data = get_agent_input(city_info["lat"], city_info["lon"], city)
        st.json(input_data)
    except Exception as e:
        st.error(f"Failed to fetch input data: {e}")
        st.stop()

st.divider()

# Check for API key
api_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not api_key:
    st.warning("Set ANTHROPIC_API_KEY environment variable to run the agent pipeline live.")
    st.code("$env:ANTHROPIC_API_KEY = 'your-key-here'  # PowerShell", language="powershell")

    # Show pre-recorded demo
    st.divider()
    st.subheader("Demo: Pre-recorded agent output")
    st.info("Below is a sample of what the agent pipeline produces when run with a valid API key.")

    demo_log = [
        {
            "agent": "Data Analyst",
            "time": "2.3s",
            "analysis": f"Analyzed 6 station readings for {city}. All readings within expected range — no sensor anomalies detected. Current atmospheric stability: Class D (neutral) based on moderate wind speed and daytime conditions. Overall air quality is {input_data['current_conditions'].get('category', 'Moderate')} with {(input_data['current_conditions'].get('dominant_pollutant') or 'pm2_5').upper()} as the dominant pollutant.",
            "alerts": [],
            "status": "complete",
        },
        {
            "agent": "Forecaster",
            "time": "2.1s",
            "analysis": f"Based on current AQI of {input_data['current_conditions'].get('aqi', 'N/A')} and recent trends, forecasting AQI to remain in the {input_data['current_conditions'].get('category', 'Moderate')} category for the next 24 hours. No threshold crossings predicted in the immediate forecast window.",
            "alerts": [],
            "status": "complete",
        },
        {
            "agent": "Attribution Analyst",
            "time": "2.8s",
            "analysis": f"Current wind conditions (Class D neutral) indicate moderate dispersion. Primary contributors to {city} air quality are traffic emissions during peak hours and nearby industrial zones. Recommending continued monitoring of industrial compliance.",
            "alerts": [],
            "status": "complete",
        },
        {
            "agent": "Health Advisory",
            "time": "1.9s",
            "analysis": f"Advisory level: routine. Current conditions do not warrant emergency advisories. Generating standard daily advisory for citizen audience in {CITIES[city].get('lang', 'hindi')}. Sensitive individuals (elderly, children, respiratory patients) should limit prolonged outdoor activity.",
            "alerts": [],
            "status": "complete",
        },
    ]

    for entry in demo_log:
        with st.container():
            status_icon = "✅" if entry["status"] == "complete" else "⚠️"
            st.markdown(f"### {status_icon} {entry['agent']} ({entry['time']})")
            st.markdown(entry["analysis"])
            if entry["alerts"]:
                for alert in entry["alerts"]:
                    st.warning(alert)
            st.divider()

else:
    # Live agent execution
    if st.button("🚀 Run agent pipeline", type="primary"):
        from src.agents.vayu_agents import build_pipeline

        pipeline = build_pipeline(verbose=True)

        with st.spinner("Running 4-agent pipeline..."):
            result = pipeline.run(input_data)

        st.success(f"Pipeline complete — {result['total_elapsed']}s total")

        # Display each agent's output
        for entry in result["execution_log"]:
            with st.container():
                status = entry["output"].get("status", "unknown")
                status_icon = "✅" if status == "complete" else "⚠️"

                st.markdown(f"### {status_icon} {entry['agent']} ({entry['elapsed_seconds']}s)")
                st.markdown(entry["output"].get("analysis", "No analysis"))

                alerts = entry["output"].get("alerts", [])
                if alerts:
                    for alert in alerts:
                        st.warning(alert)

                with st.expander("Raw output"):
                    st.json(entry["output"])

                st.divider()

        # Save log for future display
        st.session_state["agent_log"] = result["execution_log"]
