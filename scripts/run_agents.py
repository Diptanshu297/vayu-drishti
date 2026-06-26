"""
Pre-run the agent pipeline and save results for demo display.
Run this once when rate limits aren't an issue (with delays).

Usage:
    uv run python scripts/run_agents.py --city Kolkata
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CITIES
from src.data_ingestion.aqi_api import fetch_current_aqi
from src.utils.aqi_categories import compute_aqi
from src.agents.pipeline import Agent

# Run agents ONE AT A TIME with long delays
AGENTS = [
    Agent(
        name="Data Analyst",
        role="Environmental Data Scientist",
        instructions="""You validate and analyze incoming air quality and weather data.
Your tasks:
1. Check data completeness — identify any missing readings or gaps
2. Detect anomalies — flag sensor readings that seem like malfunctions
3. Classify atmospheric stability using wind speed, time of day, and cloud cover
4. Summarize current conditions across all stations

In your result, include:
- data_quality: overall assessment
- anomalies: list of any flagged readings
- stability_class: current Pasquill-Gifford class
- condition_summary: 2-3 sentence overview""",
    ),
    Agent(
        name="Forecaster",
        role="AQI Forecasting Specialist",
        instructions="""You analyze forecast data and identify critical threshold crossings.
Your tasks:
1. Review the AQI forecast values for the next 24 hours
2. Identify stations where AQI is predicted to cross category boundaries
3. Determine which crossings are most urgent
4. Flag the time windows and stations that need attribution analysis

In your result, include:
- forecast_summary: overview of predicted conditions
- threshold_crossings: list of crossings
- stations_for_attribution: stations where AQI > 200 forecast
- time_window: the critical period to watch""",
    ),
    Agent(
        name="Attribution Analyst",
        role="Pollution Source Attribution Specialist",
        instructions="""You analyze dispersion model results to determine pollution sources.
Your tasks:
1. Review the attribution results for flagged stations
2. Identify the dominant pollution sources
3. Assess driving factors (stability, wind, emissions)
4. Generate prioritized enforcement recommendations

In your result, include:
- attribution_summary: overview of source contributions
- dominant_sources: list of sources with percentages
- driving_factors: what's causing elevated levels
- enforcement_recommendations: specific actionable recommendations""",
    ),
    Agent(
        name="Health Advisory",
        role="Public Health Communication Specialist",
        instructions="""You generate advisory recommendations based on forecast and attribution.
Your tasks:
1. Determine the appropriate advisory level for each affected area
2. Identify vulnerable populations
3. Generate advisory content for different audiences
4. Recommend languages and channels for each area

In your result, include:
- advisory_level: routine/elevated/high/emergency
- affected_areas: list with advisory levels
- audience_recommendations: key messages per audience type
- language_channels: which languages for which areas""",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Kolkata")
    args = parser.parse_args()

    city = args.city
    city_info = CITIES[city]

    print(f"Fetching current data for {city}...")
    df = fetch_current_aqi(city_info["lat"], city_info["lon"])

    readings = []
    for _, row in df.head(6).iterrows():
        conc = {
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "no2": row["no2"],
            "co": row["co"] / 1000 if row["co"] == row["co"] else None,
        }
        result = compute_aqi(conc)
        readings.append({
            "timestamp": str(row["timestamp"]),
            "aqi": result["aqi"],
            "category": result["category"],
            "dominant_pollutant": result["dominant_pollutant"],
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
        })

    input_data = {
        "city": city,
        "station_readings": readings,
        "current_conditions": readings[0] if readings else {},
    }

    context = {"initial_input": input_data}
    execution_log = []

    for i, agent in enumerate(AGENTS):
        if i > 0:
            print(f"\n   Waiting 20 seconds (rate limit)...")
            time.sleep(20)

        print(f"\n{'─'*50}")
        print(f"Running {agent.name}...")
        result = agent.run(context, verbose=True)
        execution_log.append(result)
        context[agent.name] = result["output"]

    # Save results
    output_path = os.path.join("data", "processed", f"agent_results_{city.lower()}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "city": city,
            "input": input_data,
            "execution_log": execution_log,
        }, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n{'═'*50}")
    print(f"✅ Saved: {output_path}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()