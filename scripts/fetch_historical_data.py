"""
Fetch historical AQI + weather data for all configured cities.
Run this once to populate data/raw/ before training.

Usage:
    python scripts/fetch_historical_data.py
    python scripts/fetch_historical_data.py --city Kolkata --months 6
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from config.settings import CITIES, RAW_DIR
from src.data_ingestion.aqi_api import fetch_aqi
from src.data_ingestion.weather_api import fetch_weather


def fetch_city_data(city_name: str, months: int = 12):
    """Fetch AQI and weather data for a single city."""
    city = CITIES[city_name]
    lat, lon = city["lat"], city["lon"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"Fetching data for {city_name}")
    print(f"  Coords: ({lat}, {lon})")
    print(f"  Range: {start_str} to {end_str}")
    print(f"{'='*50}")

    os.makedirs(RAW_DIR, exist_ok=True)

    # Fetch AQI data
    print("  Fetching AQI data...")
    try:
        aqi_df = fetch_aqi(lat, lon, start_str, end_str)
        aqi_path = os.path.join(RAW_DIR, f"{city_name.lower()}_aqi.csv")
        aqi_df.to_csv(aqi_path, index=False)
        print(f"  ✓ AQI data: {len(aqi_df)} rows → {aqi_path}")
    except Exception as e:
        print(f"  ✗ AQI fetch failed: {e}")
        return False

    # Fetch weather data
    print("  Fetching weather data...")
    try:
        weather_df = fetch_weather(lat, lon, start_str, end_str)
        weather_path = os.path.join(RAW_DIR, f"{city_name.lower()}_weather.csv")
        weather_df.to_csv(weather_path, index=False)
        print(f"  ✓ Weather data: {len(weather_df)} rows → {weather_path}")
    except Exception as e:
        print(f"  ✗ Weather fetch failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch historical AQI + weather data")
    parser.add_argument("--city", type=str, default=None, help="Fetch for a specific city only")
    parser.add_argument("--months", type=int, default=12, help="Months of history to fetch")
    args = parser.parse_args()

    if args.city:
        if args.city not in CITIES:
            print(f"Unknown city: {args.city}")
            print(f"Available: {', '.join(CITIES.keys())}")
            sys.exit(1)
        cities = [args.city]
    else:
        cities = list(CITIES.keys())

    results = {}
    for city in cities:
        results[city] = fetch_city_data(city, args.months)

    print(f"\n{'='*50}")
    print("Summary:")
    for city, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {city}")


if __name__ == "__main__":
    main()
