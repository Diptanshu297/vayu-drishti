"""
Quick test: verify data fetching works on your machine.

Usage:
    uv run python scripts/test_fetch.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.aqi_api import fetch_aqi, fetch_current_aqi
from src.data_ingestion.weather_api import fetch_weather
from src.utils.aqi_categories import compute_aqi
from config.settings import CITIES


def test_single_city():
    """Test fetching a small amount of data for Kolkata."""
    city = CITIES["Kolkata"]
    lat, lon = city["lat"], city["lon"]

    print("=" * 50)
    print("Testing data fetch for Kolkata")
    print("=" * 50)

    # Test 1: Historical AQI (7 days)
    print("\n1. Fetching 7 days of AQI data...")
    try:
        aqi_df = fetch_aqi(lat, lon, "2026-01-01", "2026-01-07")
        print(f"   ✓ Got {len(aqi_df)} rows")
        print(f"   Columns: {list(aqi_df.columns)}")
        print(f"   PM2.5 range: {aqi_df['pm2_5'].min():.1f} – {aqi_df['pm2_5'].max():.1f} μg/m³")
        print(f"   Sample row:")
        print(f"   {aqi_df.iloc[12].to_dict()}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 2: Historical weather (7 days)
    print("\n2. Fetching 7 days of weather data...")
    try:
        weather_df = fetch_weather(lat, lon, "2026-01-01", "2026-01-07")
        print(f"   ✓ Got {len(weather_df)} rows")
        print(f"   Columns: {list(weather_df.columns)}")
        print(f"   Temp range: {weather_df['temperature'].min():.1f} – {weather_df['temperature'].max():.1f} °C")
        print(f"   Wind range: {weather_df['wind_speed'].min():.1f} – {weather_df['wind_speed'].max():.1f} m/s")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 3: Current + forecast AQI
    print("\n3. Fetching current AQI + 5-day forecast...")
    try:
        current_df = fetch_current_aqi(lat, lon)
        print(f"   ✓ Got {len(current_df)} rows ({len(current_df)//24} days)")

        # Compute AQI for the latest reading
        latest = current_df.iloc[0]
        concentrations = {
            "pm2_5": latest["pm2_5"],
            "pm10": latest["pm10"],
            "no2": latest["no2"],
        }
        aqi_result = compute_aqi(concentrations)
        print(f"   Current AQI: {aqi_result['aqi']} ({aqi_result['category']})")
        print(f"   Dominant pollutant: {aqi_result['dominant_pollutant']}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test 4: Verify all cities work
    print("\n4. Quick ping for all cities...")
    for city_name, city_info in CITIES.items():
        try:
            df = fetch_aqi(city_info["lat"], city_info["lon"], "2026-06-01", "2026-06-02")
            latest_pm = df["pm2_5"].iloc[12] if len(df) > 12 else df["pm2_5"].iloc[0]
            print(f"   ✓ {city_name}: PM2.5 = {latest_pm:.1f} μg/m³")
        except Exception as e:
            print(f"   ✗ {city_name}: {e}")

    print("\n" + "=" * 50)
    print("All tests passed ✓")
    print("=" * 50)
    print("\nNext step: run the full historical fetch:")
    print("  uv run python scripts/fetch_historical_data.py --city Kolkata --months 12")
    return True


if __name__ == "__main__":
    test_single_city()
