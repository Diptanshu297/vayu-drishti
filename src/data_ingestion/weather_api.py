"""
Fetch historical and current weather data from Open-Meteo.
Free, no API key required.
"""

import requests
import pandas as pd
from config.settings import OPEN_METEO_WEATHER_URL


def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo.

    Args:
        lat, lon: coordinates
        start_date, end_date: 'YYYY-MM-DD' format

    Returns:
        DataFrame with columns:
            timestamp, temperature, humidity, wind_speed,
            wind_direction, pressure, cloud_cover
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cloud_cover",
        ]),
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(OPEN_METEO_WEATHER_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "pressure": hourly["surface_pressure"],
        "cloud_cover": hourly["cloud_cover"],
    })

    return df
