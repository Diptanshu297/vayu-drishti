"""
Fetch historical and current weather data from Open-Meteo.
Splits requests across archive and forecast endpoints as needed.
Free, no API key required.
"""

from datetime import datetime, timedelta

import requests
import pandas as pd
from config.settings import OPEN_METEO_WEATHER_URL, OPEN_METEO_WEATHER_ARCHIVE_URL


HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
]

COLUMN_MAP = {
    "time": "timestamp",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "surface_pressure": "pressure",
    "cloud_cover": "cloud_cover",
}


def _fetch_chunk(url: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch a single date range from one endpoint."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]

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


def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo.
    Automatically splits across archive and forecast endpoints
    when the date range spans both historical and recent data.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Archive covers up to ~5 days ago; forecast covers last 5 days + future
    cutoff_dt = datetime.now() - timedelta(days=7)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

    chunks = []

    if start_dt < cutoff_dt:
        # Need archive for the historical portion
        archive_end = min(end_dt, cutoff_dt).strftime("%Y-%m-%d")
        print(f"    [weather] archive: {start_date} → {archive_end}")
        chunks.append(_fetch_chunk(
            OPEN_METEO_WEATHER_ARCHIVE_URL, lat, lon, start_date, archive_end
        ))

    if end_dt >= cutoff_dt:
        # Need forecast for the recent portion
        forecast_start = max(start_dt, cutoff_dt).strftime("%Y-%m-%d")
        print(f"    [weather] forecast: {forecast_start} → {end_date}")
        chunks.append(_fetch_chunk(
            OPEN_METEO_WEATHER_URL, lat, lon, forecast_start, end_date
        ))

    if not chunks:
        raise ValueError(f"No data available for range {start_date} to {end_date}")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df