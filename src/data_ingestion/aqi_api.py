"""
Fetch historical and current air quality data from Open-Meteo.
Free, no API key required.
"""

import requests
import pandas as pd
from config.settings import OPEN_METEO_AQI_URL


def fetch_aqi(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly air quality data from Open-Meteo.

    Args:
        lat, lon: coordinates
        start_date, end_date: 'YYYY-MM-DD' format

    Returns:
        DataFrame with columns:
            timestamp, pm2_5, pm10, no2, so2, o3, co
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "carbon_monoxide",
        ]),
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(OPEN_METEO_AQI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm2_5": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "o3": hourly["ozone"],
        "co": hourly["carbon_monoxide"],
    })

    return df


def fetch_current_aqi(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch current + 5-day forecast AQI from Open-Meteo.
    Useful for live dashboard display.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide",
        "timezone": "Asia/Kolkata",
        "forecast_days": 5,
    }

    response = requests.get(OPEN_METEO_AQI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    return pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm2_5": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "o3": hourly["ozone"],
        "co": hourly["carbon_monoxide"],
    })
