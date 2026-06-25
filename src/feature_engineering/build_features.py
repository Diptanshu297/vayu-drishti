"""
Build ML-ready features from raw AQI + weather data.
Lag features, rolling statistics, temporal encodings, weather features.
"""

import numpy as np
import pandas as pd

from config.settings import LAG_FEATURES, ROLLING_WINDOWS, FORECAST_HORIZON_HOURS
from src.utils.aqi_categories import compute_aqi


def add_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall AQI from individual pollutant columns."""
    df = df.copy()
    aqi_values = []

    for _, row in df.iterrows():
        concentrations = {
            col: row[col]
            for col in ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
            if col in df.columns and pd.notna(row[col])
        }
        result = compute_aqi(concentrations)
        aqi_values.append(result["aqi"])

    df["aqi"] = aqi_values
    return df


def add_lag_features(df: pd.DataFrame, column: str = "aqi") -> pd.DataFrame:
    """Add lagged values of the target column."""
    df = df.copy()
    for lag in LAG_FEATURES:
        df[f"{column}_lag_{lag}h"] = df[column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, column: str = "aqi") -> pd.DataFrame:
    """Add rolling mean and std for the target column."""
    df = df.copy()
    for window in ROLLING_WINDOWS:
        df[f"{column}_rolling_{window}h_mean"] = df[column].rolling(window).mean()
        df[f"{column}_rolling_{window}h_std"] = df[column].rolling(window).std()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and calendar features."""
    df = df.copy()
    hour = df["timestamp"].dt.hour
    month = df["timestamp"].dt.month

    # Cyclical encoding — so hour 23 and hour 0 are adjacent
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def add_target(df: pd.DataFrame, column: str = "aqi") -> pd.DataFrame:
    """Add the prediction target: AQI N hours in the future."""
    df = df.copy()
    df[f"target_{column}_{FORECAST_HORIZON_HOURS}h"] = df[column].shift(-FORECAST_HORIZON_HOURS)
    return df


def build_features(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        aqi_df: DataFrame with timestamp + pollutant columns
        weather_df: DataFrame with timestamp + weather columns

    Returns:
        ML-ready DataFrame with features and target, NaN rows dropped.
    """
    # Merge AQI and weather on timestamp
    df = pd.merge(aqi_df, weather_df, on="timestamp", how="inner")

    # Compute overall AQI
    df = add_aqi_column(df)

    # Feature engineering
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temporal_features(df)
    df = add_target(df)

    # Define feature columns
    feature_cols = get_feature_columns()

    # Drop rows with NaN in features or target
    target_col = f"target_aqi_{FORECAST_HORIZON_HOURS}h"
    required_cols = feature_cols + [target_col]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used by the model."""
    lag_cols = [f"aqi_lag_{lag}h" for lag in LAG_FEATURES]
    rolling_cols = []
    for w in ROLLING_WINDOWS:
        rolling_cols.extend([f"aqi_rolling_{w}h_mean", f"aqi_rolling_{w}h_std"])

    weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure"]
    temporal_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"]

    return lag_cols + rolling_cols + weather_cols + temporal_cols
