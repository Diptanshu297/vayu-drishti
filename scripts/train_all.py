"""
Train AQI forecasters for all cities that have fetched data.

Usage:
    uv run python scripts/train_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config.settings import CITIES, RAW_DIR, PROCESSED_DIR, FORECAST_HORIZON_HOURS
from src.feature_engineering.build_features import build_features
from src.models.train_forecaster import train_forecaster


def main():
    results = {}

    for city_name in CITIES:
        city_lower = city_name.lower()
        aqi_path = os.path.join(RAW_DIR, f"{city_lower}_aqi.csv")
        weather_path = os.path.join(RAW_DIR, f"{city_lower}_weather.csv")

        if not os.path.exists(aqi_path) or not os.path.exists(weather_path):
            print(f"⏭  {city_name}: no data found, skipping")
            results[city_name] = None
            continue

        print(f"\n{'='*60}")
        print(f"Training: {city_name}")
        print(f"{'='*60}")

        try:
            aqi_df = pd.read_csv(aqi_path, parse_dates=["timestamp"])
            weather_df = pd.read_csv(weather_path, parse_dates=["timestamp"])

            df = build_features(aqi_df, weather_df)

            os.makedirs(PROCESSED_DIR, exist_ok=True)
            df.to_csv(os.path.join(PROCESSED_DIR, f"features_{city_lower}.csv"), index=False)

            result = train_forecaster(df, city=city_lower)
            metrics = result["metrics"]
            results[city_name] = metrics

            print(f"  RMSE: {metrics['model_rmse']} (baseline: {metrics['baseline_rmse']})")
            print(f"  Improvement: {metrics['improvement_pct']}%")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[city_name] = None

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'CITY':<15} {'RMSE':>8} {'BASELINE':>10} {'IMPROVE':>10} {'R²':>8}")
    print(f"{'='*60}")

    for city_name, metrics in results.items():
        if metrics:
            print(f"{city_name:<15} {metrics['model_rmse']:>8} {metrics['baseline_rmse']:>10} {metrics['improvement_pct']:>9}% {metrics['model_r2']:>8}")
        else:
            print(f"{city_name:<15} {'—':>8} {'—':>10} {'—':>10} {'—':>8}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()