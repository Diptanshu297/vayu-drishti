"""
Train the XGBoost 24h AQI forecaster for a city.

Usage:
    uv run python scripts/train_model.py
    uv run python scripts/train_model.py --city Kolkata
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config.settings import RAW_DIR, PROCESSED_DIR, FORECAST_HORIZON_HOURS
from src.feature_engineering.build_features import build_features, get_feature_columns
from src.models.train_forecaster import train_forecaster


def main():
    parser = argparse.ArgumentParser(description="Train AQI forecaster")
    parser.add_argument("--city", type=str, default="Kolkata")
    args = parser.parse_args()

    city = args.city
    city_lower = city.lower()

    # Load raw data
    aqi_path = os.path.join(RAW_DIR, f"{city_lower}_aqi.csv")
    weather_path = os.path.join(RAW_DIR, f"{city_lower}_weather.csv")

    if not os.path.exists(aqi_path) or not os.path.exists(weather_path):
        print(f"✗ Raw data not found for {city}.")
        print(f"  Run: uv run python scripts/fetch_historical_data.py --city {city}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Training AQI forecaster for {city}")
    print(f"{'='*60}")

    # Load CSVs
    print("\n1. Loading raw data...")
    aqi_df = pd.read_csv(aqi_path, parse_dates=["timestamp"])
    weather_df = pd.read_csv(weather_path, parse_dates=["timestamp"])
    print(f"   AQI: {len(aqi_df)} rows")
    print(f"   Weather: {len(weather_df)} rows")

    # Build features
    print("\n2. Building features...")
    df = build_features(aqi_df, weather_df)
    print(f"   Feature-engineered: {len(df)} rows (dropped NaN from lag/rolling)")
    print(f"   Features: {len(get_feature_columns())} columns")

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, f"features_{city_lower}.csv")
    df.to_csv(processed_path, index=False)
    print(f"   Saved: {processed_path}")

    # Train model
    print("\n3. Training XGBoost...")
    result = train_forecaster(df, city=city_lower)

    # Print results
    metrics = result["metrics"]
    print(f"\n{'='*60}")
    print(f"RESULTS — {FORECAST_HORIZON_HOURS}h AQI Forecast for {city}")
    print(f"{'='*60}")
    print(f"")
    print(f"  XGBoost RMSE:     {metrics['model_rmse']}")
    print(f"  Baseline RMSE:    {metrics['baseline_rmse']}")
    print(f"  Improvement:      {metrics['improvement_pct']}%")
    print(f"")
    print(f"  XGBoost MAE:      {metrics['model_mae']}")
    print(f"  Baseline MAE:     {metrics['baseline_mae']}")
    print(f"  XGBoost R²:       {metrics['model_r2']}")
    print(f"")

    # Top 5 features
    print(f"  Top 5 features:")
    importances = result["feature_importances"]
    for i, (feat, imp) in enumerate(list(importances.items())[:5]):
        bar = "█" * int(imp * 100)
        print(f"    {i+1}. {feat:30s} {imp:.3f} {bar}")

    print(f"\n  Model saved to: models/saved/xgb_{city_lower}_{FORECAST_HORIZON_HOURS}h.joblib")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()