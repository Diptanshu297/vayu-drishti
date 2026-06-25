"""
Train XGBoost 24h AQI forecaster and compare against persistence baseline.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config.settings import (
    XGBOOST_PARAMS,
    TRAIN_TEST_SPLIT_RATIO,
    FORECAST_HORIZON_HOURS,
    MODELS_DIR,
)
from src.feature_engineering.build_features import get_feature_columns


def train_forecaster(df: pd.DataFrame, city: str = "kolkata") -> dict:
    """
    Train XGBoost model on feature-engineered DataFrame.

    Args:
        df: output of build_features() — must have feature columns and target
        city: for naming saved model files

    Returns:
        dict with model, metrics, and feature importances
    """
    feature_cols = get_feature_columns()
    target_col = f"target_aqi_{FORECAST_HORIZON_HOURS}h"

    X = df[feature_cols]
    y = df[target_col]

    # Chronological split — NEVER shuffle time series
    split_idx = int(len(df) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train XGBoost
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    predictions = model.predict(X_test)

    # Persistence baseline: AQI(t+24) = AQI(t)
    baseline_col = f"aqi_lag_{FORECAST_HORIZON_HOURS}h"
    baseline_predictions = df[baseline_col].iloc[split_idx:].values

    # Metrics
    metrics = evaluate(y_test.values, predictions, baseline_predictions)

    # Feature importances
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Save model and metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"xgb_{city}_{FORECAST_HORIZON_HOURS}h.joblib")
    joblib.dump(model, model_path)

    meta_path = os.path.join(MODELS_DIR, f"xgb_{city}_{FORECAST_HORIZON_HOURS}h_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "feature_columns": feature_cols,
            "metrics": metrics,
            "feature_importances": {k: float(v) for k, v in importance.items()},
            "train_size": len(X_train),
            "test_size": len(X_test),
        }, f, indent=2)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": importance,
        "predictions": predictions,
        "actuals": y_test.values,
        "baseline_predictions": baseline_predictions,
        "timestamps": df["timestamp"].iloc[split_idx:].values,
    }


def evaluate(
    actuals: np.ndarray,
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict:
    """
    Compute metrics for model and persistence baseline.
    """
    model_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    model_mae = mean_absolute_error(actuals, predictions)
    model_r2 = r2_score(actuals, predictions)

    baseline_rmse = np.sqrt(mean_squared_error(actuals, baseline_predictions))
    baseline_mae = mean_absolute_error(actuals, baseline_predictions)

    improvement = (1 - model_rmse / baseline_rmse) * 100

    return {
        "model_rmse": round(float(model_rmse), 2),
        "model_mae": round(float(model_mae), 2),
        "model_r2": round(float(model_r2), 4),
        "baseline_rmse": round(float(baseline_rmse), 2),
        "baseline_mae": round(float(baseline_mae), 2),
        "improvement_pct": round(float(improvement), 1),
    }


def load_forecaster(city: str = "kolkata"):
    """Load a trained model and its metadata."""
    model_path = os.path.join(MODELS_DIR, f"xgb_{city}_{FORECAST_HORIZON_HOURS}h.joblib")
    meta_path = os.path.join(MODELS_DIR, f"xgb_{city}_{FORECAST_HORIZON_HOURS}h_meta.json")

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta
