"""
VayuDrishti — Forecast page.
Shows 24h AQI forecast vs actuals vs persistence baseline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

from config.settings import CITIES, MODELS_DIR, PROCESSED_DIR, FORECAST_HORIZON_HOURS


st.set_page_config(page_title="VayuDrishti — Forecast", page_icon="📈", layout="wide")
st.title("📈 24-Hour AQI Forecast")

city = st.selectbox("Select city", list(CITIES.keys()), key="forecast_city")
city_lower = city.lower()

# Load processed data and model metadata
features_path = os.path.join(PROCESSED_DIR, f"features_{city_lower}.csv")
meta_path = os.path.join(MODELS_DIR, f"xgb_{city_lower}_{FORECAST_HORIZON_HOURS}h_meta.json")

if not os.path.exists(features_path) or not os.path.exists(meta_path):
    st.warning(f"No trained model found for {city}. Run `uv run python scripts/train_all.py` first.")
    st.stop()

# Load data
df = pd.read_csv(features_path, parse_dates=["timestamp"])
with open(meta_path) as f:
    meta = json.load(f)

metrics = meta["metrics"]
importances = meta["feature_importances"]

# Recompute predictions for visualization
from src.models.train_forecaster import load_forecaster
from src.feature_engineering.build_features import get_feature_columns

model, _ = load_forecaster(city_lower)
feature_cols = get_feature_columns()
target_col = f"target_aqi_{FORECAST_HORIZON_HOURS}h"

split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy()
predictions = model.predict(test_df[feature_cols])

# --- Metrics cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model RMSE", f"{metrics['model_rmse']}")
col2.metric("Baseline RMSE", f"{metrics['baseline_rmse']}")
col3.metric("Improvement", f"{metrics['improvement_pct']}%",
            delta=f"{metrics['improvement_pct']}%",
            delta_color="normal" if metrics['improvement_pct'] > 0 else "inverse")
col4.metric("R²", f"{metrics['model_r2']}")

# --- Prediction vs Actual chart ---
st.subheader("Predicted vs actual AQI (test set)")

# Show last N days for readability
days_to_show = st.slider("Days to display", 7, 60, 30)
hours = days_to_show * 24
plot_df = test_df.iloc[-hours:].copy()
plot_preds = predictions[-hours:]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df["timestamp"],
    y=plot_df[target_col],
    mode="lines",
    name="Actual AQI",
    line=dict(color="#3b82f6", width=2),
))

fig.add_trace(go.Scatter(
    x=plot_df["timestamp"],
    y=plot_preds,
    mode="lines",
    name="XGBoost prediction",
    line=dict(color="#f59e0b", width=2),
))

fig.add_trace(go.Scatter(
    x=plot_df["timestamp"],
    y=plot_df[f"aqi_lag_{FORECAST_HORIZON_HOURS}h"],
    mode="lines",
    name="Persistence baseline",
    line=dict(color="#94a3b8", width=1, dash="dash"),
))

# AQI category bands
category_bands = [
    (0, 50, "Good", "rgba(85,168,104,0.08)"),
    (50, 100, "Satisfactory", "rgba(163,200,83,0.08)"),
    (100, 200, "Moderate", "rgba(255,244,79,0.08)"),
    (200, 300, "Poor", "rgba(255,140,0,0.08)"),
    (300, 400, "Very Poor", "rgba(255,68,68,0.08)"),
]

for low, high, label, color in category_bands:
    fig.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0,
                  annotation_text=label, annotation_position="right")

fig.update_layout(
    height=450,
    margin=dict(t=20, b=40),
    xaxis_title="Date",
    yaxis_title="AQI",
    legend=dict(orientation="h", y=1.08),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# --- Feature importance ---
st.subheader("Feature importance")

imp_df = pd.DataFrame([
    {"Feature": k.replace("_", " ").title(), "Importance": v}
    for k, v in importances.items()
]).head(10)

fig_imp = go.Figure(go.Bar(
    x=imp_df["Importance"],
    y=imp_df["Feature"],
    orientation="h",
    marker_color="#3b82f6",
))
fig_imp.update_layout(
    height=350,
    margin=dict(t=10, b=20, l=150),
    xaxis_title="Importance",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_imp, use_container_width=True)

# --- Error distribution ---
st.subheader("Prediction error distribution")

errors = plot_df[target_col].values - plot_preds
fig_err = go.Figure(go.Histogram(
    x=errors,
    nbinsx=40,
    marker_color="#3b82f6",
    opacity=0.7,
))
fig_err.add_vline(x=0, line_dash="dash", line_color="red")
fig_err.update_layout(
    height=300,
    margin=dict(t=10, b=30),
    xaxis_title="Prediction Error (Actual - Predicted)",
    yaxis_title="Count",
)
st.plotly_chart(fig_err, use_container_width=True)
