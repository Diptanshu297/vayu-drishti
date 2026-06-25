"""
VayuDrishti — Benchmark page.
Model performance comparison across all cities.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json

from config.settings import CITIES, MODELS_DIR, FORECAST_HORIZON_HOURS


st.set_page_config(page_title="VayuDrishti — Benchmark", page_icon="📊", layout="wide")
st.title("📊 Model Benchmark")
st.caption(f"XGBoost {FORECAST_HORIZON_HOURS}h AQI Forecaster — Performance across cities")

# Load all city metrics
rows = []
for city_name in CITIES:
    city_lower = city_name.lower()
    meta_path = os.path.join(MODELS_DIR, f"xgb_{city_lower}_{FORECAST_HORIZON_HOURS}h_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        m = meta["metrics"]
        rows.append({
            "City": city_name,
            "Model RMSE": m["model_rmse"],
            "Baseline RMSE": m["baseline_rmse"],
            "Improvement (%)": m["improvement_pct"],
            "MAE": m["model_mae"],
            "R²": m["model_r2"],
            "Train size": meta.get("train_size", "—"),
            "Test size": meta.get("test_size", "—"),
        })

if not rows:
    st.warning("No trained models found. Run `uv run python scripts/train_all.py` first.")
    st.stop()

bench_df = pd.DataFrame(rows)

# --- RMSE comparison bar chart ---
st.subheader("RMSE: Model vs persistence baseline")

fig = go.Figure()

fig.add_trace(go.Bar(
    name="XGBoost",
    x=bench_df["City"],
    y=bench_df["Model RMSE"],
    marker_color="#3b82f6",
    text=bench_df["Model RMSE"],
    textposition="outside",
))

fig.add_trace(go.Bar(
    name="Persistence baseline",
    x=bench_df["City"],
    y=bench_df["Baseline RMSE"],
    marker_color="#94a3b8",
    text=bench_df["Baseline RMSE"],
    textposition="outside",
))

fig.update_layout(
    barmode="group",
    height=400,
    margin=dict(t=20, b=40),
    yaxis_title="RMSE (lower = better)",
    legend=dict(orientation="h", y=1.08),
)

st.plotly_chart(fig, use_container_width=True)

# --- Improvement chart ---
st.subheader("Improvement over baseline")

colors = ["#22c55e" if v > 0 else "#ef4444" for v in bench_df["Improvement (%)"]]

fig2 = go.Figure(go.Bar(
    x=bench_df["City"],
    y=bench_df["Improvement (%)"],
    marker_color=colors,
    text=[f"{v}%" for v in bench_df["Improvement (%)"]],
    textposition="outside",
))

fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(
    height=350,
    margin=dict(t=20, b=40),
    yaxis_title="Improvement (%)",
)

st.plotly_chart(fig2, use_container_width=True)

# --- Full metrics table ---
st.subheader("Full metrics")
st.dataframe(
    bench_df.style.format({
        "Model RMSE": "{:.2f}",
        "Baseline RMSE": "{:.2f}",
        "Improvement (%)": "{:.1f}",
        "MAE": "{:.2f}",
        "R²": "{:.4f}",
    }).background_gradient(subset=["Improvement (%)"], cmap="RdYlGn"),
    hide_index=True,
    use_container_width=True,
)

# --- Key takeaways ---
st.divider()
st.subheader("Key findings")

best = bench_df.loc[bench_df["Improvement (%)"].idxmax()]
worst = bench_df.loc[bench_df["Improvement (%)"].idxmin()]

st.markdown(f"""
**Best performer:** {best['City']} — {best['Improvement (%)']}% improvement, R²={best['R²']}

**Weakest performer:** {worst['City']} — {worst['Improvement (%)']}% improvement

**Insight:** Inland cities (Kolkata, Delhi, Lucknow, Bengaluru) show strong improvement because
24-hour AQI patterns are driven by predictable traffic cycles and atmospheric stability.
Coastal cities (Mumbai, Chennai) show weaker improvement because sea breeze dynamics
introduce variability that the current feature set doesn't capture. Adding oceanic features
(sea surface temperature, tidal wind patterns) is a deployment-stage refinement.
""")
