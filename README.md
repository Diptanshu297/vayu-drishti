# VayuDrishti

**AI-Powered Urban Air Quality Intelligence for Smart City Intervention**

ET AI Hackathon 2026 — Problem Statement #5

## What it does

Fuses satellite imagery, ground sensor data, and meteorological forecasts to:
- **Predict** AQI 24 hours ahead at hyperlocal resolution
- **Attribute** pollution to specific sources using atmospheric dispersion physics
- **Generate** health advisories in Bengali, Hindi, and Kannada
- **Compare** air quality across 6 Indian cities

## Tech stack

| Layer | Technology |
|-------|------------|
| Satellite data | Sentinel-5P via Google Earth Engine |
| Ground sensors | CAAQMS via Open-Meteo API |
| Forecasting | XGBoost (24h AQI prediction) |
| Source attribution | Gaussian plume dispersion model |
| Agent orchestration | Custom multi-agent pipeline (Anthropic SDK) |
| Multilingual advisory | Anthropic Claude API |
| Dashboard | Streamlit + Folium + Plotly |

## Quick start

```bash
# Clone and install
git clone https://github.com/Diptanshu297/vayu-drishti.git
cd vayu-drishti
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Fetch historical data (runs ~2 min)
uv run python scripts/fetch_historical_data.py --city Kolkata --months 12

# Train the forecasting model
uv run python scripts/train_model.py

# Launch dashboard
uv run streamlit run app/🏠_Home.py
```

## Project structure

```
vayu-drishti/
├── config/             # City definitions, API endpoints, model params
├── src/
│   ├── data_ingestion/ # API clients (Open-Meteo, OpenWeather, OSM)
│   ├── feature_engineering/  # Lag features, rolling stats, time encoding
│   ├── models/         # XGBoost training + evaluation
│   ├── attribution/    # Gaussian plume dispersion model
│   ├── agents/         # CrewAI multi-agent orchestration
│   ├── advisory/       # Claude API multilingual advisory generator
│   └── utils/          # AQI calculator, geo utilities
├── app/                # Streamlit dashboard pages
├── scripts/            # Data fetching + model training scripts
├── data/               # Raw, processed, GeoJSON, satellite data
├── models/             # Saved trained models
└── notebooks/          # EDA and experiments
```

## Cities covered

Kolkata · Delhi · Bengaluru · Mumbai · Chennai · Lucknow

## Team

Built for ET AI Hackathon 2026
