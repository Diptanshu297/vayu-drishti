"""
VayuDrishti — Central configuration.
All city definitions, API endpoints, AQI breakpoints, and model parameters.
"""

# ──────────────────────────────────────────────
# Cities
# ──────────────────────────────────────────────
CITIES = {
    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "lang": "bengali",
        "bbox": [88.25, 22.45, 88.45, 22.65],  # for Sentinel-5P queries
    },
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "lang": "hindi",
        "bbox": [76.95, 28.40, 77.35, 28.85],
    },
    "Bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "lang": "kannada",
        "bbox": [77.45, 12.85, 77.75, 13.10],
    },
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "lang": "hindi",
        "bbox": [72.75, 18.90, 73.00, 19.25],
    },
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "lang": "hindi",
        "bbox": [80.15, 12.95, 80.35, 13.20],
    },
    "Lucknow": {
        "lat": 26.8467,
        "lon": 80.9462,
        "lang": "hindi",
        "bbox": [80.80, 26.75, 81.10, 26.95],
    },
}

DEFAULT_CITY = "Kolkata"

# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENWEATHER_AQI_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

# ──────────────────────────────────────────────
# AQI Breakpoints (India NAQI standard)
# Format: (C_low, C_high, I_low, I_high)
# ──────────────────────────────────────────────
AQI_BREAKPOINTS = {
    "pm2_5": [  # 24-hour average, μg/m³
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "pm10": [  # 24-hour average, μg/m³
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "no2": [  # 24-hour average, μg/m³
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 500, 401, 500),
    ],
    "so2": [  # 24-hour average, μg/m³
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2100, 401, 500),
    ],
    "co": [  # 8-hour average, mg/m³
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    "o3": [  # 8-hour average, μg/m³
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}

AQI_CATEGORIES = [
    {"range": (0, 50), "label": "Good", "color": "#55a868"},
    {"range": (51, 100), "label": "Satisfactory", "color": "#a3c853"},
    {"range": (101, 200), "label": "Moderate", "color": "#fff44f"},
    {"range": (201, 300), "label": "Poor", "color": "#ff8c00"},
    {"range": (301, 400), "label": "Very poor", "color": "#ff4444"},
    {"range": (401, 500), "label": "Severe", "color": "#8b0000"},
]

# ──────────────────────────────────────────────
# Model Parameters
# ──────────────────────────────────────────────
FORECAST_HORIZON_HOURS = 24
LAG_FEATURES = [1, 6, 24, 168]  # hours
ROLLING_WINDOWS = [6, 24]  # hours

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 20,
    "random_state": 42,
}

TRAIN_TEST_SPLIT_RATIO = 0.8

# ──────────────────────────────────────────────
# Dispersion Model Parameters
# ──────────────────────────────────────────────
PASQUILL_GIFFORD = {
    "A": {"ay": 0.3658, "by": 0.9024, "az": 0.192, "bz": 1.20},
    "B": {"ay": 0.2751, "by": 0.9031, "az": 0.156, "bz": 1.00},
    "C": {"ay": 0.2090, "by": 0.9031, "az": 0.116, "bz": 0.91},
    "D": {"ay": 0.1471, "by": 0.9031, "az": 0.079, "bz": 0.85},
    "E": {"ay": 0.1046, "by": 0.9031, "az": 0.063, "bz": 0.78},
    "F": {"ay": 0.0722, "by": 0.9031, "az": 0.053, "bz": 0.71},
}

DEFAULT_STACK_HEIGHT = 30  # meters
DEFAULT_RECEPTOR_HEIGHT = 1.5  # meters (breathing height)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GEOJSON_DIR = os.path.join(DATA_DIR, "geojson")
SATELLITE_DIR = os.path.join(DATA_DIR, "satellite")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")