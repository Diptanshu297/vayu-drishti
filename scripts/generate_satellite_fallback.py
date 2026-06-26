"""
Generate spatial pollution grid as a satellite data fallback.
Uses Open-Meteo air quality API at multiple grid points to create
a spatial NO2/PM2.5 surface — functionally similar to Sentinel-5P
but sourced from CAMS reanalysis instead of raw satellite imagery.

Use this when GEE access is unavailable.

Usage:
    uv run python scripts/generate_satellite_fallback.py
    uv run python scripts/generate_satellite_fallback.py --city Kolkata
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

from config.settings import CITIES, SATELLITE_DIR, OPEN_METEO_AQI_URL


def generate_spatial_grid(city_name: str, lat: float, lon: float, bbox: list, grid_size: int = 6):
    """
    Query Open-Meteo at grid_size × grid_size points within the city bbox
    to create a spatial pollution surface.
    """
    lat_min, lon_min = bbox[1], bbox[0]
    lat_max, lon_max = bbox[3], bbox[2]

    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)

    grid = []
    total = grid_size * grid_size
    done = 0

    for lat_val in lats:
        for lon_val in lons:
            done += 1
            try:
                params = {
                    "latitude": round(float(lat_val), 4),
                    "longitude": round(float(lon_val), 4),
                    "hourly": "nitrogen_dioxide,pm2_5,pm10,sulphur_dioxide",
                    "timezone": "Asia/Kolkata",
                    "forecast_days": 1,
                }
                resp = requests.get(OPEN_METEO_AQI_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                hourly = data["hourly"]
                # Take the current hour's reading (index 0)
                grid.append({
                    "lat": round(float(lat_val), 4),
                    "lon": round(float(lon_val), 4),
                    "no2": hourly["nitrogen_dioxide"][0],
                    "pm2_5": hourly["pm2_5"][0],
                    "pm10": hourly["pm10"][0],
                    "so2": hourly["sulphur_dioxide"][0],
                })

                print(f"    [{done}/{total}] ({lat_val:.3f}, {lon_val:.3f}) "
                      f"NO2={hourly['nitrogen_dioxide'][0]:.1f} PM2.5={hourly['pm2_5'][0]:.1f}")

            except Exception as e:
                print(f"    [{done}/{total}] ({lat_val:.3f}, {lon_val:.3f}) failed: {e}")
                grid.append({
                    "lat": round(float(lat_val), 4),
                    "lon": round(float(lon_val), 4),
                    "no2": None,
                    "pm2_5": None,
                    "pm10": None,
                    "so2": None,
                })

            # Rate limiting — be nice to the free API
            time.sleep(0.3)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate spatial pollution grid")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--grid-size", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(SATELLITE_DIR, exist_ok=True)

    cities = [args.city] if args.city else list(CITIES.keys())

    for city_name in cities:
        if city_name not in CITIES:
            print(f"Unknown city: {city_name}")
            continue

        city = CITIES[city_name]
        print(f"\n{'='*50}")
        print(f"Generating spatial grid for {city_name}")
        print(f"  Grid: {args.grid_size}×{args.grid_size} = {args.grid_size**2} points")
        print(f"{'='*50}")

        grid = generate_spatial_grid(
            city_name,
            lat=city["lat"], lon=city["lon"],
            bbox=city["bbox"], grid_size=args.grid_size,
        )

        output = {
            "city": city_name,
            "product": "CAMS reanalysis (Open-Meteo spatial grid)",
            "unit": "μg/m³",
            "grid_size": args.grid_size,
            "grid": grid,
            "source": "fallback",
        }

        output_path = os.path.join(SATELLITE_DIR, f"{city_name.lower()}_spatial.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        valid = sum(1 for g in grid if g["no2"] is not None)
        print(f"\n  ✓ {valid}/{len(grid)} valid points → {output_path}")


if __name__ == "__main__":
    main()