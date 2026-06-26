"""
Fetch Sentinel-5P NO2/SO2/aerosol data via Google Earth Engine.
Exports GeoTIFF rasters for map overlay.

Requires: authenticated GEE account.
Run `earthengine authenticate` first.

Usage:
    uv run python scripts/fetch_satellite.py
    uv run python scripts/fetch_satellite.py --city Kolkata --months 3
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from config.settings import CITIES, SATELLITE_DIR


def authenticate_gee():
    """Authenticate and initialize Google Earth Engine."""
    try:
        import ee
        ee.Initialize(project=os.environ.get("GEE_PROJECT_ID", "ee-vayu-drishti"))
        print("✓ GEE authenticated")
        return ee
    except Exception as e:
        print(f"✗ GEE auth failed: {e}")
        print("  Run: earthengine authenticate")
        print("  Set GEE_PROJECT_ID in .env")
        sys.exit(1)


def fetch_no2_monthly(ee, city_name: str, lat: float, lon: float, bbox: list, months: int = 3):
    """
    Fetch monthly average NO2 column density from Sentinel-5P.
    Saves results as JSON (lat/lon/value grids) for map overlay.
    """
    import numpy as np

    end_date = datetime.now()
    os.makedirs(SATELLITE_DIR, exist_ok=True)

    results = []

    for m in range(months):
        month_end = end_date - timedelta(days=30 * m)
        month_start = month_end - timedelta(days=30)

        start_str = month_start.strftime("%Y-%m-%d")
        end_str = month_end.strftime("%Y-%m-%d")
        month_label = month_start.strftime("%Y-%m")

        print(f"  Fetching NO2 for {month_label}...")

        try:
            # Define area of interest from bbox [lon_min, lat_min, lon_max, lat_max]
            aoi = ee.Geometry.Rectangle(bbox)

            # Query Sentinel-5P NO2 collection
            collection = (
                ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
                .filterDate(start_str, end_str)
                .filterBounds(aoi)
                .select("tropospheric_NO2_column_number_density")
            )

            count = collection.size().getInfo()
            if count == 0:
                print(f"    No images found for {month_label}")
                continue

            print(f"    Found {count} images, computing monthly mean...")

            # Compute monthly mean
            monthly_mean = collection.mean()

            # Sample on a grid within the bbox
            # Create a grid of ~25 points (5x5)
            lat_min, lon_min = bbox[1], bbox[0]
            lat_max, lon_max = bbox[3], bbox[2]

            grid_size = 8
            lats = np.linspace(lat_min, lat_max, grid_size)
            lons = np.linspace(lon_min, lon_max, grid_size)

            grid_points = []
            for lat_val in lats:
                for lon_val in lons:
                    grid_points.append(ee.Geometry.Point([lon_val, lat_val]))

            # Sample all points
            multi_point = ee.Geometry.MultiPoint(
                [[lon_val, lat_val] for lat_val in lats for lon_val in lons]
            )

            sampled = monthly_mean.sampleRegions(
                collection=ee.FeatureCollection(
                    [ee.Feature(p) for p in grid_points]
                ),
                scale=5500,  # Sentinel-5P resolution
            ).getInfo()

            # Extract values
            grid_data = []
            idx = 0
            for i, lat_val in enumerate(lats):
                for j, lon_val in enumerate(lons):
                    val = None
                    if idx < len(sampled.get("features", [])):
                        props = sampled["features"][idx].get("properties", {})
                        val = props.get("tropospheric_NO2_column_number_density")
                    grid_data.append({
                        "lat": float(lat_val),
                        "lon": float(lon_val),
                        "no2": float(val) if val is not None else None,
                    })
                    idx += 1

            results.append({
                "month": month_label,
                "start_date": start_str,
                "end_date": end_str,
                "image_count": count,
                "grid_size": grid_size,
                "grid": grid_data,
            })

            valid = sum(1 for g in grid_data if g["no2"] is not None)
            print(f"    ✓ {valid}/{len(grid_data)} valid grid points")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue

    # Save all months
    output_path = os.path.join(SATELLITE_DIR, f"{city_name.lower()}_no2.json")
    with open(output_path, "w") as f:
        json.dump({
            "city": city_name,
            "product": "Sentinel-5P NO2",
            "unit": "mol/m²",
            "months": results,
        }, f, indent=2)

    print(f"  ✓ Saved: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Sentinel-5P satellite data")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--months", type=int, default=3)
    args = parser.parse_args()

    ee = authenticate_gee()

    cities = [args.city] if args.city else list(CITIES.keys())

    for city_name in cities:
        if city_name not in CITIES:
            print(f"Unknown city: {city_name}")
            continue

        city = CITIES[city_name]
        print(f"\n{'='*50}")
        print(f"Fetching satellite data for {city_name}")
        print(f"{'='*50}")

        fetch_no2_monthly(
            ee, city_name,
            lat=city["lat"], lon=city["lon"],
            bbox=city["bbox"], months=args.months,
        )


if __name__ == "__main__":
    main()