"""
Extract industrial zones, major roads, and known pollution sources
from OpenStreetMap via osmnx.
"""

import json
import os

import osmnx as ox
import geopandas as gpd

from config.settings import GEOJSON_DIR


def get_industrial_zones(city_name: str, lat: float, lon: float, radius: int = 10000) -> gpd.GeoDataFrame:
    """
    Fetch industrial land use polygons from OSM within radius of city center.

    Args:
        city_name: for caching
        lat, lon: city center
        radius: search radius in meters

    Returns:
        GeoDataFrame with industrial zone polygons
    """
    cache_path = os.path.join(GEOJSON_DIR, f"{city_name.lower()}_industrial.geojson")
    if os.path.exists(cache_path):
        return gpd.read_file(cache_path)

    tags = {"landuse": "industrial"}
    gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)

    if len(gdf) > 0:
        # Keep only polygons, compute centroids
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf["centroid_lat"] = gdf.geometry.centroid.y
        gdf["centroid_lon"] = gdf.geometry.centroid.x
        gdf["source_type"] = "industrial"

        os.makedirs(GEOJSON_DIR, exist_ok=True)
        gdf.to_file(cache_path, driver="GeoJSON")

    return gdf


def get_major_roads(city_name: str, lat: float, lon: float, radius: int = 10000) -> gpd.GeoDataFrame:
    """
    Fetch major road network for traffic-based pollution estimation.
    """
    cache_path = os.path.join(GEOJSON_DIR, f"{city_name.lower()}_roads.geojson")
    if os.path.exists(cache_path):
        return gpd.read_file(cache_path)

    G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False)

    # Keep only major roads
    major = edges[edges["highway"].isin([
        "trunk", "primary", "secondary", "motorway",
        "trunk_link", "primary_link", "motorway_link",
    ])].copy()

    os.makedirs(GEOJSON_DIR, exist_ok=True)
    major.to_file(cache_path, driver="GeoJSON")

    return major


def get_pollution_sources(city_name: str, lat: float, lon: float, radius: int = 10000) -> list[dict]:
    """
    Get a list of known/estimated pollution source locations.
    Returns list of dicts with: name, lat, lon, source_type, estimated_emission_rate (g/s)
    """
    gdf = get_industrial_zones(city_name, lat, lon, radius)

    sources = []
    for _, row in gdf.iterrows():
        sources.append({
            "name": row.get("name", f"Industrial zone ({row['centroid_lat']:.3f}, {row['centroid_lon']:.3f})"),
            "lat": row["centroid_lat"],
            "lon": row["centroid_lon"],
            "source_type": "industrial",
            "emission_rate": 50.0,  # estimated default, g/s
            "stack_height": 30.0,
        })

    return sources
