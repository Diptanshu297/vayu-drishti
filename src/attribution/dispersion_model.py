"""
Gaussian plume atmospheric dispersion model.
Computes pollutant concentration at any point from a point source,
and performs multi-source attribution.
"""

import numpy as np

from config.settings import PASQUILL_GIFFORD, DEFAULT_STACK_HEIGHT, DEFAULT_RECEPTOR_HEIGHT
from src.utils.geo_utils import latlon_to_plume_coords, classify_stability


def sigma_y(x: float, stability_class: str) -> float:
    """Horizontal dispersion coefficient (meters)."""
    c = PASQUILL_GIFFORD[stability_class]
    return c["ay"] * (x ** c["by"])


def sigma_z(x: float, stability_class: str) -> float:
    """Vertical dispersion coefficient (meters)."""
    c = PASQUILL_GIFFORD[stability_class]
    return c["az"] * (x ** c["bz"])


def gaussian_plume(
    Q: float,
    u: float,
    H: float,
    x: float,
    y: float,
    z: float,
    stability_class: str,
) -> float:
    """
    Compute pollutant concentration at point (x, y, z) from a point source.

    Args:
        Q: emission rate (g/s)
        u: wind speed (m/s)
        H: effective stack height (m)
        x: downwind distance from source (m), must be > 0
        y: crosswind distance (m), 0 = directly downwind
        z: receptor height (m), typically 1.5 (breathing height)
        stability_class: 'A' through 'F'

    Returns:
        Concentration in g/m³
    """
    if x <= 0 or u <= 0.5:
        return 0.0

    sy = sigma_y(x, stability_class)
    sz = sigma_z(x, stability_class)

    coeff = Q / (2 * np.pi * u * sy * sz)
    lateral = np.exp(-0.5 * (y / sy) ** 2)
    vertical = (
        np.exp(-0.5 * ((z - H) / sz) ** 2)
        + np.exp(-0.5 * ((z + H) / sz) ** 2)  # ground reflection
    )

    return coeff * lateral * vertical


def compute_source_attribution(
    sources: list[dict],
    receptor_lat: float,
    receptor_lon: float,
    wind_speed: float,
    wind_direction: float,
    stability_class: str,
    receptor_height: float = DEFAULT_RECEPTOR_HEIGHT,
) -> dict:
    """
    Compute attribution percentages from multiple sources at a receptor point.

    Args:
        sources: list of dicts with keys: name, lat, lon, emission_rate, stack_height
        receptor_lat, receptor_lon: monitoring station location
        wind_speed: m/s
        wind_direction: degrees (meteorological, "wind FROM")
        stability_class: 'A' through 'F'

    Returns:
        dict with:
            - attributions: list of {name, concentration, percentage}
            - total_concentration: sum of all contributions
            - stability_class: used class
    """
    contributions = []

    for src in sources:
        downwind, crosswind = latlon_to_plume_coords(
            src["lat"], src["lon"],
            receptor_lat, receptor_lon,
            wind_direction,
        )

        if downwind > 0:
            conc = gaussian_plume(
                Q=src.get("emission_rate", 50.0),
                u=wind_speed,
                H=src.get("stack_height", DEFAULT_STACK_HEIGHT),
                x=downwind,
                y=crosswind,
                z=receptor_height,
                stability_class=stability_class,
            )
        else:
            conc = 0.0

        contributions.append({
            "name": src["name"],
            "source_type": src.get("source_type", "unknown"),
            "concentration": conc,
            "downwind_distance": downwind,
            "crosswind_distance": crosswind,
        })

    total = sum(c["concentration"] for c in contributions)

    for c in contributions:
        c["percentage"] = round((c["concentration"] / total * 100), 1) if total > 0 else 0.0

    # Sort by contribution (highest first)
    contributions.sort(key=lambda x: x["concentration"], reverse=True)

    return {
        "attributions": contributions,
        "total_concentration": total,
        "stability_class": stability_class,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
    }


def generate_plume_grid(
    source: dict,
    wind_speed: float,
    wind_direction: float,
    stability_class: str,
    grid_size: int = 50,
    max_distance: float = 10000,
) -> dict:
    """
    Generate a 2D concentration grid for visualization on a map.

    Returns dict with:
        - lats, lons: 1D arrays of grid coordinates
        - concentrations: 2D array of concentrations
    """
    src_lat = source["lat"]
    src_lon = source["lon"]

    # Create grid around source
    lat_range = max_distance / 111_320
    lon_range = max_distance / (111_320 * np.cos(np.radians(src_lat)))

    lats = np.linspace(src_lat - lat_range, src_lat + lat_range, grid_size)
    lons = np.linspace(src_lon - lon_range, src_lon + lon_range, grid_size)

    concentrations = np.zeros((grid_size, grid_size))

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            downwind, crosswind = latlon_to_plume_coords(
                src_lat, src_lon, lat, lon, wind_direction
            )
            if downwind > 50:  # skip very close to source
                concentrations[i, j] = gaussian_plume(
                    Q=source.get("emission_rate", 50.0),
                    u=wind_speed,
                    H=source.get("stack_height", DEFAULT_STACK_HEIGHT),
                    x=downwind,
                    y=crosswind,
                    z=DEFAULT_RECEPTOR_HEIGHT,
                    stability_class=stability_class,
                )

    return {
        "lats": lats,
        "lons": lons,
        "concentrations": concentrations,
    }
