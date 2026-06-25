"""
Geospatial utilities: coordinate transforms, distance calculations,
and plume coordinate conversion.
"""

import numpy as np


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points in meters.
    """
    R = 6_371_000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def latlon_to_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """
    Approximate conversion of lat/lon difference to meters.
    Returns (dx_meters, dy_meters) where:
        dx = east-west distance
        dy = north-south distance
    """
    dy = (lat2 - lat1) * 111_320
    dx = (lon2 - lon1) * 111_320 * np.cos(np.radians((lat1 + lat2) / 2))
    return dx, dy


def latlon_to_plume_coords(
    src_lat: float, src_lon: float,
    rec_lat: float, rec_lon: float,
    wind_from_deg: float,
) -> tuple[float, float]:
    """
    Convert source→receptor vector into plume-aligned coordinates.

    Args:
        src_lat, src_lon: source (emission) location
        rec_lat, rec_lon: receptor (monitoring station) location
        wind_from_deg: meteorological wind direction (degrees, "wind FROM")

    Returns:
        (downwind, crosswind) in meters.
        downwind > 0 means receptor is downwind of source.
    """
    # Vector from source to receptor in meters
    dx, dy = latlon_to_meters(src_lat, src_lon, rec_lat, rec_lon)

    # Convert "wind from" to "wind toward" (direction plume travels)
    wind_toward_rad = np.radians((wind_from_deg + 180) % 360)

    # Rotate into wind-aligned coordinates
    # wind_toward_rad: 0=N, 90=E, 180=S, 270=W
    downwind = dx * np.sin(wind_toward_rad) + dy * np.cos(wind_toward_rad)
    crosswind = -dx * np.cos(wind_toward_rad) + dy * np.sin(wind_toward_rad)

    return downwind, crosswind


def classify_stability(wind_speed: float, hour: int, cloud_cover: float = 0.5) -> str:
    """
    Determine Pasquill-Gifford atmospheric stability class.

    Simplified classification using wind speed and time of day.

    Args:
        wind_speed: in m/s
        hour: hour of day (0-23)
        cloud_cover: fraction (0-1), 0=clear, 1=overcast

    Returns:
        Stability class 'A' through 'F'
    """
    is_daytime = 6 <= hour <= 18
    strong_sun = is_daytime and cloud_cover < 0.3
    moderate_sun = is_daytime and 0.3 <= cloud_cover < 0.7
    clear_night = not is_daytime and cloud_cover < 0.5

    if is_daytime:
        if wind_speed < 2:
            return "A" if strong_sun else "B"
        elif wind_speed < 3:
            return "A" if strong_sun else ("B" if moderate_sun else "C")
        elif wind_speed < 5:
            return "B" if strong_sun else ("C" if moderate_sun else "D")
        elif wind_speed < 6:
            return "C" if strong_sun else "D"
        else:
            return "D"
    else:
        if wind_speed < 2:
            return "F" if clear_night else "E"
        elif wind_speed < 3:
            return "F" if clear_night else "E"
        elif wind_speed < 5:
            return "E"
        else:
            return "D"
