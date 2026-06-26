"""
AQI calculation using India's National Air Quality Index (NAQI) standard.
Computes sub-indices per pollutant and overall AQI.
"""

from config.settings import AQI_BREAKPOINTS, AQI_CATEGORIES


def compute_sub_index(pollutant: str, concentration: float) -> int | None:
    """
    Compute AQI sub-index for a single pollutant.

    Uses linear interpolation between NAQI breakpoints:
        I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low

    Args:
        pollutant: one of 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3'
        concentration: measured concentration in μg/m³ (mg/m³ for CO)

    Returns:
        Sub-index (0-500) or None if pollutant not recognized or out of range.
    """
    import math

    if pollutant not in AQI_BREAKPOINTS:
        return None

    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return None

    breakpoints = AQI_BREAKPOINTS[pollutant]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            sub_index = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(sub_index)

    # Concentration exceeds all breakpoints
    if concentration > breakpoints[-1][1]:
        return 500

    return None


def compute_aqi(concentrations: dict[str, float]) -> dict:
    """
    Compute overall AQI from multiple pollutant concentrations.

    Args:
        concentrations: dict of {pollutant: concentration}
            e.g. {"pm2_5": 85.0, "pm10": 120.0, "no2": 45.0}

    Returns:
        dict with:
            - aqi: overall AQI value (max of sub-indices)
            - dominant_pollutant: which pollutant drives the AQI
            - category: AQI category label
            - color: hex color for the category
            - sub_indices: dict of all computed sub-indices
    """
    sub_indices = {}

    for pollutant, conc in concentrations.items():
        try:
            if conc is not None and conc == conc and float(conc) >= 0:  # conc != conc catches NaN
                si = compute_sub_index(pollutant, float(conc))
                if si is not None:
                    sub_indices[pollutant] = si
        except (TypeError, ValueError):
            continue

    if not sub_indices:
        return {
            "aqi": None,
            "dominant_pollutant": None,
            "category": "Unknown",
            "color": "#999999",
            "sub_indices": {},
        }

    aqi = max(sub_indices.values())
    dominant = max(sub_indices, key=sub_indices.get)
    category, color = _get_category(aqi)

    return {
        "aqi": aqi,
        "dominant_pollutant": dominant,
        "category": category,
        "color": color,
        "sub_indices": sub_indices,
    }


def _get_category(aqi: int) -> tuple[str, str]:
    """Get AQI category label and color."""
    for cat in AQI_CATEGORIES:
        low, high = cat["range"]
        if low <= aqi <= high:
            return cat["label"], cat["color"]
    return "Severe", "#8b0000"