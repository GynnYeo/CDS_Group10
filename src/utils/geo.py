from __future__ import annotations

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two points on Earth in kilometers.
    Supports scalar-to-vector comparisons for label generation.
    """
    radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c
