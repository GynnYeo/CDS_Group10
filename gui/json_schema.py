from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_MINIMUM_KEYS = ["latitude", "longitude", "depth_km", "magnitude"]


def load_json_payload(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object.")

    # Optional nested formats
    if "earthquake" in payload and isinstance(payload["earthquake"], dict):
        payload = payload["earthquake"]

    return payload


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize common input names to the raw keys expected by the inference layer.
    """
    normalized = dict(payload)

    rename_map = {
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "depth": "depth_km",
        "mag": "magnitude",
    }

    for old_key, new_key in rename_map.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]

    # Optional default synthetic ID
    normalized.setdefault("trigger_event_id", "gui_input_event")

    missing = [k for k in REQUIRED_MINIMUM_KEYS if k not in normalized]
    if missing:
        raise ValueError(
            "JSON is missing required keys: "
            f"{missing}. Need at least latitude, longitude, depth_km, magnitude."
        )

    return normalized


JSON_TEMPLATE = {
    "trigger_event_id": "sim_2026_001",
    "trigger_time": "2026-04-10T14:22:00Z",
    "latitude": -3.12,
    "longitude": 128.45,
    "depth_km": 18.2,
    "magnitude": 5.9,
    "gap": 45.0,
    "dmin": 0.12,
    "rms": 0.45,
    "nst": 52
}