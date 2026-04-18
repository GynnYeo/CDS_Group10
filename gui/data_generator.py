from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dataset(dataset_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "datasets" / f"{dataset_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "trigger_latitude",
        "trigger_longitude",
        "trigger_depth_km",
        "trigger_magnitude",
    ]
    return df.dropna(subset=[c for c in required if c in df.columns])


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "trigger_event_id": str(row.get("trigger_event_id", "demo_event")),
        "trigger_time": str(row.get("trigger_time", "")),
        "latitude": float(row.get("trigger_latitude")),
        "longitude": float(row.get("trigger_longitude")),
        "depth_km": float(row.get("trigger_depth_km")),
        "magnitude": float(row.get("trigger_magnitude")),
        "trigger_month": row.get("trigger_month"),
        "trigger_dayofyear": row.get("trigger_dayofyear"),
        "trigger_hour": row.get("trigger_hour"),
        "trigger_year_feature": row.get("trigger_year_feature"),
        "gap": row.get("gap"),
        "dmin": row.get("dmin"),
        "rms": row.get("rms"),
        "nst": row.get("nst"),
    }


def generate_demo_jsons(dataset_name: str, n_files: int = 4):
    df = load_dataset(dataset_name)
    df = filter_valid_rows(df)

    output_dir = PROJECT_ROOT / "gui" / "demo_jsons"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = df.sample(n=n_files, random_state=42).reset_index(drop=True)

    paths = []

    for i, row in samples.iterrows():
        payload = row_to_payload(row)
        file_path = output_dir / f"demo_earthquake_{i+1}.json"

        with open(file_path, "w") as f:
            json.dump(payload, f, indent=2)

        paths.append(file_path)

    return paths


if __name__ == "__main__":
    files = generate_demo_jsons("earthquake_aftershock_v2_gcmt", 4)
    print("Generated JSON files:")
    for f in files:
        print(f)