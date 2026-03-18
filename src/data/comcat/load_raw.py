from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import COMCAT_RAW_DIR


RECOMMENDED_COLUMNS = [
    "event_id",
    "time",
    "updated",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "status",
    "event_type",
    "net",
    "gap",
    "dmin",
    "rms",
    "nst",
]


def comcat_geojson_to_dataframe(filepath: str | Path) -> pd.DataFrame:
    """
    Read one ComCat GeoJSON file and convert it into a flat dataframe.
    """
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    rows: list[dict] = []

    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])

        longitude = coords[0] if len(coords) > 0 else None
        latitude = coords[1] if len(coords) > 1 else None
        depth_km = coords[2] if len(coords) > 2 else None

        rows.append(
            {
                "event_id": feature.get("id"),
                "time": props.get("time"),
                "updated": props.get("updated"),
                "latitude": latitude,
                "longitude": longitude,
                "depth_km": depth_km,
                "magnitude": props.get("mag"),
                "magnitude_type": props.get("magType"),
                "status": props.get("status"),
                "event_type": props.get("type"),
                "net": props.get("net"),
                "gap": props.get("gap"),
                "dmin": props.get("dmin"),
                "rms": props.get("rms"),
                "nst": props.get("nst"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=RECOMMENDED_COLUMNS)

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    df["updated"] = pd.to_datetime(df["updated"], unit="ms", utc=True, errors="coerce")

    df = df[RECOMMENDED_COLUMNS]
    df = df.sort_values("time").reset_index(drop=True)

    return df


def month_start(dt: datetime) -> datetime:
    return dt.replace(day=1)


def next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)


def iter_months(start_date: str, end_date: str):
    """
    Yield (year, month) pairs from start_date to end_date inclusive.
    Dates must be YYYY-MM-DD.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    current = month_start(start_dt)

    while current <= end_dt:
        yield current.year, current.month
        current = next_month(current)


def build_comcat_filepath(root_dir: str | Path, year: int, month: int) -> Path:
    """
    Build a monthly raw GeoJSON filepath like:
    data/raw/comcat/2025/comcat_2025_01.geojson
    """
    root_dir = Path(root_dir)
    return root_dir / str(year) / f"comcat_{year}_{month:02d}.geojson"


def load_comcat_date_range(
    start_date: str,
    end_date: str,
    root_dir: str | Path = COMCAT_RAW_DIR,
    skip_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and combine all monthly ComCat GeoJSON files whose month falls within
    the provided date range.
    """
    frames: list[pd.DataFrame] = []

    for year, month in iter_months(start_date, end_date):
        filepath = build_comcat_filepath(root_dir, year, month)

        if not filepath.exists():
            if skip_missing:
                if verbose:
                    print(f"[SKIP] Missing file: {filepath}")
                continue
            raise FileNotFoundError(f"Missing file: {filepath}")

        if verbose:
            print(f"[LOAD] {filepath}")

        df_month = comcat_geojson_to_dataframe(filepath)
        frames.append(df_month)

    if not frames:
        return pd.DataFrame(columns=RECOMMENDED_COLUMNS)

    df = pd.concat(frames, ignore_index=True)

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

    df = df[(df["time"] >= start_ts) & (df["time"] < end_ts)].copy()
    df = df.sort_values("time").reset_index(drop=True)

    return df


def build_and_save_raw_comcat_dataset(
    start_date: str,
    end_date: str,
    root_dir: str | Path = COMCAT_RAW_DIR,
    output_path: str | Path = COMCAT_RAW_DIR / "comcat.parquet",
    skip_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load raw monthly ComCat files for a date range, combine them,
    and save as CSV or Parquet.
    """
    df = load_comcat_date_range(
        start_date=start_date,
        end_date=end_date,
        root_dir=root_dir,
        skip_missing=skip_missing,
        verbose=verbose,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    if verbose:
        print(f"[SAVE] {len(df):,} rows -> {output_path}")

    return df


if __name__ == "__main__":
    df = build_and_save_raw_comcat_dataset(
        start_date="2025-01-01",
        end_date="2025-12-31",
        root_dir=COMCAT_RAW_DIR,
        output_path=COMCAT_RAW_DIR / "comcat_2025.parquet",
        skip_missing=True,
        verbose=True,
    )

    print(df.head())
    print(df.shape)