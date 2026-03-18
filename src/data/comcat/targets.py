from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import COMCAT_INTERIM_DIR


MAINSHOCK_MIN_MAG = 4.0
AFTERSHOCK_MIN_MAG = 2.5
MAX_RADIUS_KM = 50.0
MAX_HOURS = 720.0


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two points on Earth in kilometers.
    Supports scalar mainshock coordinates and vectorized candidate coordinates.
    """
    R = 6371.0

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
    return R * c


def _build_no_aftershock_row(mainshock: pd.Series) -> dict:
    return {
        "mainshock_event_id": mainshock["event_id"],
        "mainshock_time": mainshock["time"],
        "mainshock_latitude": mainshock["latitude"],
        "mainshock_longitude": mainshock["longitude"],
        "mainshock_depth_km": mainshock["depth_km"],
        "mainshock_magnitude": mainshock["magnitude"],
        "mainshock_magnitude_type": mainshock.get("magnitude_type"),
        "mainshock_status": mainshock.get("status"),
        "mainshock_gap": mainshock.get("gap"),
        "mainshock_dmin": mainshock.get("dmin"),
        "mainshock_rms": mainshock.get("rms"),
        "mainshock_nst": mainshock.get("nst"),
        "next_aftershock_event_id": None,
        "next_aftershock_time": pd.NaT,
        "next_aftershock_magnitude": np.nan,
        "next_aftershock_distance_km": np.nan,
        "target_hours_raw": np.nan,
        "target_hours_capped": MAX_HOURS,
        "has_aftershock_within_720h": False,
        "is_capped_720h": True,
    }


def _build_aftershock_row(mainshock: pd.Series, next_aftershock: pd.Series) -> dict:
    target_hours = (
        next_aftershock["time"] - mainshock["time"]
    ).total_seconds() / 3600.0

    return {
        "mainshock_event_id": mainshock["event_id"],
        "mainshock_time": mainshock["time"],
        "mainshock_latitude": mainshock["latitude"],
        "mainshock_longitude": mainshock["longitude"],
        "mainshock_depth_km": mainshock["depth_km"],
        "mainshock_magnitude": mainshock["magnitude"],
        "mainshock_magnitude_type": mainshock.get("magnitude_type"),
        "mainshock_status": mainshock.get("status"),
        "mainshock_gap": mainshock.get("gap"),
        "mainshock_dmin": mainshock.get("dmin"),
        "mainshock_rms": mainshock.get("rms"),
        "mainshock_nst": mainshock.get("nst"),
        "next_aftershock_event_id": next_aftershock["event_id"],
        "next_aftershock_time": next_aftershock["time"],
        "next_aftershock_magnitude": next_aftershock["magnitude"],
        "next_aftershock_distance_km": next_aftershock["distance_km"],
        "target_hours_raw": target_hours,
        "target_hours_capped": min(target_hours, MAX_HOURS),
        "has_aftershock_within_720h": True,
        "is_capped_720h": False,
    }


def build_mainshock_target_dataset(
    comcat_df_clean: pd.DataFrame,
    mainshock_min_mag: float = MAINSHOCK_MIN_MAG,
    aftershock_min_mag: float = AFTERSHOCK_MIN_MAG,
    max_radius_km: float = MAX_RADIUS_KM,
    max_hours: float = MAX_HOURS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one-row-per-mainshock target dataset from cleaned ComCat events.
    """
    events_df = comcat_df_clean.sort_values("time").reset_index(drop=True).copy()

    mainshocks_df = (
        events_df.loc[events_df["magnitude"] >= mainshock_min_mag]
        .sort_values("time")
        .reset_index(drop=True)
    )

    if verbose:
        print(f"[INFO] Total cleaned events: {len(events_df):,}")
        print(f"[INFO] Candidate mainshocks (M >= {mainshock_min_mag}): {len(mainshocks_df):,}")

    target_rows: list[dict] = []

    for i, (_, mainshock) in enumerate(mainshocks_df.iterrows(), start=1):
        mainshock_time = mainshock["time"]
        mainshock_lat = mainshock["latitude"]
        mainshock_lon = mainshock["longitude"]

        future_events = events_df.loc[
            (events_df["time"] > mainshock_time)
            & (events_df["time"] <= mainshock_time + pd.Timedelta(hours=max_hours))
            & (events_df["magnitude"] >= aftershock_min_mag)
        ].copy()

        if future_events.empty:
            target_rows.append(_build_no_aftershock_row(mainshock))
            continue

        future_events["distance_km"] = haversine_km(
            mainshock_lat,
            mainshock_lon,
            future_events["latitude"].values,
            future_events["longitude"].values,
        )

        qualifying = future_events.loc[future_events["distance_km"] <= max_radius_km].copy()

        if qualifying.empty:
            target_rows.append(_build_no_aftershock_row(mainshock))
            continue

        next_aftershock = qualifying.sort_values("time").iloc[0]
        target_rows.append(_build_aftershock_row(mainshock, next_aftershock))

        if verbose and i % 5000 == 0:
            print(f"[PROGRESS] Processed {i:,} / {len(mainshocks_df):,} mainshocks")

    target_df = pd.DataFrame(target_rows).sort_values("mainshock_time").reset_index(drop=True)
    return target_df


def load_and_build_targets(
    input_path: str | Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load cleaned ComCat events from parquet and build target dataset.
    """
    input_path = Path(input_path)

    if verbose:
        print(f"[LOAD] {input_path}")

    comcat_df_clean = pd.read_parquet(input_path)
    target_df = build_mainshock_target_dataset(
        comcat_df_clean=comcat_df_clean,
        verbose=verbose,
    )
    return target_df


def save_mainshock_target_dataset(
    input_path: str | Path,
    output_path: str | Path = COMCAT_INTERIM_DIR / "mainshock_target_dataset.parquet",
    force_recompute: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load cleaned ComCat events, build mainshock target dataset, and save to parquet.
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_recompute:
        if verbose:
            print(f"[LOAD EXISTING] {output_path}")
        return pd.read_parquet(output_path)

    target_df = load_and_build_targets(
        input_path=input_path,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(output_path, index=False)

    if verbose:
        print(f"[SAVE] Target dataset -> {output_path}")
        print(f"[SAVE] Rows: {len(target_df):,}")

    return target_df


if __name__ == "__main__":
    target_df = save_mainshock_target_dataset(
        input_path=COMCAT_INTERIM_DIR / "comcat_clean_events_2025.parquet",
        output_path=COMCAT_INTERIM_DIR / "mainshock_target_dataset_2025.parquet",
        force_recompute=False,
        verbose=True,
    )

    print(target_df.head())
    print(target_df.shape)