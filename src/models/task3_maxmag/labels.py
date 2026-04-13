from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.utils.geo import haversine_km
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import INTERIM_COMCAT_DIR, INTERIM_TRIGGERS_DIR, PROCESSED_DIR


TASK3_MAXMAG_LABEL_DIR = PROCESSED_DIR / "task3_maxmag"
TASK3_MAXMAG_HORIZONS = (24, 72)
TASK3_MAXMAG_RADIUS_KM = 50.0
TASK3_MAXMAG_MIN_AFTERSHOCK_MAG = 2.5


def _slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("dataset_name must contain at least one letter or number.")
    return slug


def _target_col(horizon: int) -> str:
    return f"max_aftershock_mag_{horizon}h"


def _label_path(dataset_slug: str, min_aftershock_magnitude: float) -> Path:
    magnitude_slug = str(min_aftershock_magnitude).replace(".", "_")
    return TASK3_MAXMAG_LABEL_DIR / (
        f"{dataset_slug}__max_aftershock_mag_labels_m{magnitude_slug}.parquet"
    )


def _required_interim_paths(dataset_slug: str) -> tuple[Path, Path]:
    triggers_path = INTERIM_TRIGGERS_DIR / f"{dataset_slug}__triggers.parquet"
    comcat_path = INTERIM_COMCAT_DIR / f"{dataset_slug}__comcat_clean.parquet"
    missing = [path for path in (triggers_path, comcat_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Task 3 max-magnitude label construction requires interim artifacts. "
            f"Missing: {missing_text}"
        )
    return triggers_path, comcat_path


def build_max_aftershock_magnitude_labels(
    triggers_df: pd.DataFrame,
    events_df: pd.DataFrame,
    min_aftershock_magnitude: float = TASK3_MAXMAG_MIN_AFTERSHOCK_MAG,
    horizons_hours: tuple[int, ...] = TASK3_MAXMAG_HORIZONS,
    max_radius_km: float = TASK3_MAXMAG_RADIUS_KM,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build sidecar regression labels for largest aftershock magnitude.

    Labels are left as NaN when no qualifying aftershock occurs within the
    requested horizon. This keeps the task as a clean conditional regression
    problem rather than mixing "no aftershock" with a fake magnitude value.
    """
    events = events_df.copy()
    events["time"] = pd.to_datetime(events["time"], utc=True, errors="coerce")
    events = events.loc[events["time"].notna()].copy()

    triggers = triggers_df.copy()
    triggers["trigger_time"] = pd.to_datetime(triggers["trigger_time"], utc=True, errors="coerce")
    triggers = triggers.loc[triggers["trigger_time"].notna()].copy()

    max_horizon = max(horizons_hours)
    rows: list[dict[str, float | str]] = []

    for i, (_, trigger) in enumerate(triggers.iterrows(), start=1):
        trigger_time = trigger["trigger_time"]
        future_events = events.loc[
            (events["time"] > trigger_time)
            & (events["time"] <= trigger_time + pd.Timedelta(hours=max_horizon))
            & (events["magnitude"] >= min_aftershock_magnitude)
        ].copy()

        row: dict[str, float | str] = {"trigger_event_id": trigger["trigger_event_id"]}
        if future_events.empty:
            for horizon in horizons_hours:
                row[_target_col(horizon)] = np.nan
            rows.append(row)
            continue

        future_events["distance_km"] = haversine_km(
            trigger["trigger_latitude"],
            trigger["trigger_longitude"],
            future_events["latitude"].values,
            future_events["longitude"].values,
        )
        future_events = future_events.loc[future_events["distance_km"] <= max_radius_km].copy()

        if future_events.empty:
            for horizon in horizons_hours:
                row[_target_col(horizon)] = np.nan
            rows.append(row)
            continue

        future_events["hours_since_trigger"] = (
            future_events["time"] - trigger_time
        ).dt.total_seconds() / 3600.0

        for horizon in horizons_hours:
            within_horizon = future_events.loc[future_events["hours_since_trigger"] <= horizon]
            row[_target_col(horizon)] = (
                float(within_horizon["magnitude"].max()) if not within_horizon.empty else np.nan
            )

        rows.append(row)

        if verbose and i % 1000 == 0:
            print(
                f"[PROGRESS] Task 3 max-magnitude labels built for {i:,} / {len(triggers):,} triggers"
            )

    label_df = pd.DataFrame(rows)
    ordered_cols = ["trigger_event_id"] + [_target_col(h) for h in horizons_hours]
    return label_df.loc[:, ordered_cols].sort_values("trigger_event_id").reset_index(drop=True)


def build_or_load_max_aftershock_magnitude_labels(
    dataset_name: str,
    min_aftershock_magnitude: float = TASK3_MAXMAG_MIN_AFTERSHOCK_MAG,
    force_recompute: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    dataset_slug = _slugify_dataset_name(dataset_name)
    output_path = _label_path(dataset_slug, min_aftershock_magnitude)
    expected_cols = {"trigger_event_id", _target_col(24), _target_col(72)}

    if output_path.exists() and not force_recompute:
        cached_df = load_dataframe(output_path)
        if expected_cols.issubset(cached_df.columns):
            if verbose:
                print(f"[LOAD EXISTING] Task 3 max-magnitude labels -> {output_path}")
            return cached_df

    triggers_path, comcat_path = _required_interim_paths(dataset_slug)
    if verbose:
        print(f"[LOAD] Triggers -> {triggers_path}")
        print(f"[LOAD] ComCat clean -> {comcat_path}")

    triggers_df = load_dataframe(triggers_path)
    comcat_df = load_dataframe(comcat_path)
    labels_df = build_max_aftershock_magnitude_labels(
        triggers_df=triggers_df,
        events_df=comcat_df,
        min_aftershock_magnitude=min_aftershock_magnitude,
        verbose=verbose,
    )
    save_dataframe(labels_df, output_path)
    if verbose:
        print(f"[SAVE] Task 3 max-magnitude labels -> {output_path}")
    return labels_df
