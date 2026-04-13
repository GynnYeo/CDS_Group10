from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from src.utils.geo import haversine_km
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import INTERIM_COMCAT_DIR, INTERIM_TRIGGERS_DIR, PROCESSED_DIR


TASK3_LABEL_DIR = PROCESSED_DIR / "task3"
TASK3_HORIZONS = (24, 72)
TASK3_RADIUS_KM = 50.0


def _slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("dataset_name must contain at least one letter or number.")
    return slug


def _threshold_slug(magnitude_threshold: float) -> str:
    return str(magnitude_threshold).replace(".", "_")


def _task3_label_path(dataset_slug: str, magnitude_threshold: float) -> Path:
    threshold = _threshold_slug(magnitude_threshold)
    return TASK3_LABEL_DIR / f"{dataset_slug}__large_aftershock_labels_m{threshold}.parquet"


def _task3_target_col(horizon: int) -> str:
    return f"y_large_{horizon}h"


def _task3_count_col(horizon: int) -> str:
    return f"n_large_aftershocks_{horizon}h"


def _required_interim_paths(dataset_slug: str) -> tuple[Path, Path]:
    triggers_path = INTERIM_TRIGGERS_DIR / f"{dataset_slug}__triggers.parquet"
    comcat_path = INTERIM_COMCAT_DIR / f"{dataset_slug}__comcat_clean.parquet"
    missing = [path for path in (triggers_path, comcat_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Task 3 label construction requires existing interim dataset artifacts. "
            f"Missing: {missing_text}"
        )
    return triggers_path, comcat_path


def build_large_aftershock_labels(
    triggers_df: pd.DataFrame,
    events_df: pd.DataFrame,
    magnitude_threshold: float,
    horizons_hours: tuple[int, ...] = TASK3_HORIZONS,
    max_radius_km: float = TASK3_RADIUS_KM,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build sidecar binary labels for larger-aftershock occurrence.

    A positive label means at least one future event occurs within the time
    horizon, within the shared 50 km radius, and at or above the requested
    aftershock magnitude threshold.
    """
    if magnitude_threshold <= 0:
        raise ValueError("magnitude_threshold must be positive.")

    events = events_df.copy()
    events["time"] = pd.to_datetime(events["time"], utc=True, errors="coerce")
    events = events.loc[events["time"].notna()].copy()

    triggers = triggers_df.copy()
    triggers["trigger_time"] = pd.to_datetime(triggers["trigger_time"], utc=True, errors="coerce")
    triggers = triggers.loc[triggers["trigger_time"].notna()].copy()

    max_horizon = max(horizons_hours)
    rows: list[dict[str, int | str]] = []

    for i, (_, trigger) in enumerate(triggers.iterrows(), start=1):
        trigger_time = trigger["trigger_time"]
        future_events = events.loc[
            (events["time"] > trigger_time)
            & (events["time"] <= trigger_time + pd.Timedelta(hours=max_horizon))
            & (events["magnitude"] >= magnitude_threshold)
        ].copy()

        row: dict[str, int | str] = {"trigger_event_id": trigger["trigger_event_id"]}
        if future_events.empty:
            for horizon in horizons_hours:
                row[_task3_target_col(horizon)] = 0
                row[_task3_count_col(horizon)] = 0
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
                row[_task3_target_col(horizon)] = 0
                row[_task3_count_col(horizon)] = 0
            rows.append(row)
            continue

        future_events["hours_since_trigger"] = (
            future_events["time"] - trigger_time
        ).dt.total_seconds() / 3600.0

        for horizon in horizons_hours:
            within_horizon = future_events.loc[future_events["hours_since_trigger"] <= horizon]
            row[_task3_target_col(horizon)] = int(not within_horizon.empty)
            row[_task3_count_col(horizon)] = int(len(within_horizon))

        rows.append(row)

        if verbose and i % 1000 == 0:
            print(f"[PROGRESS] Task 3 labels built for {i:,} / {len(triggers):,} triggers")

    label_df = pd.DataFrame(rows)
    ordered_cols = ["trigger_event_id"]
    ordered_cols.extend([_task3_target_col(h) for h in horizons_hours])
    ordered_cols.extend([_task3_count_col(h) for h in horizons_hours])
    return label_df.loc[:, ordered_cols].sort_values("trigger_event_id").reset_index(drop=True)


def build_or_load_large_aftershock_labels(
    dataset_name: str,
    magnitude_threshold: float = 4.0,
    force_recompute: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load cached task-3 labels or build them from saved interim dataset artifacts.
    """
    dataset_slug = _slugify_dataset_name(dataset_name)
    output_path = _task3_label_path(dataset_slug, magnitude_threshold)

    expected_cols = {"trigger_event_id"}
    expected_cols.update({_task3_target_col(h) for h in TASK3_HORIZONS})
    expected_cols.update({_task3_count_col(h) for h in TASK3_HORIZONS})

    if output_path.exists() and not force_recompute:
        cached_df = load_dataframe(output_path)
        if expected_cols.issubset(cached_df.columns):
            if verbose:
                print(f"[LOAD EXISTING] Task 3 labels -> {output_path}")
            return cached_df
        if verbose:
            print(f"[REBUILD] Task 3 label cache missing newer columns -> {output_path}")

    triggers_path, comcat_path = _required_interim_paths(dataset_slug)
    if verbose:
        print(f"[LOAD] Triggers -> {triggers_path}")
        print(f"[LOAD] ComCat clean -> {comcat_path}")

    triggers_df = load_dataframe(triggers_path)
    comcat_df = load_dataframe(comcat_path)
    labels_df = build_large_aftershock_labels(
        triggers_df=triggers_df,
        events_df=comcat_df,
        magnitude_threshold=magnitude_threshold,
        verbose=verbose,
    )
    save_dataframe(labels_df, output_path)
    if verbose:
        print(f"[SAVE] Task 3 labels -> {output_path}")

    return labels_df
