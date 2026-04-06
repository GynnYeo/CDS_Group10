from __future__ import annotations

import numpy as np
import pandas as pd


def _count_prior_events(
    event_times_ns: np.ndarray,
    trigger_time_ns: int,
    window_hours: int,
) -> int:
    window_ns = int(window_hours * 3600 * 1e9)
    left = np.searchsorted(event_times_ns, trigger_time_ns - window_ns, side="left")
    right = np.searchsorted(event_times_ns, trigger_time_ns, side="left")
    return int(right - left)


def build_trigger_features(
    triggers_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a simple, readable feature table from the trigger-event table.

    This stays intentionally modest for a school project and can be expanded
    later once the baseline datasets are stable.
    """
    features = triggers_df[["trigger_event_id", "trigger_time", "trigger_magnitude"]].copy()
    features["trigger_time"] = pd.to_datetime(features["trigger_time"], utc=True, errors="coerce")

    features["trigger_month"] = features["trigger_time"].dt.month
    features["trigger_year_feature"] = features["trigger_time"].dt.year
    features["trigger_dayofyear"] = features["trigger_time"].dt.dayofyear
    features["trigger_hour"] = features["trigger_time"].dt.hour
    features["trigger_magnitude_bin"] = pd.cut(
        features["trigger_magnitude"],
        bins=[5.0, 5.5, 6.0, 6.5, 7.0, float("inf")],
        right=False,
        include_lowest=True,
    ).astype(str)

    if events_df is not None and not events_df.empty:
        history = events_df.copy()
        history["time"] = pd.to_datetime(history["time"], utc=True, errors="coerce")
        history = history.loc[
            history["time"].notna()
            & history["magnitude"].notna()
            & (history["magnitude"] >= 2.5)
        ].sort_values("time")

        event_times_ns = history["time"].astype("int64").to_numpy()
        trigger_times_ns = features["trigger_time"].astype("int64").to_numpy()

        features["prior_global_event_count_24h"] = [
            _count_prior_events(event_times_ns, trigger_time_ns, window_hours=24)
            for trigger_time_ns in trigger_times_ns
        ]
        features["prior_global_event_count_7d"] = [
            _count_prior_events(event_times_ns, trigger_time_ns, window_hours=24 * 7)
            for trigger_time_ns in trigger_times_ns
        ]

    features = features.drop(columns=["trigger_time", "trigger_magnitude"])

    # TODO:
    # - Add regional features if you decide to bucket geographic zones.
    # - Add fault/mechanism-derived features when moment tensor coverage is usable.

    return features
