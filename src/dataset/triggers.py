from __future__ import annotations

import pandas as pd


TRIGGER_MIN_MAGNITUDE = 5.0


def build_trigger_events(
    events_df: pd.DataFrame,
    trigger_min_magnitude: float = TRIGGER_MIN_MAGNITUDE,
) -> pd.DataFrame:
    """
    Build the trigger-event table used as the unit of prediction in v2.

    Final modeling datasets should contain exactly one row per trigger event.
    """
    required_columns = ["event_id", "time", "latitude", "longitude", "depth_km", "magnitude"]
    missing = [column for column in required_columns if column not in events_df.columns]
    if missing:
        raise ValueError(f"events_df is missing required columns: {missing}")

    triggers = events_df.loc[events_df["magnitude"] >= trigger_min_magnitude].copy()
    triggers = triggers.sort_values("time").reset_index(drop=True)

    triggers = triggers.rename(
        columns={
            "event_id": "trigger_event_id",
            "time": "trigger_time",
            "latitude": "trigger_latitude",
            "longitude": "trigger_longitude",
            "depth_km": "trigger_depth_km",
            "magnitude": "trigger_magnitude",
        }
    )

    triggers["trigger_year"] = pd.to_datetime(triggers["trigger_time"], utc=True).dt.year

    return triggers
