from __future__ import annotations

import pandas as pd

from src.utils.geo import haversine_km


AFTERSHOCK_MIN_MAGNITUDE = 2.5
AFTERSHOCK_MAX_RADIUS_KM = 50.0
LABEL_HORIZONS_HOURS = (24, 72)


def _label_single_trigger(
    trigger: pd.Series,
    events_df: pd.DataFrame,
    aftershock_min_magnitude: float,
    aftershock_max_radius_km: float,
    horizons_hours: tuple[int, ...],
) -> dict:
    trigger_time = trigger["trigger_time"]
    max_horizon_hours = max(horizons_hours)
    candidate_events = events_df.loc[
        (events_df["time"] > trigger_time)
        & (events_df["time"] <= trigger_time + pd.Timedelta(hours=max_horizon_hours))
        & (events_df["magnitude"] >= aftershock_min_magnitude)
    ].copy()

    if candidate_events.empty:
        row = {"trigger_event_id": trigger["trigger_event_id"]}
        for horizon in horizons_hours:
            row[f"y_{horizon}h"] = 0
            row[f"n_aftershocks_{horizon}h"] = 0
        return row

    candidate_events["distance_km"] = haversine_km(
        trigger["trigger_latitude"],
        trigger["trigger_longitude"],
        candidate_events["latitude"].values,
        candidate_events["longitude"].values,
    )

    candidate_events["hours_since_trigger"] = (
        candidate_events["time"] - trigger_time
    ).dt.total_seconds() / 3600.0

    nearby_events = candidate_events.loc[
        candidate_events["distance_km"] <= aftershock_max_radius_km
    ].copy()

    row = {"trigger_event_id": trigger["trigger_event_id"]}

    for horizon in horizons_hours:
        within_horizon = nearby_events.loc[nearby_events["hours_since_trigger"] <= horizon]
        row[f"y_{horizon}h"] = int(len(within_horizon) > 0)
        row[f"n_aftershocks_{horizon}h"] = int(len(within_horizon))

    return row


def build_aftershock_labels(
    triggers_df: pd.DataFrame,
    events_df: pd.DataFrame,
    aftershock_min_magnitude: float = AFTERSHOCK_MIN_MAGNITUDE,
    aftershock_max_radius_km: float = AFTERSHOCK_MAX_RADIUS_KM,
    horizons_hours: tuple[int, ...] = LABEL_HORIZONS_HOURS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build binary and count labels for each trigger event.

    Labels in v2 are horizon-based and intentionally replace the old
    time-to-next-aftershock target logic.
    """
    rows: list[dict] = []

    events = events_df.copy()
    events["time"] = pd.to_datetime(events["time"], utc=True, errors="coerce")

    triggers = triggers_df.copy()
    triggers["trigger_time"] = pd.to_datetime(triggers["trigger_time"], utc=True, errors="coerce")

    for i, (_, trigger) in enumerate(triggers.iterrows(), start=1):
        rows.append(
            _label_single_trigger(
                trigger=trigger,
                events_df=events,
                aftershock_min_magnitude=aftershock_min_magnitude,
                aftershock_max_radius_km=aftershock_max_radius_km,
                horizons_hours=horizons_hours,
            )
        )

        if verbose and i % 1000 == 0:
            print(f"[PROGRESS] Labeled {i:,} / {len(triggers):,} trigger events")

    labels_df = pd.DataFrame(rows)
    ordered_columns = ["trigger_event_id"]
    ordered_columns.extend([f"y_{horizon}h" for horizon in horizons_hours])
    ordered_columns.extend([f"n_aftershocks_{horizon}h" for horizon in horizons_hours])
    labels_df = labels_df[ordered_columns]
    return labels_df.sort_values("trigger_event_id").reset_index(drop=True)
