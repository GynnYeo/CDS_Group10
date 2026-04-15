from __future__ import annotations

import numpy as np
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
    """
    Label one trigger using the project aftershock definition.

    Aftershocks are future earthquakes that satisfy the shared magnitude,
    distance, and time-horizon rules. They are not required to share the same
    catalog event ID as the trigger.
    """
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
            row[f"max_aftershock_magnitude_{horizon}h"] = np.nan
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
        if within_horizon.empty:
            row[f"max_aftershock_magnitude_{horizon}h"] = np.nan
        else:
            row[f"max_aftershock_magnitude_{horizon}h"] = float(
                within_horizon["magnitude"].max()
            )

    return row


def _validate_aftershock_label_consistency(
    labels_df: pd.DataFrame,
    horizons_hours: tuple[int, ...],
) -> None:
    """Validate horizon-based count/max-magnitude consistency."""
    for horizon in horizons_hours:
        count_col = f"n_aftershocks_{horizon}h"
        max_mag_col = f"max_aftershock_magnitude_{horizon}h"

        zero_count_with_value = labels_df.loc[
            (labels_df[count_col] == 0) & labels_df[max_mag_col].notna()
        ]
        if not zero_count_with_value.empty:
            raise ValueError(
                f"Rows with zero aftershocks must have NaN in '{max_mag_col}'."
            )

        positive_count_missing_value = labels_df.loc[
            (labels_df[count_col] > 0) & labels_df[max_mag_col].isna()
        ]
        if not positive_count_missing_value.empty:
            raise ValueError(
                f"Rows with positive aftershock counts must have a value in '{max_mag_col}'."
            )

    sorted_horizons = sorted(horizons_hours)
    for earlier_horizon, later_horizon in zip(sorted_horizons, sorted_horizons[1:]):
        earlier_col = f"max_aftershock_magnitude_{earlier_horizon}h"
        later_col = f"max_aftershock_magnitude_{later_horizon}h"
        monotonic_violation = labels_df.loc[
            labels_df[earlier_col].notna()
            & labels_df[later_col].notna()
            & (labels_df[later_col] < labels_df[earlier_col])
        ]
        if not monotonic_violation.empty:
            raise ValueError(
                "Later-horizon max aftershock magnitudes must be greater than or "
                f"equal to earlier horizons: '{later_col}' < '{earlier_col}'."
            )


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
    _validate_aftershock_label_consistency(
        labels_df=labels_df,
        horizons_hours=horizons_hours,
    )
    ordered_columns = ["trigger_event_id"]
    ordered_columns.extend([f"y_{horizon}h" for horizon in horizons_hours])
    ordered_columns.extend([f"n_aftershocks_{horizon}h" for horizon in horizons_hours])
    ordered_columns.extend(
        [f"max_aftershock_magnitude_{horizon}h" for horizon in horizons_hours]
    )
    labels_df = labels_df[ordered_columns]
    return labels_df.sort_values("trigger_event_id").reset_index(drop=True)
