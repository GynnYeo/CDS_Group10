from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.geo import haversine_km


STRICT_TIME_DIFF_SEC = 30.0
STRICT_DISTANCE_KM = 30.0
STRICT_MAG_DIFF = 0.3

RELAXED_TIME_DIFF_SEC = 120.0
RELAXED_DISTANCE_KM = 100.0
RELAXED_MAG_DIFF = 0.5

MIN_GCMT_ENRICHMENT_COLUMNS = [
    "gcmt_event_name",
    "gcmt_time",
    "gcmt_latitude",
    "gcmt_longitude",
    "gcmt_depth_km",
    "gcmt_magnitude",
    "strike",
    "dip",
    "rake",
]


def _add_gcmt_diagnostics(base: pd.DataFrame) -> pd.DataFrame:
    enriched = base.copy()
    enriched["has_gcmt"] = False
    enriched["gcmt_match_rule"] = pd.Series(pd.NA, index=enriched.index, dtype="string")
    enriched["gcmt_time_diff_sec"] = np.nan
    enriched["gcmt_distance_km"] = np.nan
    enriched["gcmt_mag_diff"] = np.nan
    return enriched



def _ensure_gcmt_columns(enriched: pd.DataFrame, gcmt_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ensure expected GCMT fields exist even when there are no matches."""
    result = enriched.copy()

    gcmt_columns = list(MIN_GCMT_ENRICHMENT_COLUMNS)
    if gcmt_df is not None:
        for column in gcmt_df.columns:
            if column not in gcmt_columns:
                gcmt_columns.append(column)

    for column in gcmt_columns:
        if column in result.columns:
            continue

        if gcmt_df is not None and column in gcmt_df.columns:
            if pd.api.types.is_datetime64_any_dtype(gcmt_df[column]):
                result[column] = pd.Series(pd.NaT, index=result.index, dtype=gcmt_df[column].dtype)
            elif pd.api.types.is_numeric_dtype(gcmt_df[column]):
                result[column] = np.nan
            else:
                result[column] = pd.NA
        else:
            result[column] = pd.NA

    return result



def _choose_unique_best(
    candidates: pd.DataFrame,
    time_col: str,
    distance_col: str,
    mag_col: str,
) -> pd.Series | None:
    if candidates.empty:
        return None

    ranking_columns = ["match_rule_rank", time_col, distance_col, mag_col]
    sorted_candidates = candidates.sort_values(ranking_columns).reset_index(drop=True)
    top = sorted_candidates.iloc[0]

    if len(sorted_candidates) == 1:
        return top

    second = sorted_candidates.iloc[1]
    tied = (
        top["match_rule_rank"] == second["match_rule_rank"]
        and np.isclose(float(top[time_col]), float(second[time_col]))
        and np.isclose(float(top[distance_col]), float(second[distance_col]))
        and np.isclose(float(top[mag_col]), float(second[mag_col]))
    )
    if tied:
        return None

    return top



def _build_match_proposals(comcat_df: pd.DataFrame, gcmt_df: pd.DataFrame) -> pd.DataFrame:
    searchable = comcat_df.dropna(subset=["time", "latitude", "longitude", "magnitude"]).copy()
    searchable = searchable.sort_values("time").reset_index().rename(columns={"index": "comcat_index"})

    if searchable.empty or gcmt_df.empty:
        return pd.DataFrame()

    comcat_time_ns = searchable["time"].astype("int64").to_numpy()
    gcmt_iter = gcmt_df.dropna(
        subset=["gcmt_time", "gcmt_latitude", "gcmt_longitude", "gcmt_magnitude"]
    ).reset_index().rename(columns={"index": "gcmt_index"})

    proposals: list[dict] = []

    for _, gcmt_row in gcmt_iter.iterrows():
        gcmt_time_ns = gcmt_row["gcmt_time"].value
        lower_ns = gcmt_time_ns - int(RELAXED_TIME_DIFF_SEC * 1e9)
        upper_ns = gcmt_time_ns + int(RELAXED_TIME_DIFF_SEC * 1e9)

        start = np.searchsorted(comcat_time_ns, lower_ns, side="left")
        stop = np.searchsorted(comcat_time_ns, upper_ns, side="right")
        if start >= stop:
            continue

        window = searchable.iloc[start:stop].copy()
        window["time_diff_sec"] = (
            window["time"] - gcmt_row["gcmt_time"]
        ).abs().dt.total_seconds()
        window["distance_km"] = haversine_km(
            gcmt_row["gcmt_latitude"],
            gcmt_row["gcmt_longitude"],
            window["latitude"].values,
            window["longitude"].values,
        )
        window["mag_diff"] = (window["magnitude"] - gcmt_row["gcmt_magnitude"]).abs()

        strict_candidates = window.loc[
            (window["time_diff_sec"] <= STRICT_TIME_DIFF_SEC)
            & (window["distance_km"] <= STRICT_DISTANCE_KM)
            & (window["mag_diff"] <= STRICT_MAG_DIFF)
        ].copy()
        strict_candidates["match_rule"] = "strict"
        strict_candidates["match_rule_rank"] = 0

        relaxed_candidates = window.loc[
            (window["time_diff_sec"] <= RELAXED_TIME_DIFF_SEC)
            & (window["distance_km"] <= RELAXED_DISTANCE_KM)
            & (window["mag_diff"] <= RELAXED_MAG_DIFF)
        ].copy()
        relaxed_candidates["match_rule"] = "relaxed"
        relaxed_candidates["match_rule_rank"] = 1

        active_candidates = strict_candidates if not strict_candidates.empty else relaxed_candidates
        best = _choose_unique_best(
            active_candidates,
            time_col="time_diff_sec",
            distance_col="distance_km",
            mag_col="mag_diff",
        )
        if best is None:
            continue

        proposals.append(
            {
                "comcat_index": int(best["comcat_index"]),
                "gcmt_index": int(gcmt_row["gcmt_index"]),
                "gcmt_match_rule": best["match_rule"],
                "match_rule_rank": int(best["match_rule_rank"]),
                "gcmt_time_diff_sec": float(best["time_diff_sec"]),
                "gcmt_distance_km": float(best["distance_km"]),
                "gcmt_mag_diff": float(best["mag_diff"]),
            }
        )

    return pd.DataFrame(proposals)



def merge_comcat_with_moment_tensor(
    comcat_df: pd.DataFrame,
    moment_tensor_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fuzzy-match optional Global CMT records onto the ComCat master table.

    Matching deliberately does not rely on shared event IDs because ComCat and
    GCMT are separate catalogs.
    """
    base = comcat_df.copy().reset_index(drop=True)
    if "time" in base.columns:
        base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce")

    enriched = _add_gcmt_diagnostics(base)
    enriched = _ensure_gcmt_columns(enriched, moment_tensor_df)

    if moment_tensor_df is None or moment_tensor_df.empty:
        return enriched

    gcmt = moment_tensor_df.copy().reset_index(drop=True)
    gcmt["gcmt_time"] = pd.to_datetime(gcmt["gcmt_time"], utc=True, errors="coerce")

    proposals = _build_match_proposals(enriched, gcmt)
    if proposals.empty:
        return enriched

    resolved_rows: list[dict] = []
    for _, group in proposals.groupby("comcat_index", sort=False):
        best = _choose_unique_best(
            group,
            time_col="gcmt_time_diff_sec",
            distance_col="gcmt_distance_km",
            mag_col="gcmt_mag_diff",
        )
        if best is not None:
            resolved_rows.append(best.to_dict())

    if not resolved_rows:
        return enriched

    final_matches = pd.DataFrame(resolved_rows)
    gcmt_columns = list(gcmt.columns)

    for _, match in final_matches.iterrows():
        comcat_index = int(match["comcat_index"])
        gcmt_index = int(match["gcmt_index"])

        enriched.at[comcat_index, "has_gcmt"] = True
        enriched.at[comcat_index, "gcmt_match_rule"] = match["gcmt_match_rule"]
        enriched.at[comcat_index, "gcmt_time_diff_sec"] = match["gcmt_time_diff_sec"]
        enriched.at[comcat_index, "gcmt_distance_km"] = match["gcmt_distance_km"]
        enriched.at[comcat_index, "gcmt_mag_diff"] = match["gcmt_mag_diff"]

        gcmt_values = gcmt.loc[gcmt_index, gcmt_columns]
        for column in gcmt_columns:
            enriched.at[comcat_index, column] = gcmt_values[column]

    return enriched
