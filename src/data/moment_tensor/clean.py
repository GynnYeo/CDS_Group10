from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import INTERIM_MOMENT_TENSOR_DIR


STANDARD_COLUMN_ALIASES = {
    "event_name": "gcmt_event_name",
    "time": "gcmt_time",
    "latitude": "gcmt_latitude",
    "longitude": "gcmt_longitude",
    "depth_km": "gcmt_depth_km",
    "magnitude": "gcmt_magnitude",
}

NUMERIC_COLUMNS = [
    "gcmt_reference_latitude",
    "gcmt_reference_longitude",
    "gcmt_reference_depth_km",
    "gcmt_mb",
    "gcmt_ms",
    "gcmt_centroid_time_shift_sec",
    "gcmt_centroid_time_shift_error_sec",
    "gcmt_latitude",
    "gcmt_latitude_error",
    "gcmt_longitude",
    "gcmt_longitude_error",
    "gcmt_depth_km",
    "gcmt_depth_error_km",
    "gcmt_magnitude",
    "gcmt_half_duration_sec",
    "gcmt_moment_exponent",
    "gcmt_mrr",
    "gcmt_mrr_error",
    "gcmt_mtt",
    "gcmt_mtt_error",
    "gcmt_mpp",
    "gcmt_mpp_error",
    "gcmt_mrt",
    "gcmt_mrt_error",
    "gcmt_mrp",
    "gcmt_mrp_error",
    "gcmt_mtp",
    "gcmt_mtp_error",
    "gcmt_eig1",
    "gcmt_eig1_plunge",
    "gcmt_eig1_azimuth",
    "gcmt_eig2",
    "gcmt_eig2_plunge",
    "gcmt_eig2_azimuth",
    "gcmt_eig3",
    "gcmt_eig3_plunge",
    "gcmt_eig3_azimuth",
    "gcmt_scalar_moment",
    "strike",
    "dip",
    "rake",
    "strike2",
    "dip2",
    "rake2",
]


def _log_step(log_rows: list[dict], step: str, rows_before: int, rows_after: int) -> None:
    log_rows.append(
        {
            "step": step,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
        }
    )


def clean_moment_tensor(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean optional Global CMT enrichment data.

    Missing moment tensor coverage is expected in v2, so this cleaner should
    preserve partial coverage rather than aggressively filtering rows.
    """
    cleaned = df.copy().rename(columns=STANDARD_COLUMN_ALIASES)
    log_rows: list[dict] = []

    rows_before = len(cleaned)
    if "gcmt_event_name" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset="gcmt_event_name", keep="first")
    _log_step(log_rows, "Drop duplicate gcmt_event_name", rows_before, len(cleaned))

    rows_before = len(cleaned)
    for column in ("gcmt_time", "gcmt_centroid_time"):
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], utc=True, errors="coerce")
    _log_step(log_rows, "Coerce datetime columns", rows_before, len(cleaned))

    rows_before = len(cleaned)
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    _log_step(log_rows, "Coerce numeric columns", rows_before, len(cleaned))

    rows_before = len(cleaned)
    for column in ("gcmt_reference_catalog", "gcmt_location_name", "gcmt_depth_type", "gcmt_source_file"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string")
    _log_step(log_rows, "Normalize text columns", rows_before, len(cleaned))

    rows_before = len(cleaned)
    required_for_matching = ["gcmt_time", "gcmt_latitude", "gcmt_longitude", "gcmt_magnitude"]
    existing_required = [column for column in required_for_matching if column in cleaned.columns]
    if existing_required:
        cleaned = cleaned.dropna(subset=existing_required).copy()
    _log_step(log_rows, "Drop rows missing core fuzzy-match fields", rows_before, len(cleaned))

    rows_before = len(cleaned)
    if "gcmt_latitude" in cleaned.columns and "gcmt_longitude" in cleaned.columns:
        cleaned = cleaned.loc[
            cleaned["gcmt_latitude"].between(-90, 90)
            & cleaned["gcmt_longitude"].between(-180, 180)
        ].copy()
    _log_step(log_rows, "Keep valid latitude/longitude ranges", rows_before, len(cleaned))

    rows_before = len(cleaned)
    if "gcmt_magnitude" in cleaned.columns:
        cleaned = cleaned.loc[cleaned["gcmt_magnitude"] > 0].copy()
    _log_step(log_rows, "Keep positive GCMT magnitudes", rows_before, len(cleaned))

    cleaned = cleaned.sort_values("gcmt_time").reset_index(drop=True)
    return cleaned, pd.DataFrame(log_rows)


def save_clean_moment_tensor(
    df: pd.DataFrame,
    output_path: str | Path = INTERIM_MOMENT_TENSOR_DIR / "moment_tensor_clean.parquet",
    log_output_path: str | Path = INTERIM_MOMENT_TENSOR_DIR / "moment_tensor_cleaning_log.csv",
    force_recompute: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean optional GCMT data and save the cleaned table plus cleaning log.

    The cache behavior mirrors the ComCat cleaner so the shared dataset runner
    can reuse prior outputs when `force_recompute` is disabled.
    """
    output_path = Path(output_path)
    log_output_path = Path(log_output_path)

    if output_path.exists() and not force_recompute:
        if verbose:
            print(f"[LOAD EXISTING] {output_path}")

        cleaned = load_dataframe(output_path)
        log_df = load_dataframe(log_output_path) if log_output_path.exists() else pd.DataFrame()
        return cleaned, log_df

    cleaned, log_df = clean_moment_tensor(df)
    save_dataframe(cleaned, output_path)
    save_dataframe(log_df, log_output_path)

    if verbose:
        print(f"[SAVE] {output_path}")
        print(f"[SAVE] {log_output_path}")

    return cleaned, log_df
