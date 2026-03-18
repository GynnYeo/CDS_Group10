from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.paths import COMCAT_INTERIM_DIR, COMCAT_PROCESSED_DIR


BASELINE_FEATURE_COLUMNS = [
    "mainshock_event_id",
    "mainshock_time",
    "mainshock_latitude",
    "mainshock_longitude",
    "mainshock_depth_km",
    "mainshock_magnitude",
    "mainshock_magnitude_type",
    "mainshock_status",
    "mainshock_gap",
    "mainshock_dmin",
    "mainshock_rms",
    "mainshock_nst",
    "mainshock_year",
    "mainshock_month",
    "mainshock_dayofyear",
    "mainshock_hour",
]

TARGET_COLUMNS = [
    "target_hours_raw",
    "target_hours_capped",
    "has_aftershock_within_720h",
    "is_capped_720h",
]

OPTIONAL_METADATA_COLUMNS = [
    "next_aftershock_event_id",
    "next_aftershock_time",
    "next_aftershock_magnitude",
    "next_aftershock_distance_km",
]


def build_baseline_dataset(
    target_df: pd.DataFrame,
    keep_aftershock_metadata: bool = False,
) -> pd.DataFrame:
    """
    Convert the mainshock target dataset into a model-ready baseline dataset.

    This keeps only:
    - mainshock-side predictor columns
    - simple time-derived features
    - target columns
    - optional aftershock metadata for reference/debugging
    """
    df = target_df.copy()

    if "mainshock_time" not in df.columns:
        raise ValueError("Input dataframe must contain 'mainshock_time'.")

    # Ensure datetime
    df["mainshock_time"] = pd.to_datetime(df["mainshock_time"], utc=True, errors="coerce")

    # Add simple time-derived features
    df["mainshock_year"] = df["mainshock_time"].dt.year
    df["mainshock_month"] = df["mainshock_time"].dt.month
    df["mainshock_dayofyear"] = df["mainshock_time"].dt.dayofyear
    df["mainshock_hour"] = df["mainshock_time"].dt.hour

    keep_cols = BASELINE_FEATURE_COLUMNS + TARGET_COLUMNS

    if keep_aftershock_metadata:
        keep_cols += [c for c in OPTIONAL_METADATA_COLUMNS if c in df.columns]

    missing_required = [c for c in keep_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Input dataframe is missing required columns: {missing_required}"
        )

    baseline_df = df[keep_cols].copy()

    # Sort for reproducibility
    baseline_df = baseline_df.sort_values("mainshock_time").reset_index(drop=True)

    return baseline_df


def load_and_build_baseline_dataset(
    input_path: str | Path,
    keep_aftershock_metadata: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load mainshock target dataset from parquet and convert it into
    a baseline model dataset.
    """
    input_path = Path(input_path)

    if verbose:
        print(f"[LOAD] {input_path}")

    target_df = pd.read_parquet(input_path)

    baseline_df = build_baseline_dataset(
        target_df=target_df,
        keep_aftershock_metadata=keep_aftershock_metadata,
    )

    if verbose:
        print(f"[DONE] Baseline dataset rows: {len(baseline_df):,}")
        print(f"[DONE] Baseline dataset columns: {len(baseline_df.columns)}")

    return baseline_df


def save_baseline_dataset(
    input_path: str | Path = COMCAT_INTERIM_DIR / "mainshock_target_dataset.parquet",
    output_path: str | Path = COMCAT_PROCESSED_DIR / "baseline_mainshock_model_dataset.parquet",
    keep_aftershock_metadata: bool = False,
    force_recompute: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the target dataset, build the baseline model dataset, and save it.
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_recompute:
        if verbose:
            print(f"[LOAD EXISTING] {output_path}")
        return pd.read_parquet(output_path)

    baseline_df = load_and_build_baseline_dataset(
        input_path=input_path,
        keep_aftershock_metadata=keep_aftershock_metadata,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_parquet(output_path, index=False)

    if verbose:
        print(f"[SAVE] Baseline dataset -> {output_path}")

    return baseline_df


if __name__ == "__main__":
    baseline_df = save_baseline_dataset(
        input_path=COMCAT_INTERIM_DIR / "mainshock_target_dataset_2025.parquet",
        output_path=COMCAT_PROCESSED_DIR / "baseline_mainshock_model_dataset_2025.parquet",
        keep_aftershock_metadata=False,
        force_recompute=False,
        verbose=True,
    )

    print(baseline_df.head())
    print(baseline_df.shape)