from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.paths import COMCAT_RAW_DIR, COMCAT_INTERIM_DIR


def _log_step(
    cleaning_log: list[dict],
    step_name: str,
    rows_before: int,
    rows_after: int,
) -> None:
    cleaning_log.append(
        {
            "step": step_name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
        }
    )


def clean_comcat_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean a raw ComCat dataframe and return:
    1. cleaned dataframe
    2. cleaning log dataframe
    """
    comcat_df_clean = df.copy()
    cleaning_log: list[dict] = []

    # 0. standardise categorical text fields
    rows_before = len(comcat_df_clean)

    if "status" in comcat_df_clean.columns:
        comcat_df_clean["status"] = comcat_df_clean["status"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    if "event_type" in comcat_df_clean.columns:
        comcat_df_clean["event_type"] = comcat_df_clean["event_type"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    if "magnitude_type" in comcat_df_clean.columns:
        comcat_df_clean["magnitude_type"] = comcat_df_clean["magnitude_type"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    _log_step(
        cleaning_log,
        "Standardise categorical text fields",
        rows_before,
        len(comcat_df_clean),
    )

    # 1. drop exact duplicate event IDs
    rows_before = len(comcat_df_clean)
    comcat_df_clean = comcat_df_clean.drop_duplicates(subset="event_id", keep="first")
    _log_step(
        cleaning_log,
        "Drop duplicate event_id",
        rows_before,
        len(comcat_df_clean),
    )

    # 2. keep only earthquake events
    rows_before = len(comcat_df_clean)
    comcat_df_clean = comcat_df_clean.loc[
        comcat_df_clean["event_type"] == "earthquake"
    ].copy()
    _log_step(
        cleaning_log,
        "Keep only event_type == 'earthquake'",
        rows_before,
        len(comcat_df_clean),
    )

    # 3. drop rows missing core fields needed for target construction
    core_cols = ["event_id", "time", "latitude", "longitude", "depth_km", "magnitude"]
    rows_before = len(comcat_df_clean)
    comcat_df_clean = comcat_df_clean.dropna(subset=core_cols).copy()
    _log_step(
        cleaning_log,
        "Drop rows missing core fields",
        rows_before,
        len(comcat_df_clean),
    )

    # 4. valid coordinate ranges
    rows_before = len(comcat_df_clean)
    comcat_df_clean = comcat_df_clean.loc[
        comcat_df_clean["latitude"].between(-90, 90)
        & comcat_df_clean["longitude"].between(-180, 180)
    ].copy()
    _log_step(
        cleaning_log,
        "Keep valid latitude/longitude ranges",
        rows_before,
        len(comcat_df_clean),
    )

    # 5. basic target-related magnitude filter
    rows_before = len(comcat_df_clean)
    comcat_df_clean = comcat_df_clean.loc[
        comcat_df_clean["magnitude"] >= 2.5
    ].copy()
    _log_step(
        cleaning_log,
        "Keep magnitude >= 2.5",
        rows_before,
        len(comcat_df_clean),
    )

    # 6. sort by time
    comcat_df_clean = comcat_df_clean.sort_values("time").reset_index(drop=True)

    cleaning_log_df = pd.DataFrame(cleaning_log)
    return comcat_df_clean, cleaning_log_df


def load_and_clean_comcat(
    input_path: str | Path,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a raw ComCat parquet file and clean it.
    """
    input_path = Path(input_path)

    if verbose:
        print(f"[LOAD] {input_path}")

    df_raw = pd.read_parquet(input_path)
    df_clean, cleaning_log_df = clean_comcat_events(df_raw)

    if verbose:
        print(f"[DONE] Cleaned rows: {len(df_clean):,}")

    return df_clean, cleaning_log_df


def save_clean_comcat(
    input_path: str | Path,
    output_path: str | Path = COMCAT_INTERIM_DIR / "comcat_clean_events.parquet",
    log_output_path: str | Path | None = COMCAT_INTERIM_DIR / "comcat_cleaning_log.csv",
    force_recompute: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw ComCat data, clean it, and save cleaned output + cleaning log.
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_recompute:
        if verbose:
            print(f"[LOAD EXISTING] {output_path}")

        df_clean = pd.read_parquet(output_path)

        if log_output_path is not None and Path(log_output_path).exists():
            cleaning_log_df = pd.read_csv(log_output_path)
        else:
            cleaning_log_df = pd.DataFrame()

        return df_clean, cleaning_log_df

    df_clean, cleaning_log_df = load_and_clean_comcat(
        input_path=input_path,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)

    if log_output_path is not None:
        log_output_path = Path(log_output_path)
        log_output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaning_log_df.to_csv(log_output_path, index=False)

    if verbose:
        print(f"[SAVE] Cleaned data -> {output_path}")
        if log_output_path is not None:
            print(f"[SAVE] Cleaning log -> {log_output_path}")

    return df_clean, cleaning_log_df


if __name__ == "__main__":
    df_clean, cleaning_log_df = save_clean_comcat(
        input_path=COMCAT_RAW_DIR / "comcat_2025.parquet",
        output_path=COMCAT_INTERIM_DIR / "comcat_clean_events_2025.parquet",
        log_output_path=COMCAT_INTERIM_DIR / "comcat_cleaning_log_2025.csv",
        force_recompute=False,
        verbose=True,
    )

    print(df_clean.head())
    print(cleaning_log_df)