from __future__ import annotations

import pandas as pd


def assign_data_split(
    dataset_df: pd.DataFrame,
    train_start_year: int = 2010,
    train_end_year: int = 2022,
    validation_years: tuple[int, ...] = (2023,),
    test_years: tuple[int, ...] = (2024, 2025),
) -> pd.DataFrame:
    """
    Assign the project split:
    train = 2010-2022, validation = 2023, test = 2024-2025.
    """
    df = dataset_df.copy()

    if "trigger_year" not in df.columns:
        df["trigger_year"] = pd.to_datetime(
            df["trigger_time"], utc=True, errors="coerce"
        ).dt.year

    train_years = tuple(range(train_start_year, train_end_year + 1))

    df["split"] = "holdout"
    df.loc[df["trigger_year"].isin(train_years), "split"] = "train"
    df.loc[df["trigger_year"].isin(validation_years), "split"] = "val"
    df.loc[df["trigger_year"].isin(test_years), "split"] = "test"

    return df