from __future__ import annotations

import pandas as pd


def assign_data_split(
    dataset_df: pd.DataFrame,
    train_years: tuple[int, ...] = tuple(range(2015, 2024)),
    validation_years: tuple[int, ...] = (2024,),
    test_years: tuple[int, ...] = (2025,),
) -> pd.DataFrame:
    """
    Assign the school-project split:
    train = 2015-2023, validation = 2024, test = 2025.
    """
    df = dataset_df.copy()

    if "trigger_year" not in df.columns:
        df["trigger_year"] = pd.to_datetime(df["trigger_time"], utc=True, errors="coerce").dt.year

    df["split"] = "holdout"
    df.loc[df["trigger_year"].isin(train_years), "split"] = "train"
    df.loc[df["trigger_year"].isin(validation_years), "split"] = "val"
    df.loc[df["trigger_year"].isin(test_years), "split"] = "test"

    return df
