from __future__ import annotations

import pandas as pd


def assemble_modeling_dataset(
    triggers_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the final one-row-per-trigger modeling dataset.
    """
    dataset = triggers_df.merge(labels_df, how="left", on="trigger_event_id")
    dataset = dataset.merge(features_df, how="left", on="trigger_event_id", suffixes=("", "_feature"))

    if dataset["trigger_event_id"].duplicated().any():
        raise ValueError("Final dataset must contain exactly one row per trigger_event_id.")

    return dataset.sort_values("trigger_time").reset_index(drop=True)
