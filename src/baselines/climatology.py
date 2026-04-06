from __future__ import annotations

import pandas as pd


DEFAULT_TARGET_COLUMNS = (
    "y_24h",
    "y_72h",
    "n_aftershocks_24h",
    "n_aftershocks_72h",
)


def fit_climatology_baseline(
    train_df: pd.DataFrame,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> dict:
    """
    Fit a simple climatology baseline using training-set averages only.

    This is intentionally simple and presentation-friendly for a school project.
    """
    missing = [column for column in target_columns if column not in train_df.columns]
    if missing:
        raise ValueError(f"train_df is missing target columns: {missing}")

    return {
        "n_train": int(len(train_df)),
        "target_means": {
            column: float(train_df[column].mean())
            for column in target_columns
        },
    }


def predict_climatology_baseline(model: dict, scoring_df: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.DataFrame({"trigger_event_id": scoring_df["trigger_event_id"].values})

    for column, mean_value in model["target_means"].items():
        predictions[f"pred_{column}"] = mean_value

    return predictions


def fit_grouped_climatology_baseline(
    train_df: pd.DataFrame,
    magnitude_bin_column: str = "trigger_magnitude_bin",
) -> pd.DataFrame:
    """
    Optional slightly-stronger climatology baseline grouped by trigger magnitude.

    TODO:
    - Decide whether to keep this as a second baseline or stick to the global mean.
    """
    if magnitude_bin_column not in train_df.columns:
        raise ValueError(f"Missing column: {magnitude_bin_column}")

    grouped = (
        train_df.groupby(magnitude_bin_column, dropna=False)[
            ["y_24h", "y_72h", "n_aftershocks_24h", "n_aftershocks_72h"]
        ]
        .mean()
        .reset_index()
    )
    return grouped
