from __future__ import annotations

import pandas as pd


def validate_prediction_frame(
    prediction_df: pd.DataFrame,
    required_columns: list[str],
    id_col: str | None = None,
    split_col: str | None = None,
    expected_splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Validate basic prediction-table structure before evaluation."""
    missing_columns = [column for column in required_columns if column not in prediction_df.columns]
    if missing_columns:
        raise ValueError(f"Prediction frame is missing required columns: {missing_columns}")

    for column in required_columns:
        if prediction_df[column].isna().any():
            raise ValueError(f"Prediction frame contains missing values in '{column}'.")

    if id_col is not None:
        if id_col not in prediction_df.columns:
            raise ValueError(f"Prediction frame is missing id column: {id_col}")
        if prediction_df[id_col].duplicated().any():
            raise ValueError(f"Prediction frame contains duplicate ids in '{id_col}'.")

    if split_col is not None and split_col in prediction_df.columns:
        observed = set(prediction_df[split_col].dropna().astype(str).unique())
        invalid = sorted(observed.difference(expected_splits))
        if invalid:
            raise ValueError(
                f"Prediction frame contains unexpected split values in '{split_col}': {invalid}"
            )


def validate_binary_prediction_columns(
    prediction_df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
) -> None:
    """Validate binary labels and probability predictions."""
    validate_prediction_frame(
        prediction_df=prediction_df,
        required_columns=[y_true_col, y_prob_col],
    )

    invalid_labels = prediction_df.loc[~prediction_df[y_true_col].isin([0, 1]), y_true_col]
    if not invalid_labels.empty:
        raise ValueError(
            f"Binary target column '{y_true_col}' must contain only 0/1 values."
        )

    out_of_range = prediction_df.loc[
        (prediction_df[y_prob_col] < 0.0) | (prediction_df[y_prob_col] > 1.0),
        y_prob_col,
    ]
    if not out_of_range.empty:
        raise ValueError(
            f"Probability column '{y_prob_col}' must stay within [0, 1]."
        )
