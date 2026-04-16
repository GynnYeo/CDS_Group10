from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import (
    evaluate_binary_probabilities,
    evaluate_capped_count_predictions,
    evaluate_count_predictions,
)
from src.evaluation.validation import (
    validate_binary_prediction_columns,
    validate_prediction_frame,
)
from src.models.input_layer import EXPECTED_SPLITS


COUNT_EVAL_CAPS_BY_HORIZON = {
    24: 300.0,
    72: 500.0,
}


def sort_metric_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Sort metrics by model, horizon, and project split order."""

    if metrics_df.empty:
        return metrics_df

    split_order = {split_name: index for index, split_name in enumerate(EXPECTED_SPLITS)}
    return (
        metrics_df.assign(_split_order=metrics_df["split"].map(split_order))
        .sort_values(by=["model_name", "horizon", "_split_order"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def compute_probability_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped probability metrics by model, split, and horizon."""

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]
    for (model_name, split_name, horizon), group_df in prediction_df.groupby(group_columns, sort=True):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_prob"],
            id_col="trigger_event_id",
            split_col="split",
        )
        validate_binary_prediction_columns(
            prediction_df=group_df,
            y_true_col="y_true",
            y_prob_col="y_prob",
        )
        metrics = evaluate_binary_probabilities(group_df["y_true"], group_df["y_prob"])
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                **metrics,
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))


def compute_count_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped raw count metrics by model, split, and horizon."""

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]
    for (model_name, split_name, horizon), group_df in prediction_df.groupby(group_columns, sort=True):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_pred"],
            id_col="trigger_event_id",
            split_col="split",
        )
        metrics = evaluate_count_predictions(group_df["y_true"], group_df["y_pred"])
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                **metrics,
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))


def compute_capped_count_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Compute horizon-specific capped count metrics by model, split, and horizon."""

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]

    for (model_name, split_name, horizon), group_df in prediction_df.groupby(group_columns, sort=True):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=[
                "trigger_event_id",
                "split",
                "horizon",
                "model_name",
                "y_true",
                "y_pred",
            ],
            id_col="trigger_event_id",
            split_col="split",
        )

        horizon_int = int(horizon)
        if horizon_int not in COUNT_EVAL_CAPS_BY_HORIZON:
            raise ValueError(
                f"No capped count evaluation cap configured for horizon {horizon_int}."
            )

        metrics = evaluate_capped_count_predictions(
            group_df["y_true"],
            group_df["y_pred"],
            cap=COUNT_EVAL_CAPS_BY_HORIZON[horizon_int],
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": horizon_int,
                **metrics,
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))


def compute_magnitude_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Compute conditional magnitude regression metrics by model, split, and horizon."""

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]

    for (model_name, split_name, horizon), group_df in prediction_df.groupby(group_columns, sort=True):
        available_df = group_df.loc[group_df["target_available"]].copy()

        validate_prediction_frame(
            prediction_df=available_df,
            required_columns=[
                "trigger_event_id",
                "split",
                "horizon",
                "model_name",
                "y_true",
                "y_pred",
            ],
            id_col="trigger_event_id",
            split_col="split",
        )

        # Reuse the simple regression MAE/RMSE helper here because the task is
        # still point-regression, not because magnitude is conceptually a count.
        metrics = evaluate_count_predictions(
            available_df["y_true"],
            available_df["y_pred"],
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                "n_total": int(len(group_df)),
                "n_available": int(len(available_df)),
                "target_available_rate": float(len(available_df) / len(group_df)),
                **metrics,
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))
