from __future__ import annotations

import numpy as np
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

# Shared diagnostic buckets for both 24h and 72h count predictions.
# These buckets are based on the true raw count, not the prediction.
COUNT_BUCKETS = [
    ("0", 0.0, 0.0),
    ("1", 1.0, 1.0),
    ("2-5", 2.0, 5.0),
    ("6-10", 6.0, 10.0),
    ("11-20", 11.0, 20.0),
    ("21-50", 21.0, 50.0),
    ("51-100", 51.0, 100.0),
    ("101-300", 101.0, 300.0),
    ("300+", 301.0, np.inf),
]

# Buckets based on predicted raw count, not true count.
# These are intentionally smaller/lower than the true-count buckets because
# the current count head is conservative and rarely predicts very large values.
PREDICTED_COUNT_BUCKETS = [
    ("0-1", 0.0, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-50", 20.0, 50.0),
    ("50+", 50.0, np.inf),
]


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


def sort_diagnostic_rows(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    """Sort diagnostics by model, horizon, split, and count-bucket order."""

    if diagnostics_df.empty:
        return diagnostics_df

    split_order = {split_name: index for index, split_name in enumerate(EXPECTED_SPLITS)}
    return (
        diagnostics_df.assign(_split_order=diagnostics_df["split"].map(split_order))
        .sort_values(by=["model_name", "horizon", "_split_order", "bucket_order"])
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


def compute_positive_count_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count metrics only where the true count is positive.

    This is diagnostic-only. It does not replace the normal all-row count metrics.
    It helps separate the occurrence problem from the positive-count severity problem.
    """

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

        positive_df = group_df.loc[group_df["y_true"] > 0].copy()
        if positive_df.empty:
            summary_rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "horizon": int(horizon),
                    "n_obs": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mean_true": np.nan,
                    "mean_pred": np.nan,
                    "positive_count_rate": 0.0,
                }
            )
            continue

        metrics = evaluate_count_predictions(positive_df["y_true"], positive_df["y_pred"])
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                **metrics,
                "positive_count_rate": float(len(positive_df) / len(group_df)),
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))


def _assign_count_bucket(true_count: float) -> tuple[int, str, float, float]:
    """Return bucket metadata for a raw true-count value."""

    value = float(true_count)
    for bucket_order, (bucket_name, lower, upper) in enumerate(COUNT_BUCKETS):
        if np.isinf(upper):
            if value >= lower:
                return bucket_order, bucket_name, lower, upper
        elif lower <= value <= upper:
            return bucket_order, bucket_name, lower, upper

    raise ValueError(f"Count value did not fit any bucket: {true_count}")


def compute_count_bucket_diagnostics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count diagnostics by true-count bucket.

    Buckets are based on y_true, not y_pred. The purpose is to check whether
    predicted counts increase as true count severity increases, and where the
    model starts underpredicting badly.
    """

    validate_prediction_frame(
        prediction_df=prediction_df,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_true",
            "y_pred",
        ],
        split_col="split",
    )

    bucket_metadata = prediction_df["y_true"].apply(_assign_count_bucket)
    diagnostic_df = prediction_df.copy()
    diagnostic_df["bucket_order"] = bucket_metadata.apply(lambda item: item[0])
    diagnostic_df["count_bucket"] = bucket_metadata.apply(lambda item: item[1])
    diagnostic_df["bucket_lower"] = bucket_metadata.apply(lambda item: item[2])
    diagnostic_df["bucket_upper"] = bucket_metadata.apply(lambda item: item[3])

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = [
        "model_name",
        "split",
        "horizon",
        "bucket_order",
        "count_bucket",
        "bucket_lower",
        "bucket_upper",
    ]

    for group_key, group_df in diagnostic_df.groupby(group_columns, sort=True):
        (
            model_name,
            split_name,
            horizon,
            bucket_order,
            count_bucket,
            bucket_lower,
            bucket_upper,
        ) = group_key

        y_true = group_df["y_true"].astype(float)
        y_pred = group_df["y_pred"].astype(float)
        error = y_pred - y_true
        abs_error = error.abs()
        squared_error = error ** 2

        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                "bucket_order": int(bucket_order),
                "count_bucket": str(count_bucket),
                "bucket_lower": float(bucket_lower),
                "bucket_upper": float(bucket_upper),
                "n_obs": int(len(group_df)),
                "mean_true": float(y_true.mean()),
                "mean_pred": float(y_pred.mean()),
                "median_true": float(y_true.median()),
                "median_pred": float(y_pred.median()),
                "mae": float(abs_error.mean()),
                "rmse": float(np.sqrt(squared_error.mean())),
                "mean_error": float(error.mean()),
                "median_error": float(error.median()),
                "mean_pred_over_mean_true": (
                    float(y_pred.mean() / y_true.mean()) if y_true.mean() > 0 else np.nan
                ),
            }
        )

    return sort_diagnostic_rows(pd.DataFrame(summary_rows))

def _assign_predicted_count_bucket(predicted_count: float) -> tuple[int, str, float, float]:
    """Return bucket metadata for a raw predicted-count value."""

    value = float(predicted_count)
    for bucket_order, (bucket_name, lower, upper) in enumerate(PREDICTED_COUNT_BUCKETS):
        if np.isinf(upper):
            if value >= lower:
                return bucket_order, bucket_name, lower, upper
        else:
            # Use half-open intervals [lower, upper) to avoid double-counting
            # boundary values such as exactly 1.0 or exactly 2.0.
            if lower <= value < upper:
                return bucket_order, bucket_name, lower, upper

    raise ValueError(f"Predicted count value did not fit any bucket: {predicted_count}")


def compute_count_prediction_bucket_diagnostics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count diagnostics by predicted-count bucket.

    Buckets are based on y_pred, not y_true. The purpose is to check whether
    rows receiving larger predicted counts actually have larger true counts.
    This tests whether the count head has useful severity-ranking signal.
    """

    validate_prediction_frame(
        prediction_df=prediction_df,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_true",
            "y_pred",
        ],
        split_col="split",
    )

    bucket_metadata = prediction_df["y_pred"].apply(_assign_predicted_count_bucket)

    diagnostic_df = prediction_df.copy()
    diagnostic_df["bucket_order"] = bucket_metadata.apply(lambda item: item[0])
    diagnostic_df["pred_count_bucket"] = bucket_metadata.apply(lambda item: item[1])
    diagnostic_df["bucket_lower"] = bucket_metadata.apply(lambda item: item[2])
    diagnostic_df["bucket_upper"] = bucket_metadata.apply(lambda item: item[3])

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = [
        "model_name",
        "split",
        "horizon",
        "bucket_order",
        "pred_count_bucket",
        "bucket_lower",
        "bucket_upper",
    ]

    for group_key, group_df in diagnostic_df.groupby(group_columns, sort=True):
        (
            model_name,
            split_name,
            horizon,
            bucket_order,
            pred_count_bucket,
            bucket_lower,
            bucket_upper,
        ) = group_key

        y_true = group_df["y_true"].astype(float)
        y_pred = group_df["y_pred"].astype(float)
        error = y_pred - y_true
        abs_error = error.abs()
        squared_error = error ** 2

        n_obs = int(len(group_df))
        n_true_positive = int((y_true > 0).sum())

        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                "bucket_order": int(bucket_order),
                "pred_count_bucket": str(pred_count_bucket),
                "bucket_lower": float(bucket_lower),
                "bucket_upper": float(bucket_upper),
                "n_obs": n_obs,
                "n_true_positive": n_true_positive,
                "true_positive_rate": float(n_true_positive / n_obs) if n_obs > 0 else np.nan,
                "mean_true": float(y_true.mean()),
                "mean_pred": float(y_pred.mean()),
                "median_true": float(y_true.median()),
                "median_pred": float(y_pred.median()),
                "max_true": float(y_true.max()),
                "max_pred": float(y_pred.max()),
                "mae": float(abs_error.mean()),
                "rmse": float(np.sqrt(squared_error.mean())),
                "mean_error": float(error.mean()),
                "median_error": float(error.median()),
            }
        )

    return sort_diagnostic_rows(pd.DataFrame(summary_rows))

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
