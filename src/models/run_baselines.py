from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.metrics import evaluate_binary_probabilities
from src.evaluation.validation import (
    validate_binary_prediction_columns,
    validate_prediction_frame,
)
from src.models.baselines import ClimatologyBaseline, SimplifiedRJBaseline
from src.models.input_layer import EXPECTED_SPLITS, load_modeling_splits


SUPPORTED_HORIZONS = (24, 72)
PREDICTION_TABLE_COLUMNS = [
    "trigger_event_id",
    "split",
    "horizon",
    "model_name",
    "y_true",
    "y_prob",
    "lambda_pred",
]


def collect_baseline_predictions(
    model: Any,
    splits: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = SUPPORTED_HORIZONS,
    id_col: str = "trigger_event_id",
) -> pd.DataFrame:
    """Collect standardized prediction rows for all requested splits and horizons."""
    prediction_frames: list[pd.DataFrame] = []
    for split_name in EXPECTED_SPLITS:
        split_df = splits[split_name]
        for horizon in horizons:
            prediction_frames.append(
                model.predict_table(
                    df=split_df,
                    split_name=split_name,
                    horizon=horizon,
                    id_col=id_col,
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    available_columns = [column for column in PREDICTION_TABLE_COLUMNS if column in predictions.columns]
    return predictions.loc[:, available_columns].copy()


def evaluate_baseline_predictions(
    predictions_df: pd.DataFrame,
    id_col: str = "trigger_event_id",
) -> pd.DataFrame:
    """Validate and evaluate baseline predictions by model, split, and horizon."""
    required_columns = ["split", "horizon", "model_name", "y_true", "y_prob"]
    validate_prediction_frame(
        prediction_df=predictions_df,
        required_columns=required_columns,
        split_col="split",
    )

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]
    for (model_name, split_name, horizon), group_df in predictions_df.groupby(group_columns, sort=True):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=required_columns,
            id_col=id_col if id_col in group_df.columns else None,
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

    metrics_df = pd.DataFrame(summary_rows)
    if metrics_df.empty:
        return metrics_df

    return metrics_df.sort_values(
        by=["model_name", "horizon", "split"],
        key=lambda series: _sort_key(series),
    ).reset_index(drop=True)


def _sort_key(series: pd.Series) -> pd.Series:
    if series.name == "split":
        order = {split_name: index for index, split_name in enumerate(EXPECTED_SPLITS)}
        return series.map(order)
    return series


def run_climatology_baseline(
    dataset_name: str | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
) -> tuple[ClimatologyBaseline, pd.DataFrame, pd.DataFrame]:
    """Fit climatology on train and evaluate on train/val/test."""
    modeling_splits = load_modeling_splits(dataset_name=dataset_name) if splits is None else splits
    model = ClimatologyBaseline().fit(modeling_splits["train"])
    predictions = collect_baseline_predictions(model=model, splits=modeling_splits)
    metrics = evaluate_baseline_predictions(predictions)
    return model, predictions, metrics


def run_rj_baseline(
    dataset_name: str | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
) -> tuple[SimplifiedRJBaseline, pd.DataFrame, pd.DataFrame]:
    """Fit the simplified RJ-style baseline on train and evaluate on train/val/test."""
    modeling_splits = load_modeling_splits(dataset_name=dataset_name) if splits is None else splits
    model = SimplifiedRJBaseline().fit(modeling_splits["train"])
    predictions = collect_baseline_predictions(model=model, splits=modeling_splits)
    metrics = evaluate_baseline_predictions(predictions)
    return model, predictions, metrics


def run_all_baselines(
    dataset_name: str | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run climatology and RJ-style baselines and return combined predictions and metrics."""
    modeling_splits = load_modeling_splits(dataset_name=dataset_name) if splits is None else splits

    _, climatology_predictions, climatology_metrics = run_climatology_baseline(
        splits=modeling_splits
    )
    _, rj_predictions, rj_metrics = run_rj_baseline(
        splits=modeling_splits
    )

    predictions = pd.concat(
        [climatology_predictions, rj_predictions],
        ignore_index=True,
    )
    metrics = pd.concat(
        [climatology_metrics, rj_metrics],
        ignore_index=True,
    )
    metrics = metrics.sort_values(
        by=["model_name", "horizon", "split"],
        key=lambda series: _sort_key(series),
    ).reset_index(drop=True)
    return predictions, metrics
