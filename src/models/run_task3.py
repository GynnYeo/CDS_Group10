from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import evaluate_binary_probabilities
from src.evaluation.validation import (
    validate_binary_prediction_columns,
    validate_prediction_frame,
)
from src.models.feature_sets import EXTENDED_TABULAR_FEATURES
from src.models.input_layer import (
    EXPECTED_SPLITS,
    InputConfig,
    PreparedInputs,
    load_modeling_splits,
    prepare_tabular_inputs,
)
from src.models.task3_labels import build_or_load_large_aftershock_labels
from src.models.task3_rj import SimplifiedTask3RJBaseline
from src.models.task3_xgb import Task3BinaryModel
from src.utils.io import save_dataframe
from src.utils.paths import METRICS_DIR


TASK3_MODEL_NAME = "task3_large_aftershock"
TASK3_PREDICTION_COLUMNS = [
    "trigger_event_id",
    "split",
    "horizon",
    "model_name",
    "y_true",
    "y_prob",
    "lambda_pred",
]


def merge_task3_labels_into_splits(
    splits: dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    merged_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_df in splits.items():
        merged_df = split_df.merge(label_df, how="left", on="trigger_event_id")
        missing_targets = merged_df[["y_large_24h", "y_large_72h"]].isna().any(axis=1)
        if missing_targets.any():
            raise ValueError(
                f"Task 3 label merge produced missing targets in split '{split_name}'."
            )
        merged_splits[split_name] = merged_df
    return merged_splits


def build_task3_climatology_predictions(
    splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []
    for horizon in (24, 72):
        target_col = f"y_large_{horizon}h"
        constant_prob = float(splits["train"][target_col].mean())
        for split_name in EXPECTED_SPLITS:
            split_df = splits[split_name]
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "trigger_event_id": split_df["trigger_event_id"].to_numpy(),
                        "split": split_name,
                        "horizon": horizon,
                        "model_name": "task3_climatology",
                        "y_true": split_df[target_col].astype(float).to_numpy(),
                        "y_prob": constant_prob,
                        "lambda_pred": pd.NA,
                    }
                )
            )
    return pd.concat(prediction_frames, ignore_index=True)


def build_task3_rj_predictions(
    splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    model = SimplifiedTask3RJBaseline().fit(splits["train"])
    prediction_frames: list[pd.DataFrame] = []
    for split_name in EXPECTED_SPLITS:
        split_df = splits[split_name]
        for horizon in (24, 72):
            prediction_frames.append(
                model.predict_table(
                    df=split_df,
                    split_name=split_name,
                    horizon=horizon,
                )
            )
    return pd.concat(prediction_frames, ignore_index=True)


def collect_task3_model_predictions(
    prepared: PreparedInputs,
    merged_splits: dict[str, pd.DataFrame],
    model: Task3BinaryModel,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    split_rows = []
    split_payloads = [
        ("train", prepared.X_train, prepared.y_train),
        ("val", prepared.X_val, prepared.y_val),
        ("test", prepared.X_test, prepared.y_test),
    ]
    for split_name, X_split, y_split in split_payloads:
        split_source = merged_splits[split_name].loc[y_split.index]
        y_prob = model.predict_proba(X_split)
        split_rows.append(
            pd.DataFrame(
                {
                    "trigger_event_id": split_source["trigger_event_id"].to_numpy(),
                    "split": split_name,
                    "horizon": horizon,
                    "model_name": model_name,
                    "y_true": y_split.astype(float).to_numpy(),
                    "y_prob": y_prob,
                    "lambda_pred": pd.NA,
                }
            )
        )
    return pd.concat(split_rows, ignore_index=True)


def evaluate_task3_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = ["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_prob"]
    validate_prediction_frame(
        prediction_df=predictions_df,
        required_columns=required_columns,
        split_col="split",
    )

    summary_rows: list[dict[str, Any]] = []
    for (model_name, split_name, horizon), group_df in predictions_df.groupby(
        ["model_name", "split", "horizon"],
        sort=True,
    ):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=required_columns,
            id_col="trigger_event_id",
            split_col="split",
        )
        validate_binary_prediction_columns(group_df, "y_true", "y_prob")
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

    split_order = {name: idx for idx, name in enumerate(EXPECTED_SPLITS)}
    return metrics_df.sort_values(
        by=["model_name", "horizon", "split"],
        key=lambda series: series.map(split_order) if series.name == "split" else series,
    ).reset_index(drop=True)


def run_task3_pipeline(
    dataset_name: str,
    magnitude_threshold: float = 4.0,
    feature_cols: list[str] | None = None,
    backend: str = "auto",
    force_recompute_labels: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the task-3 sidecar pipeline from saved processed splits.
    """
    splits = load_modeling_splits(dataset_name=dataset_name)
    label_df = build_or_load_large_aftershock_labels(
        dataset_name=dataset_name,
        magnitude_threshold=magnitude_threshold,
        force_recompute=force_recompute_labels,
        verbose=verbose,
    )
    merged_splits = merge_task3_labels_into_splits(splits, label_df)

    requested_features = feature_cols or list(EXTENDED_TABULAR_FEATURES)
    all_prediction_frames = [
        build_task3_climatology_predictions(merged_splits),
        build_task3_rj_predictions(merged_splits),
    ]

    for horizon in (24, 72):
        target_col = f"y_large_{horizon}h"
        config = InputConfig(
            feature_cols=requested_features,
            target_col=target_col,
            missing_strategy="median",
            scale=False,
            drop_rows_with_missing_target=True,
            allow_missing_optional=True,
        )
        prepared = prepare_tabular_inputs(config=config, splits=merged_splits)
        model = Task3BinaryModel(backend=backend).fit(
            prepared.X_train,
            prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
        )
        model_name = f"{TASK3_MODEL_NAME}_{model.backend_}"
        if verbose:
            print(
                f"[FIT] horizon={horizon}h backend={model.backend_} "
                f"train_rows={len(prepared.X_train):,} features={len(prepared.feature_cols)}"
            )
        all_prediction_frames.append(
            collect_task3_model_predictions(
                prepared=prepared,
                merged_splits=merged_splits,
                model=model,
                horizon=horizon,
                model_name=model_name,
            )
        )

    predictions_df = pd.concat(all_prediction_frames, ignore_index=True)
    predictions_df = predictions_df.loc[:, TASK3_PREDICTION_COLUMNS].copy()
    metrics_df = evaluate_task3_predictions(predictions_df)
    return predictions_df, metrics_df


def save_task3_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_path: Path,
    metrics_path: Path,
) -> None:
    save_dataframe(predictions_df, predictions_path)
    save_dataframe(metrics_df, metrics_path)
