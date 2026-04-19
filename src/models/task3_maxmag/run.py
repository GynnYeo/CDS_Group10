from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import evaluate_count_predictions
from src.models.feature_sets import EXTENDED_TABULAR_FEATURES, MAXMAG_COMPACT_FEATURES
from src.models.input_layer import EXPECTED_SPLITS, InputConfig, PreparedInputs, load_modeling_splits, prepare_tabular_inputs
from src.models.task3_maxmag.baselines import (
    BathsLawBaseline,
    LinearMaxMagnitudeBaseline,
    MagnitudeOnlyLinearMaxMagnitudeBaseline,
    MeanMaxMagnitudeBaseline,
)
from src.models.task3_maxmag.labels import build_or_load_max_aftershock_magnitude_labels
from src.models.task3_maxmag.xgb import Task3MaxMagRegressor
from src.utils.io import save_dataframe


TASK3_MAXMAG_MODEL_NAME = "task3_maxmag"
TASK3_MAXMAG_PREDICTION_COLUMNS = [
    "trigger_event_id",
    "split",
    "horizon",
    "model_name",
    "y_true",
    "y_pred",
]


def merge_task3_maxmag_labels_into_splits(
    splits: dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    merged_splits: dict[str, pd.DataFrame] = {}
    label_cols = [column for column in label_df.columns if column != "trigger_event_id"]
    for split_name, split_df in splits.items():
        missing_label_cols = [column for column in label_cols if column not in split_df.columns]
        if missing_label_cols:
            merge_cols = ["trigger_event_id", *missing_label_cols]
            merged_df = split_df.merge(label_df.loc[:, merge_cols], how="left", on="trigger_event_id")
        else:
            merged_df = split_df.copy()
        merged_splits[split_name] = merged_df
    return merged_splits


def build_task3_maxmag_baseline_predictions(
    splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    mean_model = MeanMaxMagnitudeBaseline().fit(splits["train"])
    baths_model = BathsLawBaseline().fit(splits["train"])
    magonly_linear_model = MagnitudeOnlyLinearMaxMagnitudeBaseline().fit(splits["train"])
    linear_model = LinearMaxMagnitudeBaseline().fit(splits["train"])
    prediction_frames: list[pd.DataFrame] = []
    for model in (mean_model, baths_model, magonly_linear_model, linear_model):
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


def collect_task3_maxmag_model_predictions(
    prepared: PreparedInputs,
    merged_splits: dict[str, pd.DataFrame],
    model: Task3MaxMagRegressor,
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
        split_rows.append(
            pd.DataFrame(
                {
                    "trigger_event_id": split_source["trigger_event_id"].to_numpy(),
                    "split": split_name,
                    "horizon": horizon,
                    "model_name": model_name,
                    "y_true": y_split.astype(float).to_numpy(),
                    "y_pred": model.predict(X_split),
                }
            )
        )
    return pd.concat(split_rows, ignore_index=True)


def evaluate_task3_maxmag_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_pred"]
    missing = [column for column in required_columns if column not in predictions_df.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {missing}")

    summary_rows: list[dict[str, Any]] = []
    for (model_name, split_name, horizon), group_df in predictions_df.groupby(
        ["model_name", "split", "horizon"],
        sort=True,
    ):
        if group_df.empty:
            continue
        if group_df[["y_true", "y_pred"]].isna().any().any():
            raise ValueError("Task 3 max-magnitude predictions contain missing y_true/y_pred values.")
        metrics = evaluate_count_predictions(group_df["y_true"], group_df["y_pred"])
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


def run_task3_maxmag_pipeline(
    dataset_name: str,
    min_aftershock_magnitude: float = 2.5,
    feature_cols: list[str] | None = None,
    feature_set: str = "extended",
    backend: str = "auto",
    tabnet_max_epochs: int | None = None,
    tabnet_patience: int = 20,
    tabnet_batch_size: int = 1024,
    tabnet_virtual_batch_size: int = 128,
    tabnet_lr: float = 0.02,
    tabnet_n_d: int = 8,
    tabnet_n_a: int = 8,
    tabnet_n_steps: int = 3,
    tabnet_gamma: float = 1.3,
    tune_xgboost: bool = False,
    early_stopping_rounds: int | None = None,
    force_recompute_labels: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    splits = load_modeling_splits(dataset_name=dataset_name)
    label_df = build_or_load_max_aftershock_magnitude_labels(
        dataset_name=dataset_name,
        min_aftershock_magnitude=min_aftershock_magnitude,
        force_recompute=force_recompute_labels,
        verbose=verbose,
    )
    merged_splits = merge_task3_maxmag_labels_into_splits(splits, label_df)
    if feature_cols is not None:
        requested_features = feature_cols
        feature_set_name = "custom"
    elif feature_set == "compact":
        requested_features = list(MAXMAG_COMPACT_FEATURES)
        feature_set_name = "compact"
    else:
        requested_features = list(EXTENDED_TABULAR_FEATURES)
        feature_set_name = "extended"
    model_metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "min_aftershock_magnitude": min_aftershock_magnitude,
        "feature_set": feature_set_name,
        "n_requested_features": len(requested_features),
        "early_stopping_rounds": early_stopping_rounds,
        "xgboost_models": {},
    }

    all_prediction_frames = [build_task3_maxmag_baseline_predictions(merged_splits)]
    for horizon in (24, 72):
        target_col = f"max_aftershock_mag_{horizon}h"
        config = InputConfig(
            feature_cols=requested_features,
            target_col=target_col,
            missing_strategy="median",
            scale=False,
            drop_rows_with_missing_target=True,
            allow_missing_optional=True,
        )
        prepared = prepare_tabular_inputs(config=config, splits=merged_splits)
        model_variants = [("xgboost", False)] if backend == "xgboost" else [(backend, tune_xgboost)]
        if backend == "xgboost" and tune_xgboost:
            model_variants.append(("xgboost", True))

        for variant_backend, variant_tune in model_variants:
            model = Task3MaxMagRegressor(
                backend=variant_backend,
                tabnet_max_epochs=tabnet_max_epochs,
                tabnet_patience=tabnet_patience,
                tabnet_batch_size=tabnet_batch_size,
                tabnet_virtual_batch_size=tabnet_virtual_batch_size,
                tabnet_lr=tabnet_lr,
                tabnet_n_d=tabnet_n_d,
                tabnet_n_a=tabnet_n_a,
                tabnet_n_steps=tabnet_n_steps,
                tabnet_gamma=tabnet_gamma,
                tune=variant_tune,
                early_stopping_rounds=early_stopping_rounds,
            ).fit(
                prepared.X_train,
                prepared.y_train,
                X_val=prepared.X_val,
                y_val=prepared.y_val,
            )
            model_name = f"{TASK3_MAXMAG_MODEL_NAME}_{model.backend_}"
            if variant_tune:
                model_name = f"{model_name}_tuned"
            if verbose:
                print(
                    f"[FIT] maxmag horizon={horizon}h backend={model.backend_} "
                    f"tuned={variant_tune} train_rows={len(prepared.X_train):,} "
                    f"features={len(prepared.feature_cols)}"
                )
                if model.best_params_ is not None and model.best_val_metrics_ is not None:
                    print(f"[TUNE] best_params={model.best_params_}")
                    print(f"[TUNE] val_metrics={model.best_val_metrics_}")
            if model.backend_ == "xgboost":
                horizon_key = f"{horizon}h"
                if horizon_key not in model_metadata["xgboost_models"]:
                    model_metadata["xgboost_models"][horizon_key] = {}
                variant_key = "tuned" if variant_tune else "untuned"
                model_metadata["xgboost_models"][horizon_key][variant_key] = {
                    "model_name": model_name,
                    "params": model.best_params_,
                    "val_metrics": model.best_val_metrics_,
                }
            all_prediction_frames.append(
                collect_task3_maxmag_model_predictions(
                    prepared=prepared,
                    merged_splits=merged_splits,
                    model=model,
                    horizon=horizon,
                    model_name=model_name,
                )
            )

    predictions_df = pd.concat(all_prediction_frames, ignore_index=True)
    predictions_df = predictions_df.loc[:, TASK3_MAXMAG_PREDICTION_COLUMNS].copy()
    metrics_df = evaluate_task3_maxmag_predictions(predictions_df)
    return predictions_df, metrics_df, model_metadata


def save_task3_maxmag_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_path: Path,
    metrics_path: Path,
    metadata_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    save_dataframe(predictions_df, predictions_path)
    save_dataframe(metrics_df, metrics_path)
    if metadata_path is not None and metadata is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
