from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation.metrics import evaluate_count_predictions
from src.models.feature_sets import EXTENDED_TABULAR_FEATURES
from src.models.input_layer import EXPECTED_SPLITS, InputConfig, load_modeling_splits, prepare_tabular_inputs
from src.models.task3_maxmag.labels import build_or_load_max_aftershock_magnitude_labels
from src.models.task3_maxmag.run import merge_task3_maxmag_labels_into_splits
from src.models.task3_maxmag.xgb import Task3MaxMagRegressor
from src.utils.io import save_dataframe


TASK3_MAXMAG_GAP_PREDICTION_COLUMNS = [
    "trigger_event_id",
    "split",
    "horizon",
    "model_name",
    "y_true_gap",
    "y_pred_gap",
    "y_true_maxmag",
    "y_pred_maxmag",
]


def _gap_target_col(horizon: int) -> str:
    return f"max_aftershock_gap_{horizon}h"


def add_gap_targets(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split_name, df in splits.items():
        enriched = df.copy()
        for horizon in (24, 72):
            maxmag_col = f"max_aftershock_mag_{horizon}h"
            gap_col = _gap_target_col(horizon)
            enriched[gap_col] = enriched[maxmag_col] - enriched["trigger_magnitude"]
        out[split_name] = enriched
    return out


def _build_gap_prediction_frame(
    *,
    prepared,
    merged_splits: dict[str, pd.DataFrame],
    horizon: int,
    model_name: str,
    predict_fn,
) -> pd.DataFrame:
    split_rows: list[pd.DataFrame] = []
    target_col = _gap_target_col(horizon)
    maxmag_col = f"max_aftershock_mag_{horizon}h"
    split_payloads = [
        ("train", prepared.X_train, prepared.y_train),
        ("val", prepared.X_val, prepared.y_val),
        ("test", prepared.X_test, prepared.y_test),
    ]
    for split_name, X_split, y_gap in split_payloads:
        split_source = merged_splits[split_name].loc[y_gap.index]
        pred_gap = predict_fn(X_split)
        y_true_maxmag = split_source[maxmag_col].astype(float).to_numpy()
        y_pred_maxmag = split_source["trigger_magnitude"].astype(float).to_numpy() + pred_gap
        split_rows.append(
            pd.DataFrame(
                {
                    "trigger_event_id": split_source["trigger_event_id"].to_numpy(),
                    "split": split_name,
                    "horizon": horizon,
                    "model_name": model_name,
                    "y_true_gap": y_gap.astype(float).to_numpy(),
                    "y_pred_gap": pred_gap,
                    "y_true_maxmag": y_true_maxmag,
                    "y_pred_maxmag": y_pred_maxmag,
                }
            )
        )
    return pd.concat(split_rows, ignore_index=True)


def _fit_linear_gap_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_gap_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for (model_name, split_name, horizon), group_df in predictions_df.groupby(
        ["model_name", "split", "horizon"],
        sort=True,
    ):
        metrics_maxmag = evaluate_count_predictions(group_df["y_true_maxmag"], group_df["y_pred_maxmag"])
        metrics_gap = evaluate_count_predictions(group_df["y_true_gap"], group_df["y_pred_gap"])
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                "mae_maxmag": metrics_maxmag["mae"],
                "rmse_maxmag": metrics_maxmag["rmse"],
                "bias_maxmag": metrics_maxmag["bias"],
                "mae_gap": metrics_gap["mae"],
                "rmse_gap": metrics_gap["rmse"],
                "bias_gap": metrics_gap["bias"],
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["model_name", "horizon", "split"]).reset_index(drop=True)


def run_task3_maxmag_gap_pipeline(
    dataset_name: str,
    min_aftershock_magnitude: float = 2.5,
    backend: str = "xgboost",
    tune_xgboost: bool = True,
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
    merged_splits = add_gap_targets(merged_splits)

    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "min_aftershock_magnitude": min_aftershock_magnitude,
        "feature_set": "extended",
        "target_type": "gap_plus_trigger_magnitude",
        "xgboost_models": {},
    }

    prediction_frames: list[pd.DataFrame] = []

    for horizon in (24, 72):
        target_col = _gap_target_col(horizon)
        config = InputConfig(
            feature_cols=list(EXTENDED_TABULAR_FEATURES),
            target_col=target_col,
            missing_strategy="median",
            scale=False,
            drop_rows_with_missing_target=True,
            allow_missing_optional=True,
        )
        prepared = prepare_tabular_inputs(config=config, splits=merged_splits)

        train_mean_gap = float(prepared.y_train.mean())
        prediction_frames.append(
            _build_gap_prediction_frame(
                prepared=prepared,
                merged_splits=merged_splits,
                horizon=horizon,
                model_name="task3_maxmag_gap_mean",
                predict_fn=lambda X, mean_gap=train_mean_gap: pd.Series(mean_gap, index=X.index).to_numpy(dtype=float),
            )
        )

        linear_model = _fit_linear_gap_baseline(prepared.X_train, prepared.y_train)
        prediction_frames.append(
            _build_gap_prediction_frame(
                prepared=prepared,
                merged_splits=merged_splits,
                horizon=horizon,
                model_name="task3_maxmag_gap_linear",
                predict_fn=lambda X, model=linear_model: model.predict(X),
            )
        )

        model_variants = [("xgboost", False)] if backend == "xgboost" else [(backend, False)]
        if backend == "xgboost" and tune_xgboost:
            model_variants.append(("xgboost", True))

        for variant_backend, variant_tune in model_variants:
            model = Task3MaxMagRegressor(
                backend=variant_backend,
                tune=variant_tune,
            ).fit(
                prepared.X_train,
                prepared.y_train,
                X_val=prepared.X_val,
                y_val=prepared.y_val,
            )
            model_name = f"task3_maxmag_gap_{model.backend_}"
            if variant_tune:
                model_name = f"{model_name}_tuned"
            prediction_frames.append(
                _build_gap_prediction_frame(
                    prepared=prepared,
                    merged_splits=merged_splits,
                    horizon=horizon,
                    model_name=model_name,
                    predict_fn=lambda X, fitted=model: fitted.predict(X),
                )
            )
            if model.backend_ == "xgboost":
                metadata["xgboost_models"].setdefault(f"{horizon}h", {})
                metadata["xgboost_models"][f"{horizon}h"]["tuned" if variant_tune else "untuned"] = {
                    "model_name": model_name,
                    "params": model.best_params_,
                    "val_metrics": model.best_val_metrics_,
                }

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df = predictions_df.loc[:, TASK3_MAXMAG_GAP_PREDICTION_COLUMNS].copy()
    metrics_df = evaluate_gap_predictions(predictions_df)
    return predictions_df, metrics_df, metadata


def save_task3_maxmag_gap_outputs(
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
