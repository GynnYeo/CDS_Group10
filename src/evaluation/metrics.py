from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss as sklearn_log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _as_numpy(values: pd.Series | pd.DataFrame | np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values)


def evaluate_binary_probabilities(
    y_true: pd.Series | np.ndarray | list[float],
    y_prob: pd.Series | np.ndarray | list[float],
    clip_eps: float = 1e-15,
) -> dict[str, float]:
    """Evaluate binary probabilistic predictions."""
    y_true_arr = _as_numpy(y_true).astype(float)
    y_prob_arr = np.clip(_as_numpy(y_prob).astype(float), clip_eps, 1.0 - clip_eps)

    unique_classes = np.unique(y_true_arr[~np.isnan(y_true_arr)])
    roc_auc = np.nan
    if len(unique_classes) >= 2:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    return {
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "log_loss": float(sklearn_log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        "roc_auc": roc_auc,
        "n_obs": int(len(y_true_arr)),
        "positive_rate": float(np.mean(y_true_arr)),
    }


def evaluate_count_predictions(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> dict[str, float]:
    """Evaluate count predictions with simple regression metrics."""
    y_true_arr = _as_numpy(y_true).astype(float)
    y_pred_arr = _as_numpy(y_pred).astype(float)

    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "n_obs": int(len(y_true_arr)),
        "mean_true": float(np.mean(y_true_arr)),
        "mean_pred": float(np.mean(y_pred_arr)),
    }

def evaluate_capped_count_predictions(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
    cap: float,
) -> dict[str, float]:
    """Evaluate count predictions after clipping true and predicted counts at cap."""
    if cap <= 0:
        raise ValueError("cap must be positive.")

    y_true_arr = _as_numpy(y_true).astype(float)
    y_pred_arr = _as_numpy(y_pred).astype(float)

    y_true_capped = np.minimum(y_true_arr, cap)
    y_pred_capped = np.minimum(y_pred_arr, cap)

    true_above_cap = y_true_arr >= cap
    pred_above_cap = y_pred_arr >= cap

    return {
        "capped_mae": float(mean_absolute_error(y_true_capped, y_pred_capped)),
        "capped_rmse": float(np.sqrt(mean_squared_error(y_true_capped, y_pred_capped))),
        "n_obs": int(len(y_true_arr)),
        "count_cap": float(cap),
        "mean_true_capped": float(np.mean(y_true_capped)),
        "mean_pred_capped": float(np.mean(y_pred_capped)),
        "n_true_at_or_above_cap": int(np.sum(true_above_cap)),
        "n_pred_at_or_above_cap": int(np.sum(pred_above_cap)),
        "n_correct_at_or_above_cap": int(np.sum(true_above_cap & pred_above_cap)),
        "n_missed_at_or_above_cap": int(np.sum(true_above_cap & ~pred_above_cap)),
        "n_false_alarm_at_or_above_cap": int(np.sum(~true_above_cap & pred_above_cap)),
    }


def evaluate_prediction_table(
    prediction_df: pd.DataFrame,
    y_true_col: str,
    prediction_col: str,
    task: str = "binary",
) -> dict[str, float]:
    """Evaluate a simple prediction table for binary or count tasks."""
    if y_true_col not in prediction_df.columns:
        raise ValueError(f"Missing y_true column: {y_true_col}")
    if prediction_col not in prediction_df.columns:
        raise ValueError(f"Missing prediction column: {prediction_col}")

    if task == "binary":
        return evaluate_binary_probabilities(
            prediction_df[y_true_col],
            prediction_df[prediction_col],
        )
    if task == "count":
        return evaluate_count_predictions(
            prediction_df[y_true_col],
            prediction_df[prediction_col],
        )

    raise ValueError("task must be either 'binary' or 'count'.")
