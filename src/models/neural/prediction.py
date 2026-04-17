from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.validation import (
    validate_binary_prediction_columns,
    validate_prediction_frame,
)
from src.models.input_layer import EXPECTED_SPLITS
from src.models.neural.data import (
    COUNT_TARGET_COLUMNS,
    MAGNITUDE_TARGET_COLUMNS,
    PROBABILITY_TARGET_COLUMNS,
    PreparedNeuralInputs,
)


HORIZONS = (24, 72)
PROBABILITY_TARGETS_BY_HORIZON = list(zip(HORIZONS, PROBABILITY_TARGET_COLUMNS))
COUNT_TARGETS_BY_HORIZON = list(zip(HORIZONS, COUNT_TARGET_COLUMNS))
MAGNITUDE_TARGETS_BY_HORIZON = list(zip(HORIZONS, MAGNITUDE_TARGET_COLUMNS))


def predict_split_outputs(
    model: torch.nn.Module,
    features_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
    count_modeling_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run split-level inference and return final probability, count, and magnitude outputs.

    In ``conditional_positive`` mode, ``outputs["count_pred"]`` is interpreted as
    positive-case severity on the log-count scale before probability weighting.
    The returned count array is always the final standardized count prediction used
    for CSV export and downstream count evaluation.
    """

    model.eval()
    feature_loader = DataLoader(
        TensorDataset(features_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    probability_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    magnitude_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for (features_batch,) in feature_loader:
            features_batch = features_batch.to(device)
            outputs = model(features_batch)
            probability_chunks.append(
                torch.sigmoid(outputs["prob_logits"]).cpu().numpy()
            )
            count_chunks.append(outputs["count_pred"].cpu().numpy())
            magnitude_chunks.append(outputs["magnitude_pred"].cpu().numpy())

    probability_predictions = np.concatenate(probability_chunks, axis=0)
    count_log_predictions = np.concatenate(count_chunks, axis=0)
    if count_modeling_mode == "standard":
        count_predictions = np.expm1(count_log_predictions)
        count_predictions = np.clip(count_predictions, a_min=0.0, a_max=None)
    elif count_modeling_mode == "conditional_positive":
        # In conditional mode the count head estimates severity given a positive
        # case, so convert back to count space before weighting by event
        # probability to recover the final unconditional count prediction.
        positive_count_predictions = np.expm1(count_log_predictions)
        positive_count_predictions = np.clip(
            positive_count_predictions,
            a_min=0.0,
            a_max=None,
        )
        count_predictions = probability_predictions * positive_count_predictions
    else:
        raise ValueError(
            "count_modeling_mode must be 'standard' or 'conditional_positive', "
            f"got {count_modeling_mode!r}."
        )
    magnitude_predictions = np.concatenate(magnitude_chunks, axis=0)
    return probability_predictions, count_predictions, magnitude_predictions


def format_probability_predictions(
    ids: pd.Series,
    split_labels: pd.Series,
    y_true: pd.DataFrame,
    y_prob: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Format long-form probability predictions for one split."""

    frames: list[pd.DataFrame] = []
    for index, (horizon, target_column) in enumerate(PROBABILITY_TARGETS_BY_HORIZON):
        frames.append(
            pd.DataFrame(
                {
                    "trigger_event_id": ids.to_numpy(),
                    "split": split_labels.to_numpy(),
                    "horizon": horizon,
                    "model_name": model_name,
                    "y_true": y_true[target_column].to_numpy(dtype=float),
                    "y_prob": y_prob[:, index].astype(float),
                }
            )
        )

    prediction_df = pd.concat(frames, ignore_index=True)
    validate_prediction_frame(
        prediction_df=prediction_df,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_true",
            "y_prob",
        ],
        split_col="split",
    )
    for (_, _), group_df in prediction_df.groupby(["split", "horizon"], sort=False):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=[
                "trigger_event_id",
                "split",
                "horizon",
                "model_name",
                "y_true",
                "y_prob",
            ],
            id_col="trigger_event_id",
            split_col="split",
        )
        validate_binary_prediction_columns(
            prediction_df=group_df,
            y_true_col="y_true",
            y_prob_col="y_prob",
        )
    return prediction_df


def format_count_predictions(
    ids: pd.Series,
    split_labels: pd.Series,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Format long-form count predictions for one split."""

    frames: list[pd.DataFrame] = []
    for index, (horizon, target_column) in enumerate(COUNT_TARGETS_BY_HORIZON):
        frames.append(
            pd.DataFrame(
                {
                    "trigger_event_id": ids.to_numpy(),
                    "split": split_labels.to_numpy(),
                    "horizon": horizon,
                    "model_name": model_name,
                    "y_true": y_true[target_column].to_numpy(dtype=float),
                    "y_pred": y_pred[:, index].astype(float),
                }
            )
        )

    prediction_df = pd.concat(frames, ignore_index=True)
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
    for (_, _), group_df in prediction_df.groupby(["split", "horizon"], sort=False):
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
    return prediction_df


def format_magnitude_predictions(
    ids: pd.Series,
    split_labels: pd.Series,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Format long-form conditional maximum-magnitude predictions for one split."""

    frames: list[pd.DataFrame] = []
    for index, (horizon, target_column) in enumerate(MAGNITUDE_TARGETS_BY_HORIZON):
        frames.append(
            pd.DataFrame(
                {
                    "trigger_event_id": ids.to_numpy(),
                    "split": split_labels.to_numpy(),
                    "horizon": horizon,
                    "model_name": model_name,
                    "y_true": y_true[target_column].to_numpy(dtype=float),
                    "y_pred": y_pred[:, index].astype(float),
                    "target_available": y_true[target_column].notna().to_numpy(),
                }
            )
        )

    prediction_df = pd.concat(frames, ignore_index=True)
    validate_prediction_frame(
        prediction_df=prediction_df,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_pred",
            "target_available",
        ],
        split_col="split",
    )
    return prediction_df


def collect_prediction_tables(
    model: torch.nn.Module,
    prepared: PreparedNeuralInputs,
    feature_tensors: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    model_name: str,
    count_modeling_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate standardized prediction tables for all project splits.

    The count table keeps the existing CSV schema in both modes. Only the meaning
    of the exported ``y_pred`` changes in ``conditional_positive`` mode, where it
    becomes the probability-weighted final count prediction.
    """

    probability_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []
    magnitude_frames: list[pd.DataFrame] = []

    split_metadata = {
        "train": (
            prepared.train_ids,
            prepared.train_splits,
            prepared.y_prob_train,
            prepared.y_count_train,
            prepared.y_magnitude_train,
        ),
        "val": (
            prepared.val_ids,
            prepared.val_splits,
            prepared.y_prob_val,
            prepared.y_count_val,
            prepared.y_magnitude_val,
        ),
        "test": (
            prepared.test_ids,
            prepared.test_splits,
            prepared.y_prob_test,
            prepared.y_count_test,
            prepared.y_magnitude_test,
        ),
    }

    for split_name in EXPECTED_SPLITS:
        probability_predictions, count_predictions, magnitude_predictions = (
            predict_split_outputs(
                model=model,
                features_tensor=feature_tensors[split_name],
                batch_size=batch_size,
                device=device,
                count_modeling_mode=count_modeling_mode,
            )
        )
        ids, split_labels, probability_targets, count_targets, magnitude_targets = split_metadata[split_name]
        probability_frames.append(
            format_probability_predictions(
                ids=ids,
                split_labels=split_labels,
                y_true=probability_targets,
                y_prob=probability_predictions,
                model_name=model_name,
            )
        )
        count_frames.append(
            format_count_predictions(
                ids=ids,
                split_labels=split_labels,
                y_true=count_targets,
                y_pred=count_predictions,
                model_name=model_name,
            )
        )
        magnitude_frames.append(
            format_magnitude_predictions(
                ids=ids,
                split_labels=split_labels,
                y_true=magnitude_targets,
                y_pred=magnitude_predictions,
                model_name=model_name,
            )
        )

    probability_df = pd.concat(probability_frames, ignore_index=True)
    count_df = pd.concat(count_frames, ignore_index=True)
    magnitude_df = pd.concat(magnitude_frames, ignore_index=True)
    return probability_df, count_df, magnitude_df
