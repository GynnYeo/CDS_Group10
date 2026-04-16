from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.input_layer import EXPECTED_SPLITS
from src.models.neural.data import PreparedNeuralInputs


class MultiTaskTensorDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Dataset returning features plus probability, count, and magnitude targets."""

    def __init__(
        self,
        features: torch.Tensor,
        probability_targets: torch.Tensor,
        count_targets: torch.Tensor,
        magnitude_targets: torch.Tensor,
        magnitude_masks: torch.Tensor,
    ) -> None:
        if not (
            len(features)
            == len(probability_targets)
            == len(count_targets)
            == len(magnitude_targets)
            == len(magnitude_masks)
        ):
            raise ValueError("Features, targets, and masks must have matching lengths.")

        self.features = features
        self.probability_targets = probability_targets
        self.count_targets = count_targets
        self.magnitude_targets = magnitude_targets
        self.magnitude_masks = magnitude_masks

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.probability_targets[index],
            self.count_targets[index],
            self.magnitude_targets[index],
            self.magnitude_masks[index],
        )


def dataframe_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Convert a pandas DataFrame to a float32 tensor."""

    return torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)


def counts_to_log_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Apply log1p to non-negative count targets and convert to float32."""

    count_values = df.to_numpy(dtype=np.float32)
    if np.any(count_values < 0):
        raise ValueError("Count targets must be non-negative before log1p transform.")
    return torch.tensor(np.log1p(count_values), dtype=torch.float32)


def magnitude_to_tensor_and_mask(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert magnitude targets to tensors plus a non-missing target mask."""

    magnitude_values = df.to_numpy(dtype=np.float32)
    magnitude_mask = ~np.isnan(magnitude_values)

    # Replace NaN with 0 only as a harmless placeholder.
    # The mask prevents these placeholder values from contributing to loss.
    filled_values = np.nan_to_num(magnitude_values, nan=0.0)

    return (
        torch.tensor(filled_values, dtype=torch.float32),
        torch.tensor(magnitude_mask, dtype=torch.bool),
    )


def build_data_loaders(
    prepared: PreparedNeuralInputs,
    batch_size: int,
) -> tuple[dict[str, DataLoader], dict[str, torch.Tensor]]:
    """Build split-wise DataLoaders and cached feature tensors for inference."""

    feature_tensors = {
        "train": dataframe_to_tensor(prepared.X_train),
        "val": dataframe_to_tensor(prepared.X_val),
        "test": dataframe_to_tensor(prepared.X_test),
    }
    probability_tensors = {
        "train": dataframe_to_tensor(prepared.y_prob_train),
        "val": dataframe_to_tensor(prepared.y_prob_val),
        "test": dataframe_to_tensor(prepared.y_prob_test),
    }
    count_tensors = {
        "train": counts_to_log_tensor(prepared.y_count_train),
        "val": counts_to_log_tensor(prepared.y_count_val),
        "test": counts_to_log_tensor(prepared.y_count_test),
    }

    magnitude_tensors: dict[str, torch.Tensor] = {}
    magnitude_masks: dict[str, torch.Tensor] = {}
    for split_name in EXPECTED_SPLITS:
        magnitude_tensors[split_name], magnitude_masks[split_name] = (
            magnitude_to_tensor_and_mask(
                getattr(prepared, f"y_magnitude_{split_name}")
            )
        )

    loaders = {
        split_name: DataLoader(
            MultiTaskTensorDataset(
                features=feature_tensors[split_name],
                probability_targets=probability_tensors[split_name],
                count_targets=count_tensors[split_name],
                magnitude_targets=magnitude_tensors[split_name],
                magnitude_masks=magnitude_masks[split_name],
            ),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )
        for split_name in EXPECTED_SPLITS
    }
    return loaders, feature_tensors
