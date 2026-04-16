from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskMLP(nn.Module):
    """Simple shared-trunk MLP with probability, count, and magnitude heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")
        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        self.shared_trunk = nn.Sequential(*layers)
        self.probability_head = nn.Linear(in_features, 2)
        self.count_head = nn.Linear(in_features, 2)
        self.magnitude_head = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw logits plus real-valued count and magnitude predictions."""

        shared_features = self.shared_trunk(x)
        return {
            "prob_logits": self.probability_head(shared_features),
            "count_pred": self.count_head(shared_features),
            "magnitude_pred": self.magnitude_head(shared_features),
        }


def build_multitask_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.2,
) -> MultiTaskMLP:
    """Build the version-1 multitask MLP."""

    return MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
