from __future__ import annotations

import torch
import torch.nn as nn


def compute_masked_magnitude_loss(
    magnitude_predictions: torch.Tensor,
    magnitude_targets: torch.Tensor,
    magnitude_masks: torch.Tensor,
    magnitude_loss_fn: nn.Module,
) -> torch.Tensor:
    """Compute magnitude loss only where per-horizon magnitude targets exist."""

    per_horizon_losses: list[torch.Tensor] = []

    for horizon_index in range(magnitude_predictions.shape[1]):
        horizon_mask = magnitude_masks[:, horizon_index]
        if horizon_mask.any():
            per_horizon_losses.append(
                magnitude_loss_fn(
                    magnitude_predictions[horizon_mask, horizon_index],
                    magnitude_targets[horizon_mask, horizon_index],
                )
            )

    if not per_horizon_losses:
        # Return a zero-valued tensor connected to the graph so batches with no
        # available magnitude targets remain safe during backpropagation.
        return magnitude_predictions.sum() * 0.0

    return torch.stack(per_horizon_losses).mean()


def compute_loss_components(
    model: nn.Module,
    features: torch.Tensor,
    probability_targets: torch.Tensor,
    count_targets: torch.Tensor,
    magnitude_targets: torch.Tensor,
    magnitude_masks: torch.Tensor,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    magnitude_loss_fn: nn.Module,
    count_loss_weight: float,
    magnitude_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total, probability, count, and masked magnitude losses for a batch.
    """

    outputs = model(features)

    prob_loss = probability_loss_fn(outputs["prob_logits"], probability_targets)
    count_loss = count_loss_fn(outputs["count_pred"], count_targets)
    magnitude_loss = compute_masked_magnitude_loss(
        magnitude_predictions=outputs["magnitude_pred"],
        magnitude_targets=magnitude_targets,
        magnitude_masks=magnitude_masks,
        magnitude_loss_fn=magnitude_loss_fn,
    )

    total_loss = (
        prob_loss
        + count_loss_weight * count_loss
        + magnitude_loss_weight * magnitude_loss
    )
    return total_loss, prob_loss, count_loss, magnitude_loss
