from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_positive_only_count_loss(
    count_predictions: torch.Tensor,
    count_targets: torch.Tensor,
    count_loss_fn: nn.Module,
) -> torch.Tensor:
    """
    Compute count loss only on rows with positive per-horizon count targets.

    Count targets are already stored as ``log1p(count)``, so zero-count rows stay
    exactly at ``0`` and positive raw counts remain greater than ``0``.
    """

    per_horizon_losses: list[torch.Tensor] = []

    for horizon_index in range(count_predictions.shape[1]):
        positive_mask = count_targets[:, horizon_index] > 0
        if positive_mask.any():
            per_horizon_losses.append(
                count_loss_fn(
                    count_predictions[positive_mask, horizon_index],
                    count_targets[positive_mask, horizon_index],
                )
            )

    if not per_horizon_losses:
        # Keep the zero safely attached to the graph when an entire batch has
        # no positive count targets across all horizons.
        return count_predictions.sum() * 0.0

    return torch.stack(per_horizon_losses).mean()


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
    count_modeling_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total, probability, count, and masked magnitude losses for a batch.

    The count-loss semantics depend on ``count_modeling_mode``:

    * ``standard`` uses all rows
    * ``conditional_positive`` uses only positive-count rows per horizon

    Because the supervised population differs, the returned count loss should be
    compared only within the same count-modeling mode.
    """

    outputs = model(features)

    prob_loss = probability_loss_fn(outputs["prob_logits"], probability_targets)
    if count_modeling_mode == "standard":
        count_loss = count_loss_fn(outputs["count_pred"], count_targets)
    elif count_modeling_mode == "conditional_positive":
        count_loss = compute_positive_only_count_loss(
            count_predictions=outputs["count_pred"],
            count_targets=count_targets,
            count_loss_fn=count_loss_fn,
        )
    else:
        raise ValueError(
            "count_modeling_mode must be 'standard' or 'conditional_positive', "
            f"got {count_modeling_mode!r}."
        )
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

class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for multi-label logits.

    This wraps binary cross entropy with logits and applies a focal weighting
    term so the model focuses more on harder examples.

    Targets are expected to be float tensors in {0, 1} with the same shape as
    logits, e.g. [batch_size, num_horizons].
    """

    def __init__(self, gamma: float = 1.5, reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}."
            )
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss



