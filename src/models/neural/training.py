from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.neural.losses import compute_loss_components


@dataclass(slots=True)
class TrainingConfig:
    """
    Configuration for the multitask MLP training loop.

    ``count_modeling_mode`` changes which rows contribute to the count loss, so
    logged count-loss values should only be compared within the same mode.
    """

    epochs: int
    count_modeling_mode: str = "standard"
    count_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    gradient_clip_max_norm: float = 1.0


def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    magnitude_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    count_modeling_mode: str,
    count_loss_weight: float,
    magnitude_loss_weight: float,
    gradient_clip_max_norm: float = 1.0,
) -> dict[str, float]:
    """
    Run one training epoch and return averaged multitask losses.

    The count-loss average reflects the selected ``count_modeling_mode`` and is
    therefore not directly comparable across different count modes.
    """

    model.train()
    total_examples = 0
    running_total = 0.0
    running_prob = 0.0
    running_count = 0.0
    running_magnitude = 0.0

    progress = tqdm(
        data_loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False,
    )
    for (
        features,
        probability_targets,
        count_targets,
        magnitude_targets,
        magnitude_masks,
    ) in progress:
        features = features.to(device)
        probability_targets = probability_targets.to(device)
        count_targets = count_targets.to(device)
        magnitude_targets = magnitude_targets.to(device)
        magnitude_masks = magnitude_masks.to(device)

        optimizer.zero_grad()
        total_loss, prob_loss, count_loss, magnitude_loss = compute_loss_components(
            model=model,
            features=features,
            probability_targets=probability_targets,
            count_targets=count_targets,
            magnitude_targets=magnitude_targets,
            magnitude_masks=magnitude_masks,
            probability_loss_fn=probability_loss_fn,
            count_loss_fn=count_loss_fn,
            magnitude_loss_fn=magnitude_loss_fn,
            count_loss_weight=count_loss_weight,
            magnitude_loss_weight=magnitude_loss_weight,
            count_modeling_mode=count_modeling_mode,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
        optimizer.step()

        batch_size = features.size(0)
        total_examples += batch_size
        running_total += float(total_loss.item()) * batch_size
        running_prob += float(prob_loss.item()) * batch_size
        running_count += float(count_loss.item()) * batch_size
        running_magnitude += float(magnitude_loss.item()) * batch_size

        progress.set_postfix(loss=f"{(running_total / total_examples):.4f}")

    return {
        "train_total_loss": running_total / total_examples,
        "train_prob_loss": running_prob / total_examples,
        "train_count_loss": running_count / total_examples,
        "train_magnitude_loss": running_magnitude / total_examples,
    }


def run_validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    magnitude_loss_fn: nn.Module,
    device: torch.device,
    count_modeling_mode: str,
    count_loss_weight: float,
    magnitude_loss_weight: float,
) -> dict[str, float]:
    """
    Run one validation epoch and return averaged multitask losses.

    The validation count loss uses the active count-modeling mode, so its numeric
    value should only be compared against runs that used the same mode.
    """

    model.eval()
    total_examples = 0
    running_total = 0.0
    running_prob = 0.0
    running_count = 0.0
    running_magnitude = 0.0

    with torch.no_grad():
        for (
            features,
            probability_targets,
            count_targets,
            magnitude_targets,
            magnitude_masks,
        ) in data_loader:
            features = features.to(device)
            probability_targets = probability_targets.to(device)
            count_targets = count_targets.to(device)
            magnitude_targets = magnitude_targets.to(device)
            magnitude_masks = magnitude_masks.to(device)

            total_loss, prob_loss, count_loss, magnitude_loss = compute_loss_components(
                model=model,
                features=features,
                probability_targets=probability_targets,
                count_targets=count_targets,
                magnitude_targets=magnitude_targets,
                magnitude_masks=magnitude_masks,
                probability_loss_fn=probability_loss_fn,
                count_loss_fn=count_loss_fn,
                magnitude_loss_fn=magnitude_loss_fn,
                count_loss_weight=count_loss_weight,
                magnitude_loss_weight=magnitude_loss_weight,
                count_modeling_mode=count_modeling_mode,
            )

            batch_size = features.size(0)
            total_examples += batch_size
            running_total += float(total_loss.item()) * batch_size
            running_prob += float(prob_loss.item()) * batch_size
            running_count += float(count_loss.item()) * batch_size
            running_magnitude += float(magnitude_loss.item()) * batch_size

    return {
        "val_total_loss": running_total / total_examples,
        "val_prob_loss": running_prob / total_examples,
        "val_count_loss": running_count / total_examples,
        "val_magnitude_loss": running_magnitude / total_examples,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    feature_cols: list[str],
    args_metadata: dict[str, Any],
    training_config: TrainingConfig,
) -> None:
    """Save a training checkpoint compatible with downstream torch.load usage."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "args": dict(args_metadata),
        "training_config": asdict(training_config),
        "feature_cols": list(feature_cols),
    }
    torch.save(checkpoint, checkpoint_path)


CHECKPOINT_METRIC_OPTIONS = {
    "val_total_loss",
    "val_prob_loss",
    "val_count_loss",
    "val_magnitude_loss",
}

# Maps each task metric to a human-readable label for logging
_METRIC_LABELS = {
    "val_prob_loss":      "prob",
    "val_count_loss":     "count",
    "val_magnitude_loss": "magnitude",
    "val_total_loss":     "total",
}


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    magnitude_loss_fn: nn.Module,
    device: torch.device,
    training_config: TrainingConfig,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    args_metadata: dict[str, Any],
    feature_cols: list[str],
    checkpoint_metric: str = "val_total_loss",
    # Per-task checkpoint paths — if provided, each task gets its own best checkpoint
    best_prob_checkpoint_path: Path | None = None,
    best_count_checkpoint_path: Path | None = None,
    best_magnitude_checkpoint_path: Path | None = None,
    best_total_checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, float | int]], float, int]:
    """Train the model and save per-task best checkpoints in ONE training run.

    Four checkpoints can be saved simultaneously during the same epoch loop:

    * ``best_prob_checkpoint_path``      — epoch with lowest ``val_prob_loss``
    * ``best_count_checkpoint_path``     — epoch with lowest ``val_count_loss``
    * ``best_magnitude_checkpoint_path`` — epoch with lowest ``val_magnitude_loss``
    * ``best_total_checkpoint_path``     — epoch with lowest ``val_total_loss``

    The legacy ``best_checkpoint_path`` is still saved using ``checkpoint_metric``
    (default ``val_total_loss``) for full backward compatibility.

    Only paths that are not ``None`` are written to disk.
    """

    if checkpoint_metric not in CHECKPOINT_METRIC_OPTIONS:
        raise ValueError(
            f"checkpoint_metric must be one of {sorted(CHECKPOINT_METRIC_OPTIONS)}, "
            f"got {checkpoint_metric!r}."
        )

    history: list[dict[str, float | int]] = []

    # Track best value and best epoch independently for each task
    task_trackers: dict[str, dict] = {
        "val_prob_loss": {
            "best": float("inf"),
            "best_epoch": 0,
            "path": best_prob_checkpoint_path,
            "label": "prob",
        },
        "val_count_loss": {
            "best": float("inf"),
            "best_epoch": 0,
            "path": best_count_checkpoint_path,
            "label": "count",
        },
        "val_magnitude_loss": {
            "best": float("inf"),
            "best_epoch": 0,
            "path": best_magnitude_checkpoint_path,
            "label": "magnitude",
        },
        "val_total_loss": {
            "best": float("inf"),
            "best_epoch": 0,
            "path": best_total_checkpoint_path,
            "label": "total",
        },
    }

    # Legacy single best checkpoint (uses checkpoint_metric)
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"Legacy checkpoint metric  : {checkpoint_metric}")
    print(f"Per-task checkpoints saved: "
          + ", ".join(
              m for m, t in task_trackers.items() if t["path"] is not None
          ))

    for epoch in range(1, training_config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_training_epoch(
            model=model,
            data_loader=loaders["train"],
            optimizer=optimizer,
            probability_loss_fn=probability_loss_fn,
            count_loss_fn=count_loss_fn,
            magnitude_loss_fn=magnitude_loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=training_config.epochs,
            count_modeling_mode=training_config.count_modeling_mode,
            count_loss_weight=training_config.count_loss_weight,
            magnitude_loss_weight=training_config.magnitude_loss_weight,
            gradient_clip_max_norm=training_config.gradient_clip_max_norm,
        )
        val_metrics = run_validation_epoch(
            model=model,
            data_loader=loaders["val"],
            probability_loss_fn=probability_loss_fn,
            count_loss_fn=count_loss_fn,
            magnitude_loss_fn=magnitude_loss_fn,
            device=device,
            count_modeling_mode=training_config.count_modeling_mode,
            count_loss_weight=training_config.count_loss_weight,
            magnitude_loss_weight=training_config.magnitude_loss_weight,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_record)

        # ── Save per-task checkpoints ─────────────────────────────────────
        for metric_key, tracker in task_trackers.items():
            if tracker["path"] is None:
                continue  # path not configured — skip
            current = val_metrics[metric_key]
            if current < tracker["best"]:
                tracker["best"] = current
                tracker["best_epoch"] = epoch
                save_checkpoint(
                    checkpoint_path=tracker["path"],
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_loss=current,
                    feature_cols=feature_cols,
                    args_metadata=args_metadata,
                    training_config=training_config,
                )

        # ── Legacy single best checkpoint ─────────────────────────────────
        if val_metrics[checkpoint_metric] < best_val_loss:
            best_val_loss = val_metrics[checkpoint_metric]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=best_val_loss,
                feature_cols=feature_cols,
                args_metadata=args_metadata,
                training_config=training_config,
            )

        print(
            f"Epoch {epoch}/{training_config.epochs} | "
            f"train_total={train_metrics['train_total_loss']:.4f} "
            f"train_prob={train_metrics['train_prob_loss']:.4f} "
            f"train_count={train_metrics['train_count_loss']:.4f} | "
            f"val_total={val_metrics['val_total_loss']:.4f} "
            f"val_prob={val_metrics['val_prob_loss']:.4f} "
            f"val_count={val_metrics['val_count_loss']:.4f} | "
            f"train_mag={train_metrics['train_magnitude_loss']:.4f} "
            f"val_mag={val_metrics['val_magnitude_loss']:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

    final_val_loss = float(history[-1]["val_total_loss"])
    save_checkpoint(
        checkpoint_path=last_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=training_config.epochs,
        validation_loss=final_val_loss,
        feature_cols=feature_cols,
        args_metadata=args_metadata,
        training_config=training_config,
    )

    # Log where each per-task best was found
    for metric_key, tracker in task_trackers.items():
        if tracker["path"] is not None:
            print(
                f"Best {tracker['label']:10s} checkpoint : "
                f"epoch {tracker['best_epoch']:3d}  "
                f"({metric_key}={tracker['best']:.4f})"
            )

    return history, best_val_loss, best_epoch
