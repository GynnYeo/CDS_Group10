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
    """Configuration for the multitask MLP training loop."""

    epochs: int
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
    count_loss_weight: float,
    magnitude_loss_weight: float,
    gradient_clip_max_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch and return averaged multitask losses."""

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
    count_loss_weight: float,
    magnitude_loss_weight: float,
) -> dict[str, float]:
    """Run one validation epoch and return averaged multitask losses."""

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
) -> tuple[list[dict[str, float | int]], float, int]:
    """Train the model, save best/last checkpoints, and return history."""

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0

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

        if val_metrics["val_total_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_total_loss"]
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
    return history, best_val_loss, best_epoch
