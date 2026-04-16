"""Train and evaluate the multitask PyTorch MLP on processed splits."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.neural_metrics import (
    compute_capped_count_metrics,
    compute_count_metrics,
    compute_magnitude_metrics,
    compute_probability_metrics,
)
from src.models.neural.artifacts import build_run_artifact_paths
from src.models.neural.data import (
    get_feature_set_by_name,
    prepare_multitask_neural_inputs,
)
from src.models.neural.model import build_multitask_mlp
from src.models.neural.prediction import collect_prediction_tables
from src.models.neural.tensors import build_data_loaders
from src.models.neural.training import TrainingConfig, train_model
from src.models.neural.utils import resolve_device, resolve_repo_path, set_random_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(
        description="Train the version-1 multitask MLP on processed aftershock splits."
    )
    parser.add_argument(
        "--dataset-name",
        default="earthquake_aftershock_v2_gcmt",
        help="Processed dataset name to load.",
    )
    parser.add_argument(
        "--feature-set",
        default="nn_enriched_v1",
        help="Named neural feature set to use.",
    )
    parser.add_argument(
        "--run-name",
        default="mlp_multitask_v1",
        help="Prefix for saved outputs and checkpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for training and inference.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate in the shared trunk.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer sizes for the shared MLP trunk.",
    )
    parser.add_argument(
        "--missing-strategy",
        default="median",
        help="Missing-value strategy passed to the modeling input layer.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable train-only feature scaling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/metrics",
        help="Base directory for run-scoped CSV metrics, predictions, and history outputs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="reports/checkpoints",
        help="Base directory for run-scoped best/last model checkpoints.",
    )
    parser.add_argument(
        "--count-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to the count loss in the multitask objective.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    device = resolve_device(args.device)
    run_paths = build_run_artifact_paths(
        run_name=args.run_name,
        base_output_dir=resolve_repo_path(args.output_dir),
        base_checkpoint_dir=resolve_repo_path(args.checkpoint_dir),
    )

    hidden_dims = tuple(args.hidden_dims)
    scale = not args.no_scale
    feature_cols = get_feature_set_by_name(args.feature_set)

    print(f"Dataset: {args.dataset_name}")
    print(f"Feature set: {args.feature_set}")
    print(f"Requested feature count: {len(feature_cols)}")
    print(f"Device: {device}")
    print(f"Scaling enabled: {scale}")

    run_start = time.perf_counter()
    prepared = prepare_multitask_neural_inputs(
        feature_cols=feature_cols,
        dataset_name=args.dataset_name,
        missing_strategy=args.missing_strategy,
        scale=scale,
        allow_missing_optional=True,
    )
    print(f"Resolved feature count: {len(prepared.feature_cols)}")

    loaders, feature_tensors = build_data_loaders(
        prepared=prepared,
        batch_size=args.batch_size,
    )

    model = build_multitask_mlp(
        input_dim=len(prepared.feature_cols),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    probability_loss_fn = nn.BCEWithLogitsLoss()
    count_loss_fn = nn.SmoothL1Loss()
    magnitude_loss_fn = nn.SmoothL1Loss()
    training_config = TrainingConfig(
        epochs=args.epochs,
        count_loss_weight=args.count_loss_weight,
        magnitude_loss_weight=1.0,
        gradient_clip_max_norm=1.0,
    )

    if run_paths.best_checkpoint_path is None or run_paths.last_checkpoint_path is None:
        raise ValueError("Training requires checkpoint paths to be configured.")

    history, best_val_loss, best_epoch = train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        probability_loss_fn=probability_loss_fn,
        count_loss_fn=count_loss_fn,
        magnitude_loss_fn=magnitude_loss_fn,
        device=device,
        training_config=training_config,
        best_checkpoint_path=run_paths.best_checkpoint_path,
        last_checkpoint_path=run_paths.last_checkpoint_path,
        args_metadata=vars(args),
        feature_cols=prepared.feature_cols,
    )

    best_checkpoint = torch.load(run_paths.best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    probability_predictions, count_predictions, magnitude_predictions = (
        collect_prediction_tables(
            model=model,
            prepared=prepared,
            feature_tensors=feature_tensors,
            batch_size=args.batch_size,
            device=device,
            model_name=args.run_name,
        )
    )
    probability_metrics = compute_probability_metrics(probability_predictions)
    count_metrics = compute_count_metrics(count_predictions)
    count_capped_metrics = compute_capped_count_metrics(count_predictions)
    magnitude_metrics = compute_magnitude_metrics(magnitude_predictions)

    pd.DataFrame(history).to_csv(run_paths.history_path, index=False)
    probability_predictions.to_csv(run_paths.probability_predictions_path, index=False)
    count_predictions.to_csv(run_paths.count_predictions_path, index=False)
    magnitude_predictions.to_csv(run_paths.magnitude_predictions_path, index=False)
    probability_metrics.to_csv(run_paths.probability_metrics_path, index=False)
    count_metrics.to_csv(run_paths.count_metrics_path, index=False)
    count_capped_metrics.to_csv(run_paths.count_capped_metrics_path, index=False)
    magnitude_metrics.to_csv(run_paths.magnitude_metrics_path, index=False)

    total_seconds = time.perf_counter() - run_start
    print(f"Total epochs: {args.epochs}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_seconds:.2f}s")
    print(f"History saved to: {run_paths.history_path}")
    print(f"Probability predictions saved to: {run_paths.probability_predictions_path}")
    print(f"Count predictions saved to: {run_paths.count_predictions_path}")
    print(f"Count metrics saved to: {run_paths.count_metrics_path}")
    print(f"Capped Count metrics saved to: {run_paths.count_capped_metrics_path}")
    print(f"Magnitude predictions saved to: {run_paths.magnitude_predictions_path}")
    print(f"Magnitude metrics saved to: {run_paths.magnitude_metrics_path}")
    print(f"Probability metrics saved to: {run_paths.probability_metrics_path}")
    print(f"Best checkpoint saved to: {run_paths.best_checkpoint_path}")
    print(f"Last checkpoint saved to: {run_paths.last_checkpoint_path}")


if __name__ == "__main__":
    main()
