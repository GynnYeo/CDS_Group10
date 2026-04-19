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
from src.models.neural.losses import BinaryFocalLoss
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
        "--count-head-hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional hidden layer sizes for a count-specific tower. "
            "If omitted, the count head remains the original linear head."
        ),
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
    parser.add_argument(
        "--count-modeling-mode",
        type=str,
        default="standard",
        choices=["standard", "conditional_positive"],
        help=(
            "Count modeling mode. 'standard' keeps the existing count loss and "
            "inference behavior. 'conditional_positive' trains count loss only "
            "on positive-count targets and writes probability-weighted positive "
            "severity predictions to the standard count prediction output."
        ),
    )
    parser.add_argument(
        "--probability-loss",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help=(
            "Probability loss to optimize. "
            "'bce' uses BCEWithLogitsLoss. "
            "'focal' uses binary focal loss on logits."
        ),
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help=(
            "Gamma parameter for focal loss. "
            "Only used when --probability-loss focal."
        ),
    )
    parser.add_argument(
        "--probability-head-hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional hidden layer sizes for a probability-specific tower. "
            "If omitted, the probability head remains the original linear head."
        ),
    )

    parser.add_argument(
        "--magnitude-head-hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional hidden layer sizes for a magnitude-specific tower. "
            "If omitted, the magnitude head remains the original linear head."
        ),
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_total_loss",
        choices=[
            "val_total_loss",
            "val_prob_loss",
            "val_count_loss",
            "val_magnitude_loss",
        ],
        help=(
            "Validation metric used to select the best model checkpoint. "
            "'val_total_loss' (default) preserves the original behaviour. "
            "Use 'val_prob_loss' to select the checkpoint that is best for "
            "the probability task specifically (recommended for Task 1). "
            "Use 'val_magnitude_loss' for magnitude-focused runs."
        ),
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
    probability_head_hidden_dims = (
        tuple(args.probability_head_hidden_dims)
        if args.probability_head_hidden_dims
        else None
    )

    count_head_hidden_dims = (
        tuple(args.count_head_hidden_dims)
        if args.count_head_hidden_dims
        else None
    )

    magnitude_head_hidden_dims = (
        tuple(args.magnitude_head_hidden_dims)
        if args.magnitude_head_hidden_dims
        else None
    )
    scale = not args.no_scale
    feature_cols = get_feature_set_by_name(args.feature_set)

    print(f"Dataset: {args.dataset_name}")
    print(f"Feature set: {args.feature_set}")
    print(f"Requested feature count: {len(feature_cols)}")
    print(f"Device: {device}")
    print(f"Scaling enabled: {scale}")
    print(f"Count modeling mode: {args.count_modeling_mode}")
    print(f"Probability loss: {args.probability_loss}")
    if args.probability_loss == "focal":
        print(f"Focal gamma: {args.focal_gamma}")

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
        probability_head_hidden_dims=probability_head_hidden_dims,
        count_head_hidden_dims=count_head_hidden_dims,
        magnitude_head_hidden_dims=magnitude_head_hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.probability_loss == "bce":
        probability_loss_fn = nn.BCEWithLogitsLoss()
    elif args.probability_loss == "focal":
        probability_loss_fn = BinaryFocalLoss(gamma=args.focal_gamma)
    else:
        raise ValueError(
            "probability_loss must be 'bce' or 'focal', "
            f"got {args.probability_loss!r}."
        )
    count_loss_fn = nn.SmoothL1Loss()
    magnitude_loss_fn = nn.SmoothL1Loss()
    training_config = TrainingConfig(
        epochs=args.epochs,
        count_modeling_mode=args.count_modeling_mode,
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
        checkpoint_metric=args.checkpoint_metric,
        # Per-task checkpoints — all four are saved in the same epoch loop
        best_prob_checkpoint_path=run_paths.best_prob_checkpoint_path,
        best_count_checkpoint_path=run_paths.best_count_checkpoint_path,
        best_magnitude_checkpoint_path=run_paths.best_magnitude_checkpoint_path,
        best_total_checkpoint_path=run_paths.best_total_checkpoint_path,
    )

    pd.DataFrame(history).to_csv(run_paths.history_path, index=False)
    print(f"History saved to: {run_paths.history_path}")

    # ── Helper: load a checkpoint, generate predictions + metrics, save CSVs ──
    def _save_task_outputs(
        ckpt_path: Path | None,
        task_label: str,
        prob_out: Path,
        count_out: Path,
        mag_out: Path,
        prob_metrics_out: Path,
        count_metrics_out: Path,
        count_capped_out: Path,
        mag_metrics_out: Path,
    ) -> None:
        """Load one checkpoint and write all prediction + metric CSVs for it."""
        if ckpt_path is None or not ckpt_path.exists():
            print(f"  [{task_label}] checkpoint not found — skipping.")
            return
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        prob_preds, count_preds, mag_preds = collect_prediction_tables(
            model=model,
            prepared=prepared,
            feature_tensors=feature_tensors,
            batch_size=args.batch_size,
            device=device,
            model_name=args.run_name,
            count_modeling_mode=args.count_modeling_mode,
        )
        prob_preds.to_csv(prob_out, index=False)
        count_preds.to_csv(count_out, index=False)
        mag_preds.to_csv(mag_out, index=False)
        compute_probability_metrics(prob_preds).to_csv(prob_metrics_out, index=False)
        compute_count_metrics(count_preds).to_csv(count_metrics_out, index=False)
        compute_capped_count_metrics(count_preds).to_csv(count_capped_out, index=False)
        compute_magnitude_metrics(mag_preds).to_csv(mag_metrics_out, index=False)
        print(f"  [{task_label}] outputs saved  (ckpt epoch={ckpt.get('epoch', '?')})")

    out = run_paths.output_dir
    rn  = run_paths.run_name

    # ── Task 1: probability-best checkpoint ───────────────────────────────────
    print("\n── Task 1 (probability) — loading best_prob checkpoint ──")
    _save_task_outputs(
        ckpt_path=run_paths.best_prob_checkpoint_path,
        task_label="prob",
        prob_out=out / f"{rn}_probability_predictions.csv",
        count_out=out / f"{rn}_count_predictions_from_prob_ckpt.csv",
        mag_out=out / f"{rn}_magnitude_predictions_from_prob_ckpt.csv",
        prob_metrics_out=out / f"{rn}_probability_metrics.csv",
        count_metrics_out=out / f"{rn}_count_metrics_from_prob_ckpt.csv",
        count_capped_out=out / f"{rn}_count_capped_metrics_from_prob_ckpt.csv",
        mag_metrics_out=out / f"{rn}_magnitude_metrics_from_prob_ckpt.csv",
    )

    # ── Task 2: count-best checkpoint ─────────────────────────────────────────
    print("\n── Task 2 (count) — loading best_count checkpoint ──")
    _save_task_outputs(
        ckpt_path=run_paths.best_count_checkpoint_path,
        task_label="count",
        prob_out=out / f"{rn}_probability_predictions_from_count_ckpt.csv",
        count_out=out / f"{rn}_count_predictions.csv",
        mag_out=out / f"{rn}_magnitude_predictions_from_count_ckpt.csv",
        prob_metrics_out=out / f"{rn}_probability_metrics_from_count_ckpt.csv",
        count_metrics_out=out / f"{rn}_count_metrics.csv",
        count_capped_out=out / f"{rn}_count_capped_metrics.csv",
        mag_metrics_out=out / f"{rn}_magnitude_metrics_from_count_ckpt.csv",
    )

    # ── Task 3: magnitude-best checkpoint ────────────────────────────────────
    print("\n── Task 3 (magnitude) — loading best_magnitude checkpoint ──")
    _save_task_outputs(
        ckpt_path=run_paths.best_magnitude_checkpoint_path,
        task_label="magnitude",
        prob_out=out / f"{rn}_probability_predictions_from_mag_ckpt.csv",
        count_out=out / f"{rn}_count_predictions_from_mag_ckpt.csv",
        mag_out=out / f"{rn}_magnitude_predictions.csv",
        prob_metrics_out=out / f"{rn}_probability_metrics_from_mag_ckpt.csv",
        count_metrics_out=out / f"{rn}_count_metrics_from_mag_ckpt.csv",
        count_capped_out=out / f"{rn}_count_capped_metrics_from_mag_ckpt.csv",
        mag_metrics_out=out / f"{rn}_magnitude_metrics.csv",
    )

    # ── Optional: total-loss checkpoint (matches old default behaviour) ───────
    if run_paths.best_total_checkpoint_path is not None:
        print("\n── Total-loss checkpoint (optional) ──")
        _save_task_outputs(
            ckpt_path=run_paths.best_total_checkpoint_path,
            task_label="total",
            prob_out=out / f"{rn}_probability_predictions_from_total_ckpt.csv",
            count_out=out / f"{rn}_count_predictions_from_total_ckpt.csv",
            mag_out=out / f"{rn}_magnitude_predictions_from_total_ckpt.csv",
            prob_metrics_out=out / f"{rn}_probability_metrics_from_total_ckpt.csv",
            count_metrics_out=out / f"{rn}_count_metrics_from_total_ckpt.csv",
            count_capped_out=out / f"{rn}_count_capped_metrics_from_total_ckpt.csv",
            mag_metrics_out=out / f"{rn}_magnitude_metrics_from_total_ckpt.csv",
        )

    total_seconds = time.perf_counter() - run_start
    print(f"\nTotal epochs        : {args.epochs}")
    print(f"Best epoch (legacy) : {best_epoch}")
    print(f"Best val loss       : {best_val_loss:.4f}")
    print(f"Total training time : {total_seconds:.2f}s")
    print(f"Best prob ckpt      : {run_paths.best_prob_checkpoint_path}")
    print(f"Best count ckpt     : {run_paths.best_count_checkpoint_path}")
    print(f"Best magnitude ckpt : {run_paths.best_magnitude_checkpoint_path}")
    print(f"Last checkpoint     : {run_paths.last_checkpoint_path}")



if __name__ == "__main__":
    main()
