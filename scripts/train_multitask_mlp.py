"""Train and evaluate the first multitask PyTorch MLP on processed splits."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    evaluate_binary_probabilities,
    evaluate_count_predictions,
)
from src.evaluation.validation import (
    validate_binary_prediction_columns,
    validate_prediction_frame,
)
from src.models.input_layer import EXPECTED_SPLITS
from src.models.neural.data import (
    COUNT_TARGET_COLUMNS,
    PROBABILITY_TARGET_COLUMNS,
    PreparedNeuralInputs,
    get_feature_set_by_name,
    prepare_multitask_neural_inputs,
)
from src.models.neural.model import build_multitask_mlp


HORIZONS = (24, 72)
PROBABILITY_TARGETS_BY_HORIZON = list(zip(HORIZONS, PROBABILITY_TARGET_COLUMNS))
COUNT_TARGETS_BY_HORIZON = list(zip(HORIZONS, COUNT_TARGET_COLUMNS))


class MultiTaskTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Minimal dataset returning features plus both target groups."""

    def __init__(
        self,
        features: torch.Tensor,
        probability_targets: torch.Tensor,
        count_targets: torch.Tensor,
    ) -> None:
        if not (
            len(features) == len(probability_targets) == len(count_targets)
        ):
            raise ValueError("Features and targets must have matching lengths.")

        self.features = features
        self.probability_targets = probability_targets
        self.count_targets = count_targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.probability_targets[index],
            self.count_targets[index],
        )


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
        help="Directory for CSV metrics, predictions, and history outputs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="reports/checkpoints",
        help="Directory for best/last model checkpoints.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    """Resolve a possibly relative path under the repository root."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested device, supporting `auto`."""

    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(normalized)


def dataframe_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Convert a pandas DataFrame to a float32 tensor."""

    return torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)


def counts_to_log_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Apply log1p to count targets and convert to float32 tensor."""

    count_values = df.to_numpy(dtype=np.float32)
    if np.any(count_values < 0):
        raise ValueError("Count targets must be non-negative before log1p transform.")
    return torch.tensor(np.log1p(count_values), dtype=torch.float32)


def build_data_loaders(
    prepared: PreparedNeuralInputs,
    batch_size: int,
) -> tuple[dict[str, DataLoader], dict[str, torch.Tensor]]:
    """Construct split-wise dataloaders and cached feature tensors."""

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

    loaders = {
        split_name: DataLoader(
            MultiTaskTensorDataset(
                features=feature_tensors[split_name],
                probability_targets=probability_tensors[split_name],
                count_targets=count_tensors[split_name],
            ),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )
        for split_name in EXPECTED_SPLITS
    }
    return loaders, feature_tensors


def compute_loss_components(
    model: nn.Module,
    features: torch.Tensor,
    probability_targets: torch.Tensor,
    count_targets: torch.Tensor,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total, probability, and count losses for one batch."""

    outputs = model(features)
    prob_loss = probability_loss_fn(outputs["prob_logits"], probability_targets)
    count_loss = count_loss_fn(outputs["count_pred"], count_targets)
    total_loss = prob_loss + count_loss
    return total_loss, prob_loss, count_loss


def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """Run one training epoch and return averaged losses."""

    model.train()
    total_examples = 0
    running_total = 0.0
    running_prob = 0.0
    running_count = 0.0

    progress = tqdm(
        data_loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False,
    )
    for features, probability_targets, count_targets in progress:
        features = features.to(device)
        probability_targets = probability_targets.to(device)
        count_targets = count_targets.to(device)

        optimizer.zero_grad()
        total_loss, prob_loss, count_loss = compute_loss_components(
            model=model,
            features=features,
            probability_targets=probability_targets,
            count_targets=count_targets,
            probability_loss_fn=probability_loss_fn,
            count_loss_fn=count_loss_fn,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = features.size(0)
        total_examples += batch_size
        running_total += float(total_loss.item()) * batch_size
        running_prob += float(prob_loss.item()) * batch_size
        running_count += float(count_loss.item()) * batch_size

        progress.set_postfix(loss=f"{(running_total / total_examples):.4f}")

    return {
        "train_total_loss": running_total / total_examples,
        "train_prob_loss": running_prob / total_examples,
        "train_count_loss": running_count / total_examples,
    }


def run_validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch and return averaged losses."""

    model.eval()
    total_examples = 0
    running_total = 0.0
    running_prob = 0.0
    running_count = 0.0

    with torch.no_grad():
        for features, probability_targets, count_targets in data_loader:
            features = features.to(device)
            probability_targets = probability_targets.to(device)
            count_targets = count_targets.to(device)

            total_loss, prob_loss, count_loss = compute_loss_components(
                model=model,
                features=features,
                probability_targets=probability_targets,
                count_targets=count_targets,
                probability_loss_fn=probability_loss_fn,
                count_loss_fn=count_loss_fn,
            )

            batch_size = features.size(0)
            total_examples += batch_size
            running_total += float(total_loss.item()) * batch_size
            running_prob += float(prob_loss.item()) * batch_size
            running_count += float(count_loss.item()) * batch_size

    return {
        "val_total_loss": running_total / total_examples,
        "val_prob_loss": running_prob / total_examples,
        "val_count_loss": running_count / total_examples,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    args: argparse.Namespace,
    feature_cols: list[str],
) -> None:
    """Save a training checkpoint."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "args": vars(args),
        "feature_cols": list(feature_cols),
    }
    torch.save(checkpoint, checkpoint_path)


def predict_split_outputs(
    model: nn.Module,
    features_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run split-level inference and return probability and count predictions."""

    model.eval()
    feature_loader = DataLoader(
        TensorDataset(features_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    probability_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for (features_batch,) in feature_loader:
            features_batch = features_batch.to(device)
            outputs = model(features_batch)
            probability_chunks.append(
                torch.sigmoid(outputs["prob_logits"]).cpu().numpy()
            )
            count_chunks.append(outputs["count_pred"].cpu().numpy())

    probability_predictions = np.concatenate(probability_chunks, axis=0)
    count_log_predictions = np.concatenate(count_chunks, axis=0)
    count_predictions = np.expm1(count_log_predictions)
    count_predictions = np.clip(count_predictions, a_min=0.0, a_max=None)
    return probability_predictions, count_predictions


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
        required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_prob"],
        split_col="split",
    )
    for (_, horizon), group_df in prediction_df.groupby(["split", "horizon"], sort=False):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_prob"],
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
        required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_pred"],
        split_col="split",
    )
    for (_, horizon), group_df in prediction_df.groupby(["split", "horizon"], sort=False):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_pred"],
            id_col="trigger_event_id",
            split_col="split",
        )
    return prediction_df


def collect_prediction_tables(
    model: nn.Module,
    prepared: PreparedNeuralInputs,
    feature_tensors: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate standardized prediction tables for all splits."""

    probability_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []

    split_metadata = {
        "train": (
            prepared.train_ids,
            prepared.train_splits,
            prepared.y_prob_train,
            prepared.y_count_train,
        ),
        "val": (
            prepared.val_ids,
            prepared.val_splits,
            prepared.y_prob_val,
            prepared.y_count_val,
        ),
        "test": (
            prepared.test_ids,
            prepared.test_splits,
            prepared.y_prob_test,
            prepared.y_count_test,
        ),
    }

    for split_name in EXPECTED_SPLITS:
        probability_predictions, count_predictions = predict_split_outputs(
            model=model,
            features_tensor=feature_tensors[split_name],
            batch_size=batch_size,
            device=device,
        )
        ids, split_labels, probability_targets, count_targets = split_metadata[split_name]
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

    probability_df = pd.concat(probability_frames, ignore_index=True)
    count_df = pd.concat(count_frames, ignore_index=True)
    return probability_df, count_df


def sort_metric_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Sort metrics by model, horizon, and project split order."""

    if metrics_df.empty:
        return metrics_df

    split_order = {split_name: index for index, split_name in enumerate(EXPECTED_SPLITS)}
    return (
        metrics_df.assign(_split_order=metrics_df["split"].map(split_order))
        .sort_values(by=["model_name", "horizon", "_split_order"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def compute_probability_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped probability metrics by split and horizon."""

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]
    for (model_name, split_name, horizon), group_df in prediction_df.groupby(group_columns, sort=True):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_prob"],
            id_col="trigger_event_id",
            split_col="split",
        )
        validate_binary_prediction_columns(
            prediction_df=group_df,
            y_true_col="y_true",
            y_prob_col="y_prob",
        )
        metrics = evaluate_binary_probabilities(group_df["y_true"], group_df["y_prob"])
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                **metrics,
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))


def compute_count_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped count metrics by split and horizon."""

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["model_name", "split", "horizon"]
    for (model_name, split_name, horizon), group_df in prediction_df.groupby(group_columns, sort=True):
        validate_prediction_frame(
            prediction_df=group_df,
            required_columns=["trigger_event_id", "split", "horizon", "model_name", "y_true", "y_pred"],
            id_col="trigger_event_id",
            split_col="split",
        )
        metrics = evaluate_count_predictions(group_df["y_true"], group_df["y_pred"])
        summary_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "horizon": int(horizon),
                **metrics,
            }
        )

    return sort_metric_rows(pd.DataFrame(summary_rows))


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    probability_loss_fn: nn.Module,
    count_loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    args: argparse.Namespace,
    feature_cols: list[str],
) -> tuple[list[dict[str, float | int]], float, int]:
    """Train the model, save checkpoints, and return history plus best stats."""

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_training_epoch(
            model=model,
            data_loader=loaders["train"],
            optimizer=optimizer,
            probability_loss_fn=probability_loss_fn,
            count_loss_fn=count_loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        val_metrics = run_validation_epoch(
            model=model,
            data_loader=loaders["val"],
            probability_loss_fn=probability_loss_fn,
            count_loss_fn=count_loss_fn,
            device=device,
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
                args=args,
                feature_cols=feature_cols,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_total={train_metrics['train_total_loss']:.4f} "
            f"train_prob={train_metrics['train_prob_loss']:.4f} "
            f"train_count={train_metrics['train_count_loss']:.4f} | "
            f"val_total={val_metrics['val_total_loss']:.4f} "
            f"val_prob={val_metrics['val_prob_loss']:.4f} "
            f"val_count={val_metrics['val_count_loss']:.4f} | "
            f"time={epoch_seconds:.2f}s"
        )

    final_val_loss = float(history[-1]["val_total_loss"])
    save_checkpoint(
        checkpoint_path=last_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        validation_loss=final_val_loss,
        args=args,
        feature_cols=feature_cols,
    )
    return history, best_val_loss, best_epoch


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    best_checkpoint_path = checkpoint_dir / f"{args.run_name}_best.pt"
    last_checkpoint_path = checkpoint_dir / f"{args.run_name}_last.pt"

    history, best_val_loss, best_epoch = train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        probability_loss_fn=probability_loss_fn,
        count_loss_fn=count_loss_fn,
        device=device,
        epochs=args.epochs,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        args=args,
        feature_cols=prepared.feature_cols,
    )

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    probability_predictions, count_predictions = collect_prediction_tables(
        model=model,
        prepared=prepared,
        feature_tensors=feature_tensors,
        batch_size=args.batch_size,
        device=device,
        model_name=args.run_name,
    )
    probability_metrics = compute_probability_metrics(probability_predictions)
    count_metrics = compute_count_metrics(count_predictions)

    history_df = pd.DataFrame(history)
    probability_predictions_path = output_dir / f"{args.run_name}_probability_predictions.csv"
    count_predictions_path = output_dir / f"{args.run_name}_count_predictions.csv"
    probability_metrics_path = output_dir / f"{args.run_name}_probability_metrics.csv"
    count_metrics_path = output_dir / f"{args.run_name}_count_metrics.csv"
    history_path = output_dir / f"{args.run_name}_history.csv"

    probability_predictions.to_csv(probability_predictions_path, index=False)
    count_predictions.to_csv(count_predictions_path, index=False)
    probability_metrics.to_csv(probability_metrics_path, index=False)
    count_metrics.to_csv(count_metrics_path, index=False)
    history_df.to_csv(history_path, index=False)

    total_seconds = time.perf_counter() - run_start
    print(f"Total epochs: {args.epochs}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_seconds:.2f}s")
    print(f"History saved to: {history_path}")
    print(f"Probability predictions saved to: {probability_predictions_path}")
    print(f"Count predictions saved to: {count_predictions_path}")
    print(f"Probability metrics saved to: {probability_metrics_path}")
    print(f"Count metrics saved to: {count_metrics_path}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Last checkpoint saved to: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
