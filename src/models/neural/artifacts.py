from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunArtifactPaths:
    """Standardized output and checkpoint paths for one run name."""

    run_name: str
    output_dir: Path
    checkpoint_dir: Path | None
    probability_predictions_path: Path
    count_predictions_path: Path
    magnitude_predictions_path: Path
    probability_metrics_path: Path
    count_metrics_path: Path
    count_capped_metrics_path: Path
    magnitude_metrics_path: Path
    history_path: Path
    # Original single best/last checkpoints (kept for back-compat)
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    # Per-task best checkpoints saved during ONE training run
    best_prob_checkpoint_path: Path | None       # best val_prob_loss  → Task 1
    best_count_checkpoint_path: Path | None      # best val_count_loss → Task 2
    best_magnitude_checkpoint_path: Path | None  # best val_magnitude_loss → Task 3
    best_total_checkpoint_path: Path | None      # best val_total_loss (optional)
    # Per-task prediction / metric outputs (written after loading each checkpoint)
    prob_predictions_path: Path
    count_predictions_from_prob_ckpt_path: Path  # not used — count uses count ckpt
    magnitude_predictions_path_from_mag_ckpt: Path


def build_run_artifact_paths(
    run_name: str,
    base_output_dir: Path,
    base_checkpoint_dir: Path | None = None,
    create_dirs: bool = True,
) -> RunArtifactPaths:
    """Build the run-scoped output and checkpoint paths used by scripts."""

    output_dir = base_output_dir / run_name
    checkpoint_dir = None if base_checkpoint_dir is None else base_checkpoint_dir / run_name

    if create_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt(suffix: str) -> Path | None:
        if checkpoint_dir is None:
            return None
        return checkpoint_dir / f"{run_name}_{suffix}.pt"

    return RunArtifactPaths(
        run_name=run_name,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        probability_predictions_path=output_dir / f"{run_name}_probability_predictions.csv",
        count_predictions_path=output_dir / f"{run_name}_count_predictions.csv",
        magnitude_predictions_path=output_dir / f"{run_name}_magnitude_predictions.csv",
        probability_metrics_path=output_dir / f"{run_name}_probability_metrics.csv",
        count_metrics_path=output_dir / f"{run_name}_count_metrics.csv",
        count_capped_metrics_path=output_dir / f"{run_name}_count_capped_metrics.csv",
        magnitude_metrics_path=output_dir / f"{run_name}_magnitude_metrics.csv",
        history_path=output_dir / f"{run_name}_history.csv",
        # Legacy single-checkpoint paths (kept so existing code still works)
        best_checkpoint_path=_ckpt("best"),
        last_checkpoint_path=_ckpt("last"),
        # Per-task checkpoints
        best_prob_checkpoint_path=_ckpt("best_prob"),
        best_count_checkpoint_path=_ckpt("best_count"),
        best_magnitude_checkpoint_path=_ckpt("best_magnitude"),
        best_total_checkpoint_path=_ckpt("best_total"),
        # These paths are convenience aliases used by the training script
        prob_predictions_path=output_dir / f"{run_name}_probability_predictions.csv",
        count_predictions_from_prob_ckpt_path=output_dir / f"{run_name}_count_predictions.csv",
        magnitude_predictions_path_from_mag_ckpt=output_dir / f"{run_name}_magnitude_predictions.csv",
    )
