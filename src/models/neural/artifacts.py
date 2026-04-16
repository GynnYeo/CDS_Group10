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
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None


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

    best_checkpoint_path = None
    last_checkpoint_path = None
    if checkpoint_dir is not None:
        best_checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
        last_checkpoint_path = checkpoint_dir / f"{run_name}_last.pt"

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
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )
