"""Evaluate saved multitask prediction CSVs without retraining a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


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
from src.models.neural.utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction-only evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate saved multitask prediction CSVs without retraining."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run name used to locate prediction CSVs and name the metric outputs.",
    )
    parser.add_argument(
        "--input-dir",
        default="reports/metrics",
        help="Base directory containing run-scoped prediction CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/metrics",
        help="Base directory where run-scoped metric CSVs should be written.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model_name override to apply before computing grouped metrics.",
    )
    parser.add_argument(
        "--skip-magnitude",
        action="store_true",
        help="Skip magnitude evaluation when the magnitude prediction file is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = build_run_artifact_paths(
        run_name=args.run_name,
        base_output_dir=resolve_repo_path(args.input_dir),
        base_checkpoint_dir=None,
        create_dirs=False,
    )
    output_paths = build_run_artifact_paths(
        run_name=args.run_name,
        base_output_dir=resolve_repo_path(args.output_dir),
        base_checkpoint_dir=None,
        create_dirs=True,
    )

    probability_predictions = pd.read_csv(input_paths.probability_predictions_path)
    count_predictions = pd.read_csv(input_paths.count_predictions_path)

    magnitude_predictions: pd.DataFrame | None = None
    if input_paths.magnitude_predictions_path.exists():
        magnitude_predictions = pd.read_csv(input_paths.magnitude_predictions_path)
    elif not args.skip_magnitude:
        raise FileNotFoundError(
            f"Magnitude prediction file not found: {input_paths.magnitude_predictions_path}"
        )

    if args.model_name is not None:
        probability_predictions["model_name"] = args.model_name
        count_predictions["model_name"] = args.model_name
        if magnitude_predictions is not None:
            magnitude_predictions["model_name"] = args.model_name

    probability_metrics = compute_probability_metrics(probability_predictions)
    count_metrics = compute_count_metrics(count_predictions)
    count_capped_metrics = compute_capped_count_metrics(count_predictions)

    probability_metrics.to_csv(output_paths.probability_metrics_path, index=False)
    count_metrics.to_csv(output_paths.count_metrics_path, index=False)
    count_capped_metrics.to_csv(output_paths.count_capped_metrics_path, index=False)

    print(f"Probability metrics saved to: {output_paths.probability_metrics_path}")
    print(f"Count metrics saved to: {output_paths.count_metrics_path}")
    print(f"Capped Count metrics saved to: {output_paths.count_capped_metrics_path}")

    if magnitude_predictions is not None:
        magnitude_metrics = compute_magnitude_metrics(magnitude_predictions)
        magnitude_metrics.to_csv(output_paths.magnitude_metrics_path, index=False)
        print(f"Magnitude metrics saved to: {output_paths.magnitude_metrics_path}")
    else:
        print("Magnitude evaluation skipped.")


if __name__ == "__main__":
    main()
