"""Evaluate saved multitask MLP prediction CSVs without retraining."""

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
    compute_count_bucket_diagnostics,
    compute_count_metrics,
    compute_count_prediction_bucket_diagnostics,
    compute_magnitude_metrics,
    compute_positive_count_metrics,
    compute_probability_metrics,
)
from src.models.neural.artifacts import build_run_artifact_paths
from src.models.neural.utils import resolve_repo_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation-only workflow."""

    parser = argparse.ArgumentParser(
        description="Recompute multitask neural metrics from saved prediction CSVs."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run name used as the prediction file prefix and output folder name.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/metrics",
        help="Base directory containing run-scoped prediction and metric files.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model_name override to apply before computing metrics.",
    )
    parser.add_argument(
        "--skip-magnitude",
        action="store_true",
        help="Skip magnitude metrics when magnitude prediction CSV is unavailable.",
    )
    return parser.parse_args()


def _maybe_override_model_name(
    prediction_df: pd.DataFrame,
    model_name: str | None,
) -> pd.DataFrame:
    """Optionally override model_name while preserving the input dataframe."""

    if model_name is None:
        return prediction_df

    updated_df = prediction_df.copy()
    updated_df["model_name"] = model_name
    return updated_df


def main() -> None:
    args = parse_args()
    run_paths = build_run_artifact_paths(
        run_name=args.run_name,
        base_output_dir=resolve_repo_path(args.output_dir),
        base_checkpoint_dir=None,
        create_dirs=True,
    )

    probability_predictions = pd.read_csv(run_paths.probability_predictions_path)
    count_predictions = pd.read_csv(run_paths.count_predictions_path)

    probability_predictions = _maybe_override_model_name(
        prediction_df=probability_predictions,
        model_name=args.model_name,
    )
    count_predictions = _maybe_override_model_name(
        prediction_df=count_predictions,
        model_name=args.model_name,
    )

    probability_metrics = compute_probability_metrics(probability_predictions)
    count_metrics = compute_count_metrics(count_predictions)
    count_capped_metrics = compute_capped_count_metrics(count_predictions)
    count_positive_metrics = compute_positive_count_metrics(count_predictions)
    count_bucket_diagnostics = compute_count_bucket_diagnostics(count_predictions)
    count_prediction_bucket_diagnostics = compute_count_prediction_bucket_diagnostics(count_predictions)

    probability_metrics.to_csv(run_paths.probability_metrics_path, index=False)
    count_metrics.to_csv(run_paths.count_metrics_path, index=False)
    count_capped_metrics.to_csv(run_paths.count_capped_metrics_path, index=False)

    count_positive_metrics_path = (
        run_paths.output_dir / f"{args.run_name}_count_positive_only_metrics.csv"
    )
    count_bucket_diagnostics_path = (
        run_paths.output_dir / f"{args.run_name}_count_bucket_diagnostics.csv"
    )
    count_prediction_bucket_diagnostics_path = (
        run_paths.output_dir / f"{args.run_name}_count_prediction_bucket_diagnostics.csv"
    )
    count_positive_metrics.to_csv(count_positive_metrics_path, index=False)
    count_bucket_diagnostics.to_csv(count_bucket_diagnostics_path, index=False)
    count_prediction_bucket_diagnostics.to_csv(
        count_prediction_bucket_diagnostics_path,
        index=False,
    )

    print(f"Probability metrics saved to: {run_paths.probability_metrics_path}")
    print(f"Count metrics saved to: {run_paths.count_metrics_path}")
    print(f"Capped count metrics saved to: {run_paths.count_capped_metrics_path}")
    print(f"Positive-only count metrics saved to: {count_positive_metrics_path}")
    print(f"Count bucket diagnostics saved to: {count_bucket_diagnostics_path}")
    print(
        "Count prediction bucket diagnostics saved to: "
        f"{count_prediction_bucket_diagnostics_path}"
    )

    if args.skip_magnitude:
        print("Magnitude metrics skipped.")
        return

    magnitude_predictions = pd.read_csv(run_paths.magnitude_predictions_path)
    magnitude_predictions = _maybe_override_model_name(
        prediction_df=magnitude_predictions,
        model_name=args.model_name,
    )
    magnitude_metrics = compute_magnitude_metrics(magnitude_predictions)
    magnitude_metrics.to_csv(run_paths.magnitude_metrics_path, index=False)
    print(f"Magnitude metrics saved to: {run_paths.magnitude_metrics_path}")


if __name__ == "__main__":
    main()
