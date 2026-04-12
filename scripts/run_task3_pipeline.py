"""Run the task-3 larger-aftershock pipeline and save outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.run_task3 import run_task3_pipeline, save_task3_outputs
from src.utils.paths import METRICS_DIR


DEFAULT_DATASET_NAME = "earthquake_aftershock_v2_gcmt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run task 3: larger-aftershock probability modeling pipeline.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Processed dataset name to load from data/processed/.",
    )
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=4.0,
        help="Aftershock magnitude threshold for the larger-aftershock target.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "xgboost", "hist_gb"),
        default="auto",
        help="Model backend. 'auto' prefers xgboost when available.",
    )
    parser.add_argument(
        "--force-recompute-labels",
        action="store_true",
        help="Rebuild cached task-3 sidecar labels from interim artifacts.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    threshold_slug = str(args.magnitude_threshold).replace(".", "_")
    predictions_path = METRICS_DIR / f"task3_large_aftershock_predictions_m{threshold_slug}.csv"
    metrics_path = METRICS_DIR / f"task3_large_aftershock_metrics_m{threshold_slug}.csv"

    predictions_df, metrics_df = run_task3_pipeline(
        dataset_name=args.dataset_name,
        magnitude_threshold=args.magnitude_threshold,
        backend=args.backend,
        force_recompute_labels=args.force_recompute_labels,
        verbose=not args.quiet,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_task3_outputs(predictions_df, metrics_df, predictions_path, metrics_path)

    print(f"Dataset: {args.dataset_name}")
    print(f"Magnitude threshold: M >= {args.magnitude_threshold}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Prediction rows saved: {len(predictions_df)}")
    print(f"Metric rows saved: {len(metrics_df)}")


if __name__ == "__main__":
    main()
