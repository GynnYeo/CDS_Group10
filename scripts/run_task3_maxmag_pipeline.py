"""Run the task-3 max-aftershock-magnitude pipeline and save outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.task3_maxmag.run import (
    run_task3_maxmag_pipeline,
    save_task3_maxmag_outputs,
)
from src.utils.paths import METRICS_DIR


DEFAULT_DATASET_NAME = "earthquake_aftershock_v2_gcmt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run task 3 max-aftershock-magnitude modeling pipeline.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Processed dataset name to load from data/processed/.",
    )
    parser.add_argument(
        "--min-aftershock-magnitude",
        type=float,
        default=2.5,
        help="Minimum aftershock magnitude used when building max-magnitude labels.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "xgboost", "gb_reg"),
        default="auto",
        help="Model backend. 'auto' prefers xgboost when available.",
    )
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Run a small validation-based hyperparameter search for the XGBoost regressor.",
    )
    parser.add_argument(
        "--force-recompute-labels",
        action="store_true",
        help="Rebuild cached task-3 max-magnitude labels from interim artifacts.",
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
    magnitude_slug = str(args.min_aftershock_magnitude).replace(".", "_")
    predictions_path = METRICS_DIR / f"task3_maxmag_predictions_m{magnitude_slug}.csv"
    metrics_path = METRICS_DIR / f"task3_maxmag_metrics_m{magnitude_slug}.csv"
    metadata_path = METRICS_DIR / f"task3_maxmag_hparams_m{magnitude_slug}.json"

    predictions_df, metrics_df, metadata = run_task3_maxmag_pipeline(
        dataset_name=args.dataset_name,
        min_aftershock_magnitude=args.min_aftershock_magnitude,
        backend=args.backend,
        tune_xgboost=args.tune_xgboost,
        force_recompute_labels=args.force_recompute_labels,
        verbose=not args.quiet,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_task3_maxmag_outputs(
        predictions_df,
        metrics_df,
        predictions_path,
        metrics_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(f"Dataset: {args.dataset_name}")
    print(f"Min aftershock magnitude: M >= {args.min_aftershock_magnitude}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Hyperparameters saved to: {metadata_path}")
    print(f"Prediction rows saved: {len(predictions_df)}")
    print(f"Metric rows saved: {len(metrics_df)}")


if __name__ == "__main__":
    main()
