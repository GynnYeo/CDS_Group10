"""Run the task-3 larger-aftershock pipeline and save outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.task3_prob.run import run_task3_pipeline, save_task3_outputs
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
        choices=("auto", "xgboost", "hist_gb", "tabnet"),
        default="auto",
        help="Model backend. 'auto' prefers xgboost, then tabnet, then hist_gb when available.",
    )
    parser.add_argument(
        "--tabnet-max-epochs",
        type=int,
        default=None,
        help="Optional max epochs for TabNet runs. Defaults to the model wrapper's existing max_iter.",
    )
    parser.add_argument(
        "--tabnet-patience",
        type=int,
        default=20,
        help="Early stopping patience for TabNet runs.",
    )
    parser.add_argument(
        "--tabnet-batch-size",
        type=int,
        default=1024,
        help="Batch size for TabNet runs.",
    )
    parser.add_argument(
        "--tabnet-virtual-batch-size",
        type=int,
        default=128,
        help="Virtual batch size for TabNet ghost batch normalization.",
    )
    parser.add_argument(
        "--tabnet-lr",
        type=float,
        default=0.02,
        help="Learning rate for TabNet runs.",
    )
    parser.add_argument(
        "--tabnet-nd",
        type=int,
        default=8,
        help="Decision-layer width (n_d) for TabNet runs.",
    )
    parser.add_argument(
        "--tabnet-na",
        type=int,
        default=8,
        help="Attention-layer width (n_a) for TabNet runs.",
    )
    parser.add_argument(
        "--tabnet-n-steps",
        type=int,
        default=3,
        help="Number of decision steps for TabNet runs.",
    )
    parser.add_argument(
        "--tabnet-gamma",
        type=float,
        default=1.3,
        help="Feature reuse coefficient gamma for TabNet runs.",
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
    tabnet_suffix = ""
    if args.backend == "tabnet":
        max_epochs_suffix = "default" if args.tabnet_max_epochs is None else str(args.tabnet_max_epochs)
        tabnet_suffix = (
            f"_tabnet_e{max_epochs_suffix}"
            f"_p{args.tabnet_patience}"
            f"_b{args.tabnet_batch_size}"
            f"_vb{args.tabnet_virtual_batch_size}"
            f"_lr{str(args.tabnet_lr).replace('.', '_')}"
            f"_nd{args.tabnet_nd}"
            f"_na{args.tabnet_na}"
            f"_ns{args.tabnet_n_steps}"
            f"_g{str(args.tabnet_gamma).replace('.', '_')}"
        )
    predictions_path = METRICS_DIR / f"task3_large_aftershock_predictions_m{threshold_slug}{tabnet_suffix}.csv"
    metrics_path = METRICS_DIR / f"task3_large_aftershock_metrics_m{threshold_slug}{tabnet_suffix}.csv"

    predictions_df, metrics_df = run_task3_pipeline(
        dataset_name=args.dataset_name,
        magnitude_threshold=args.magnitude_threshold,
        backend=args.backend,
        tabnet_max_epochs=args.tabnet_max_epochs,
        tabnet_patience=args.tabnet_patience,
        tabnet_batch_size=args.tabnet_batch_size,
        tabnet_virtual_batch_size=args.tabnet_virtual_batch_size,
        tabnet_lr=args.tabnet_lr,
        tabnet_n_d=args.tabnet_nd,
        tabnet_n_a=args.tabnet_na,
        tabnet_n_steps=args.tabnet_n_steps,
        tabnet_gamma=args.tabnet_gamma,
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
