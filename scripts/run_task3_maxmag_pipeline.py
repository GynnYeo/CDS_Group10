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
        choices=("auto", "xgboost", "gb_reg", "tabnet"),
        default="auto",
        help="Model backend. 'auto' prefers xgboost, then tabnet, then gb_reg when available.",
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
        "--feature-set",
        choices=("extended", "compact"),
        default="extended",
        help="Feature set to use for the max-magnitude model.",
    )
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Run a small validation-based hyperparameter search for the XGBoost regressor.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help="Optional XGBoost early stopping rounds to use when validation data is available.",
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
    feature_suffix = "" if args.feature_set == "extended" else f"_{args.feature_set}"
    es_suffix = "" if args.early_stopping_rounds is None else f"_es{args.early_stopping_rounds}"
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
    predictions_path = METRICS_DIR / f"task3_maxmag_predictions_m{magnitude_slug}{feature_suffix}{es_suffix}{tabnet_suffix}.csv"
    metrics_path = METRICS_DIR / f"task3_maxmag_metrics_m{magnitude_slug}{feature_suffix}{es_suffix}{tabnet_suffix}.csv"
    metadata_path = METRICS_DIR / f"task3_maxmag_hparams_m{magnitude_slug}{feature_suffix}{es_suffix}{tabnet_suffix}.json"

    predictions_df, metrics_df, metadata = run_task3_maxmag_pipeline(
        dataset_name=args.dataset_name,
        min_aftershock_magnitude=args.min_aftershock_magnitude,
        feature_set=args.feature_set,
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
        tune_xgboost=args.tune_xgboost,
        early_stopping_rounds=args.early_stopping_rounds,
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
