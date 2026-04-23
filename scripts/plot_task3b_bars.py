"""Plot a simple grouped bar chart for Task 3b test metrics by model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS_PATH = Path("reports/metrics/task3_maxmag_metrics_m2_5.csv")
DEFAULT_XGBOOST_METRICS_PATH = Path("reports/metrics/task3_maxmag_metrics_m2_5_es30.csv")
DEFAULT_TABNET_METRICS_PATH = Path(
    "reports/metrics/task3_maxmag_metrics_m2_5_tabnet_e200_p30_b256_vb64_lr0_004_nd24_na24_ns4_g1_2.csv"
)
DEFAULT_OUTPUT_PATH = Path("reports/figures/task3b_model_bars.png")

MODEL_LABELS = {
    "task3_maxmag_mean": "Mean",
    "task3_maxmag_baths_law": "Bath's law",
    "task3_maxmag_linear_magonly": "Linear\n(mag only)",
    "task3_maxmag_linear": "Linear\n(multi)",
    "task3_maxmag_xgboost": "XGBoost",
    "task3_maxmag_tabnet": "TabNet",
}

MODEL_ORDER = list(MODEL_LABELS)

REPORTED_RESULTS = [
    {"model_name": "task3_maxmag_mean", "split": "test", "horizon": 24, "mae": 0.5690, "rmse": 0.7290},
    {"model_name": "task3_maxmag_mean", "split": "test", "horizon": 72, "mae": 0.5910, "rmse": 0.7540},
    {"model_name": "task3_maxmag_baths_law", "split": "test", "horizon": 24, "mae": 0.4840, "rmse": 0.6310},
    {"model_name": "task3_maxmag_baths_law", "split": "test", "horizon": 72, "mae": 0.4960, "rmse": 0.6390},
    {"model_name": "task3_maxmag_linear_magonly", "split": "test", "horizon": 24, "mae": 0.4620, "rmse": 0.6010},
    {"model_name": "task3_maxmag_linear_magonly", "split": "test", "horizon": 72, "mae": 0.4640, "rmse": 0.6030},
    {"model_name": "task3_maxmag_linear", "split": "test", "horizon": 24, "mae": 0.4480, "rmse": 0.5860},
    {"model_name": "task3_maxmag_linear", "split": "test", "horizon": 72, "mae": 0.4470, "rmse": 0.5880},
    {"model_name": "task3_maxmag_xgboost", "split": "test", "horizon": 24, "mae": 0.4383, "rmse": 0.5750},
    {"model_name": "task3_maxmag_xgboost", "split": "test", "horizon": 72, "mae": 0.4385, "rmse": 0.5790},
    {"model_name": "task3_maxmag_tabnet", "split": "test", "horizon": 24, "mae": 0.4834, "rmse": 0.6218},
    {"model_name": "task3_maxmag_tabnet", "split": "test", "horizon": 72, "mae": 0.4940, "rmse": 0.6367},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Task 3b grouped bar charts from saved metrics CSVs.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the Task 3b metrics CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--xgboost-metrics-path",
        type=Path,
        default=DEFAULT_XGBOOST_METRICS_PATH,
        help=(
            "Optional separate Task 3b XGBoost metrics CSV. Useful when the main "
            "metrics file contains baselines while XGBoost was saved to a "
            "backend-specific file."
        ),
    )
    parser.add_argument(
        "--xgboost-model-name",
        default="task3_maxmag_xgboost",
        help=(
            "Model name to use from the XGBoost metrics file. Set this to "
            "`task3_maxmag_xgboost_tuned` to plot the tuned XGBoost result."
        ),
    )
    parser.add_argument(
        "--tabnet-metrics-path",
        type=Path,
        default=DEFAULT_TABNET_METRICS_PATH,
        help=(
            "Optional separate Task 3b TabNet metrics CSV. Useful when the main "
            "metrics file contains XGBoost/baselines and TabNet was saved to a "
            "backend-specific file."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to plot (default: test).",
    )
    parser.add_argument(
        "--use-reported-results",
        action="store_true",
        help="Use the final Task 3b numbers already reported in the report instead of reading metrics CSVs.",
    )
    return parser.parse_args()


def _resolve_metrics(df: pd.DataFrame, model_name: str, split_name: str) -> pd.DataFrame:
    subset = df.loc[(df["split"] == split_name) & (df["model_name"] == model_name)].copy()
    if not subset.empty:
        return subset

    # Allow best-matching prefixed variants if the exact model name is absent.
    prefixed = df.loc[
        (df["split"] == split_name) & (df["model_name"].str.startswith(model_name)),
    ].copy()
    if prefixed.empty:
        raise ValueError(
            f"Could not find metrics for model prefix {model_name!r} on split {split_name!r}."
        )
    return prefixed


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency hint
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    args = parse_args()
    if args.use_reported_results:
        df = pd.DataFrame(REPORTED_RESULTS)
    else:
        df = pd.read_csv(args.metrics_path)
        if args.xgboost_metrics_path.exists():
            xgb_df = pd.read_csv(args.xgboost_metrics_path)
            df = pd.concat([df, xgb_df], ignore_index=True)
        if args.tabnet_metrics_path.exists():
            tabnet_df = pd.read_csv(args.tabnet_metrics_path)
            df = pd.concat([df, tabnet_df], ignore_index=True)
        df = df.drop_duplicates(subset=["model_name", "split", "horizon"], keep="last")

    rows: list[dict[str, float | int | str]] = []
    for model_name in MODEL_ORDER:
        source_model_name = model_name
        if model_name == "task3_maxmag_xgboost" and not args.use_reported_results:
            source_model_name = args.xgboost_model_name
        subset = _resolve_metrics(df, model_name=source_model_name, split_name=args.split)
        for horizon in [24, 72]:
            horizon_df = subset.loc[subset["horizon"] == horizon].copy()
            if horizon_df.empty:
                raise ValueError(
                    f"Missing horizon {horizon} metrics for model {source_model_name!r}."
                )
            row = horizon_df.iloc[0]
            rows.append(
                {
                    "model_name": model_name,
                    "label": MODEL_LABELS[model_name],
                    "horizon": horizon,
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                }
            )

    plot_df = pd.DataFrame(rows)
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160, sharex=True)
    metric_specs = [
        ("mae", "MAE", axes[0], "#4C78A8", "#F58518"),
        ("rmse", "RMSE", axes[1], "#72B7B2", "#E45756"),
    ]

    for metric_key, title, ax, color_24, color_72 in metric_specs:
        vals_24 = [
            plot_df.loc[
                (plot_df["model_name"] == model_name) & (plot_df["horizon"] == 24),
                metric_key,
            ].iloc[0]
            for model_name in MODEL_ORDER
        ]
        vals_72 = [
            plot_df.loc[
                (plot_df["model_name"] == model_name) & (plot_df["horizon"] == 72),
                metric_key,
            ].iloc[0]
            for model_name in MODEL_ORDER
        ]
        ax.bar(x - width / 2, vals_24, width, label="24h", color=color_24)
        ax.bar(x + width / 2, vals_72, width, label="72h", color=color_72)
        ax.set_title(f"Task 3b {title} ({args.split.title()} split)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved Task 3b bar chart to: {args.output_path}")


if __name__ == "__main__":
    main()
