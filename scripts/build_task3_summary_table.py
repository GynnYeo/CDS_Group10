"""Build a simplified markdown summary table from task-3 metrics CSV output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_parent_dir
from src.utils.paths import METRICS_DIR


DEFAULT_THRESHOLD = "4_0"
MODEL_LABELS = {
    "task3_climatology": "Climatology",
    "task3_simplified_rj": "RJ baseline",
    "task3_large_aftershock_xgboost": "XGB Model",
    "task3_large_aftershock_hist_gb": "GB Model",
}
MODEL_ORDER = [
    "task3_climatology",
    "task3_simplified_rj",
    "task3_large_aftershock_xgboost",
    "task3_large_aftershock_hist_gb",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a simple markdown table from task-3 metrics output.",
    )
    parser.add_argument(
        "--metrics-path",
        default=str(METRICS_DIR / f"task3_large_aftershock_metrics_m{DEFAULT_THRESHOLD}.csv"),
        help="Path to the task-3 metrics CSV file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(METRICS_DIR / f"task3_large_aftershock_summary_table_m{DEFAULT_THRESHOLD}.md"),
        help="Path to the markdown summary file to create.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Which split to summarize.",
    )
    return parser


def _format_metric(value: float) -> str:
    return f"{value:.2f}"


def _display_model_name(model_name: str, horizon: int) -> str:
    label = MODEL_LABELS.get(model_name, model_name)
    if model_name.startswith("task3_large_aftershock_"):
        return f"{label} y{horizon}"
    return label


def _build_table_for_horizon(metrics_df: pd.DataFrame, horizon: int) -> str:
    horizon_df = metrics_df.loc[metrics_df["horizon"] == horizon].copy()
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    horizon_df["model_order"] = horizon_df["model_name"].map(order_map).fillna(999)
    horizon_df = horizon_df.sort_values(["model_order", "model_name"]).reset_index(drop=True)

    lines = [
        f"### {horizon}h",
        "",
        "| Model | ROC-AUC | LogLoss | Brier |",
        "|---|---:|---:|---:|",
    ]

    for _, row in horizon_df.iterrows():
        model_label = _display_model_name(str(row["model_name"]), horizon)
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label,
                    _format_metric(float(row["roc_auc"])),
                    _format_metric(float(row["log_loss"])),
                    _format_metric(float(row["brier_score"])),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_summary_markdown(metrics_df: pd.DataFrame, split_name: str) -> str:
    filtered = metrics_df.loc[metrics_df["split"] == split_name].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for split '{split_name}'.")

    sections = [
        f"# Task 3 Summary ({split_name.capitalize()} Split)",
        "",
        _build_table_for_horizon(filtered, 24),
        "",
        _build_table_for_horizon(filtered, 72),
        "",
        "Summary:",
        "Task 3 compares larger-aftershock probability forecasting across climatology,",
        "a simplified RJ-style baseline, and the ML model using ROC-AUC, LogLoss, and Brier score.",
    ]
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = ensure_parent_dir(args.output_path)

    metrics_df = pd.read_csv(metrics_path)
    markdown = build_summary_markdown(metrics_df, args.split)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Metrics loaded from: {metrics_path}")
    print(f"Summary table saved to: {output_path}")


if __name__ == "__main__":
    main()
