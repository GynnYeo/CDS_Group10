"""Build a simplified markdown summary table from task-3 max-magnitude metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_parent_dir
from src.utils.paths import METRICS_DIR


DEFAULT_MAGNITUDE = "2_5"
MODEL_LABELS = {
    "task3_maxmag_mean": "Mean baseline",
    "task3_maxmag_baths_law": "B\u00e5th's law",
    "task3_maxmag_linear_magonly": "Linear (mag only)",
    "task3_maxmag_linear": "Linear (multi)",
    "task3_maxmag_xgboost": "XGB Regressor",
    "task3_maxmag_xgboost_tuned": "XGB Regressor (tuned)",
    "task3_maxmag_gb_reg": "GB Regressor",
}
MODEL_ORDER = [
    "task3_maxmag_mean",
    "task3_maxmag_baths_law",
    "task3_maxmag_linear_magonly",
    "task3_maxmag_linear",
    "task3_maxmag_xgboost",
    "task3_maxmag_xgboost_tuned",
    "task3_maxmag_gb_reg",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a simple markdown table from task-3 max-magnitude metrics.",
    )
    parser.add_argument(
        "--metrics-path",
        default=str(METRICS_DIR / f"task3_maxmag_metrics_m{DEFAULT_MAGNITUDE}.csv"),
        help="Path to the task-3 max-magnitude metrics CSV file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(METRICS_DIR / f"task3_maxmag_summary_table_m{DEFAULT_MAGNITUDE}.md"),
        help="Path to the markdown summary file to create.",
    )
    parser.add_argument(
        "--hparams-path",
        default=str(METRICS_DIR / f"task3_maxmag_hparams_m{DEFAULT_MAGNITUDE}.json"),
        help="Optional path to saved XGBoost hyperparameter metadata.",
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


def _build_table_for_horizon(metrics_df: pd.DataFrame, horizon: int) -> str:
    horizon_df = metrics_df.loc[metrics_df["horizon"] == horizon].copy()
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    horizon_df["model_order"] = horizon_df["model_name"].map(order_map).fillna(999)
    horizon_df = horizon_df.sort_values(["model_order", "model_name"]).reset_index(drop=True)

    lines = [
        f"### {horizon}h",
        "",
        "| Model | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|",
    ]

    for _, row in horizon_df.iterrows():
        model_name = str(row["model_name"])
        label = MODEL_LABELS.get(model_name, model_name)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_metric(float(row["mae"])),
                    _format_metric(float(row["rmse"])),
                    _format_metric(float(row["bias"])),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def _format_param_dict(params: dict | None) -> str:
    if not params:
        return "Not available"
    ordered_keys = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    ]
    parts = [f"{key}={params[key]}" for key in ordered_keys if key in params]
    return ", ".join(parts) if parts else "Not available"


def _build_hparam_section(hparams: dict | None) -> list[str]:
    if not hparams or "xgboost_models" not in hparams:
        return []

    xgb_models = hparams.get("xgboost_models", {})
    if not xgb_models:
        return []

    lines = [
        "",
        "## XGBoost Hyperparameters",
        "",
    ]
    for horizon in ("24h", "72h"):
        horizon_info = xgb_models.get(horizon)
        if not horizon_info:
            continue
        untuned_info = horizon_info.get("untuned")
        tuned_info = horizon_info.get("tuned")
        if untuned_info:
            lines.append(f"- {horizon} untuned: {_format_param_dict(untuned_info.get('params'))}")
        if tuned_info:
            lines.append(f"- {horizon} tuned: {_format_param_dict(tuned_info.get('params'))}")
    return lines


def build_summary_markdown(metrics_df: pd.DataFrame, split_name: str, hparams: dict | None = None) -> str:
    filtered = metrics_df.loc[metrics_df["split"] == split_name].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for split '{split_name}'.")

    sections = [
        f"# Task 3 Max-Magnitude Summary ({split_name.capitalize()} Split)",
        "",
        _build_table_for_horizon(filtered, 24),
        "",
        _build_table_for_horizon(filtered, 72),
        "",
        "Summary:",
        "Task 3 max-magnitude compares simple empirical, statistical, and nonlinear",
        "regression baselines using MAE, RMSE, and bias.",
    ]
    sections.extend(_build_hparam_section(hparams))
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = ensure_parent_dir(args.output_path)
    hparams_path = Path(args.hparams_path)

    metrics_df = pd.read_csv(metrics_path)
    hparams = None
    if hparams_path.exists():
        hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    markdown = build_summary_markdown(metrics_df, args.split, hparams=hparams)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Metrics loaded from: {metrics_path}")
    print(f"Summary table saved to: {output_path}")


if __name__ == "__main__":
    main()
