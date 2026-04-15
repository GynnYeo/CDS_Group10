"""Build readable markdown tables for task-3 overfitting checks."""

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


PROB_MODEL_LABELS = {
    "task3_climatology": "Climatology",
    "task3_simplified_rj": "RJ baseline",
    "task3_large_aftershock_xgboost": "XGB",
    "task3_large_aftershock_hist_gb": "GB",
}
MAXMAG_MODEL_LABELS = {
    "task3_maxmag_mean": "Mean baseline",
    "task3_maxmag_baths_law": "Bath's law",
    "task3_maxmag_linear_magonly": "Linear (mag only)",
    "task3_maxmag_linear": "Linear (multi)",
    "task3_maxmag_xgboost": "XGB",
    "task3_maxmag_xgboost_tuned": "XGB (tuned)",
    "task3_maxmag_gb_reg": "GB regressor",
}
PROB_MODEL_ORDER = [
    "task3_climatology",
    "task3_simplified_rj",
    "task3_large_aftershock_xgboost",
    "task3_large_aftershock_hist_gb",
]
MAXMAG_MODEL_ORDER = [
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
        description="Create markdown overfitting tables for both task-3 pipelines.",
    )
    parser.add_argument(
        "--prob-metrics",
        default=str(METRICS_DIR / "task3_large_aftershock_metrics_m4_0.csv"),
        help="Path to task-3 probability metrics CSV.",
    )
    parser.add_argument(
        "--maxmag-metrics",
        default=str(METRICS_DIR / "task3_maxmag_metrics_m2_5.csv"),
        help="Path to task-3 max-magnitude metrics CSV.",
    )
    parser.add_argument(
        "--output-path",
        default=str(METRICS_DIR / "task3_overfitting_summary.md"),
        help="Path to markdown output file.",
    )
    return parser


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _risk_score_label(score: int) -> str:
    return {0: "low", 1: "moderate", 2: "high"}[score]


def _gap_risk(delta: float) -> int:
    gap = abs(delta)
    if gap < 0.03:
        return 0
    if gap < 0.08:
        return 1
    return 2


def _build_probability_table(df: pd.DataFrame, horizon: int) -> str:
    horizon_df = df.loc[df["horizon"] == horizon].copy()
    order_map = {name: idx for idx, name in enumerate(PROB_MODEL_ORDER)}
    horizon_df["model_order"] = horizon_df["model_name"].map(order_map).fillna(999)
    horizon_df = horizon_df.sort_values(["model_order", "model_name"])

    lines = [
        f"### Probability {horizon}h",
        "",
        "| Model | Train Brier | Val Brier | Test Brier | Train AUC | Val AUC | Test AUC | Risk |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for model_name, model_df in horizon_df.groupby("model_name", sort=False):
        indexed = model_df.set_index("split")
        if not {"train", "val", "test"}.issubset(indexed.index):
            continue
        train_brier = float(indexed.loc["train", "brier_score"])
        val_brier = float(indexed.loc["val", "brier_score"])
        test_brier = float(indexed.loc["test", "brier_score"])
        train_auc = float(indexed.loc["train", "roc_auc"])
        val_auc = float(indexed.loc["val", "roc_auc"])
        test_auc = float(indexed.loc["test", "roc_auc"])
        risk = max(
            _gap_risk(val_brier - train_brier),
            _gap_risk(test_brier - train_brier),
            _gap_risk(train_auc - val_auc),
            _gap_risk(train_auc - test_auc),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    PROB_MODEL_LABELS.get(model_name, model_name),
                    _fmt(train_brier),
                    _fmt(val_brier),
                    _fmt(test_brier),
                    _fmt(train_auc),
                    _fmt(val_auc),
                    _fmt(test_auc),
                    _risk_score_label(risk),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def _build_maxmag_table(df: pd.DataFrame, horizon: int) -> str:
    horizon_df = df.loc[df["horizon"] == horizon].copy()
    order_map = {name: idx for idx, name in enumerate(MAXMAG_MODEL_ORDER)}
    horizon_df["model_order"] = horizon_df["model_name"].map(order_map).fillna(999)
    horizon_df = horizon_df.sort_values(["model_order", "model_name"])

    lines = [
        f"### Max Magnitude {horizon}h",
        "",
        "| Model | Train MAE | Val MAE | Test MAE | Train RMSE | Val RMSE | Test RMSE | Risk |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for model_name, model_df in horizon_df.groupby("model_name", sort=False):
        indexed = model_df.set_index("split")
        if not {"train", "val", "test"}.issubset(indexed.index):
            continue
        train_mae = float(indexed.loc["train", "mae"])
        val_mae = float(indexed.loc["val", "mae"])
        test_mae = float(indexed.loc["test", "mae"])
        train_rmse = float(indexed.loc["train", "rmse"])
        val_rmse = float(indexed.loc["val", "rmse"])
        test_rmse = float(indexed.loc["test", "rmse"])
        risk = max(
            _gap_risk(val_mae - train_mae),
            _gap_risk(test_mae - train_mae),
            _gap_risk(val_rmse - train_rmse),
            _gap_risk(test_rmse - train_rmse),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    MAXMAG_MODEL_LABELS.get(model_name, model_name),
                    _fmt(train_mae),
                    _fmt(val_mae),
                    _fmt(test_mae),
                    _fmt(train_rmse),
                    _fmt(val_rmse),
                    _fmt(test_rmse),
                    _risk_score_label(risk),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_markdown(prob_df: pd.DataFrame, maxmag_df: pd.DataFrame) -> str:
    sections = [
        "# Task 3 Overfitting Summary",
        "",
        "Interpretation:",
        "- `low`: train and val/test are close",
        "- `moderate`: noticeable but not huge train-vs-val/test gap",
        "- `high`: large generalization gap, so overfitting risk is stronger",
        "",
        _build_probability_table(prob_df, 24),
        "",
        _build_probability_table(prob_df, 72),
        "",
        _build_maxmag_table(maxmag_df, 24),
        "",
        _build_maxmag_table(maxmag_df, 72),
        "",
        "Note:",
        "A bad validation year can also make risk look worse, especially when validation only covers one year.",
    ]
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    prob_df = pd.read_csv(args.prob_metrics)
    maxmag_df = pd.read_csv(args.maxmag_metrics)
    output_path = ensure_parent_dir(args.output_path)
    markdown = build_markdown(prob_df, maxmag_df)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Probability metrics loaded from: {args.prob_metrics}")
    print(f"Max-magnitude metrics loaded from: {args.maxmag_metrics}")
    print(f"Overfitting summary saved to: {output_path}")


if __name__ == "__main__":
    main()
