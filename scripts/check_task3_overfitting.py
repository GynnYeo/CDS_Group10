"""Quick train/val/test gap check for both task-3 pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize train/val/test generalization gaps for both task-3 pipelines.",
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
    return parser


def _format_value(value: float) -> str:
    return f"{value:.3f}"


def _risk_label(delta: float, *, higher_is_better: bool) -> str:
    gap = abs(delta)
    if gap < 0.03:
        return "low"
    if gap < 0.08:
        return "moderate"
    return "high"


def _metric_delta(train_value: float, eval_value: float, *, higher_is_better: bool) -> float:
    if higher_is_better:
        return train_value - eval_value
    return eval_value - train_value


def _summarize_probability(df: pd.DataFrame) -> str:
    lines = [
        "== Task 3 Probability ==",
        "",
    ]
    target_models = [m for m in df["model_name"].unique() if m != "task3_climatology"]
    for horizon in (24, 72):
        lines.append(f"[{horizon}h]")
        horizon_df = df.loc[df["horizon"] == horizon].copy()
        for model_name in sorted(target_models):
            model_df = horizon_df.loc[horizon_df["model_name"] == model_name].set_index("split")
            if not {"train", "val", "test"}.issubset(model_df.index):
                continue
            train_brier = float(model_df.loc["train", "brier_score"])
            val_brier = float(model_df.loc["val", "brier_score"])
            test_brier = float(model_df.loc["test", "brier_score"])
            train_auc = float(model_df.loc["train", "roc_auc"])
            val_auc = float(model_df.loc["val", "roc_auc"])
            test_auc = float(model_df.loc["test", "roc_auc"])
            val_gap = _metric_delta(train_brier, val_brier, higher_is_better=False)
            test_gap = _metric_delta(train_brier, test_brier, higher_is_better=False)
            auc_test_gap = _metric_delta(train_auc, test_auc, higher_is_better=True)
            risk = max(
                _risk_label(val_gap, higher_is_better=False),
                _risk_label(test_gap, higher_is_better=False),
                _risk_label(auc_test_gap, higher_is_better=True),
                key=lambda label: {"low": 0, "moderate": 1, "high": 2}[label],
            )
            label = PROB_MODEL_LABELS.get(model_name, model_name)
            lines.append(
                "  "
                + f"{label}: "
                + f"Brier train/val/test={_format_value(train_brier)}/{_format_value(val_brier)}/{_format_value(test_brier)}, "
                + f"ROC-AUC train/val/test={_format_value(train_auc)}/{_format_value(val_auc)}/{_format_value(test_auc)}, "
                + f"gap-risk={risk}"
            )
        lines.append("")
    lines.append("Rule of thumb: if train is much better than val/test, that suggests overfitting.")
    return "\n".join(lines)


def _summarize_maxmag(df: pd.DataFrame) -> str:
    lines = [
        "== Task 3 Max Magnitude ==",
        "",
    ]
    target_models = [m for m in df["model_name"].unique() if m != "task3_maxmag_mean"]
    for horizon in (24, 72):
        lines.append(f"[{horizon}h]")
        horizon_df = df.loc[df["horizon"] == horizon].copy()
        for model_name in sorted(target_models):
            model_df = horizon_df.loc[horizon_df["model_name"] == model_name].set_index("split")
            if not {"train", "val", "test"}.issubset(model_df.index):
                continue
            train_rmse = float(model_df.loc["train", "rmse"])
            val_rmse = float(model_df.loc["val", "rmse"])
            test_rmse = float(model_df.loc["test", "rmse"])
            train_mae = float(model_df.loc["train", "mae"])
            val_mae = float(model_df.loc["val", "mae"])
            test_mae = float(model_df.loc["test", "mae"])
            val_gap = _metric_delta(train_rmse, val_rmse, higher_is_better=False)
            test_gap = _metric_delta(train_rmse, test_rmse, higher_is_better=False)
            risk = max(
                _risk_label(val_gap, higher_is_better=False),
                _risk_label(test_gap, higher_is_better=False),
                key=lambda label: {"low": 0, "moderate": 1, "high": 2}[label],
            )
            label = MAXMAG_MODEL_LABELS.get(model_name, model_name)
            lines.append(
                "  "
                + f"{label}: "
                + f"RMSE train/val/test={_format_value(train_rmse)}/{_format_value(val_rmse)}/{_format_value(test_rmse)}, "
                + f"MAE train/val/test={_format_value(train_mae)}/{_format_value(val_mae)}/{_format_value(test_mae)}, "
                + f"gap-risk={risk}"
            )
        lines.append("")
    lines.append("Rule of thumb: if train MAE/RMSE is much lower than val/test, that suggests overfitting.")
    return "\n".join(lines)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    prob_df = pd.read_csv(args.prob_metrics)
    maxmag_df = pd.read_csv(args.maxmag_metrics)

    print(_summarize_probability(prob_df))
    print()
    print(_summarize_maxmag(maxmag_df))


if __name__ == "__main__":
    main()
