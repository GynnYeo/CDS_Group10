"""Plot Task 3a ROC curves for one model across train/val/test and 24h/72h."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_PREDICTIONS_PATH = Path("reports/metrics/task3_large_aftershock_predictions_m4_0.csv")
DEFAULT_OUTPUT_PATH = Path("reports/figures/task3a_roc_curves.png")
DEFAULT_MODEL_NAME = "task3_large_aftershock_xgboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Task 3a ROC curve plots from saved prediction CSVs.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to the Task 3a predictions CSV.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name to plot from the predictions CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency hint
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    args = parse_args()
    df = pd.read_csv(args.predictions_path)

    model_df = df.loc[df["model_name"] == args.model_name].copy()
    if model_df.empty:
        available_models = sorted(df["model_name"].unique())
        raise ValueError(
            f"Model {args.model_name!r} not found in {args.predictions_path}. "
            f"Available models: {available_models}"
        )

    split_order = ["train", "val", "test"]
    split_colors = {
        "train": "#4C78A8",
        "val": "#F58518",
        "test": "#54A24B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160, sharey=True)

    for ax, horizon in zip(axes, [24, 72], strict=True):
        horizon_df = model_df.loc[model_df["horizon"] == horizon].copy()
        if horizon_df.empty:
            ax.set_visible(False)
            continue

        for split_name in split_order:
            split_df = horizon_df.loc[horizon_df["split"] == split_name].copy()
            if split_df.empty:
                continue
            fpr, tpr, _ = roc_curve(split_df["y_true"], split_df["y_prob"])
            auc = roc_auc_score(split_df["y_true"], split_df["y_prob"])
            ax.plot(
                fpr,
                tpr,
                label=f"{split_name.title()} AUC = {auc:.3f}",
                color=split_colors[split_name],
                linewidth=2,
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"{horizon}h horizon")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Task 3a ROC Curves: {args.model_name}")
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved ROC curves to: {args.output_path}")


if __name__ == "__main__":
    main()
