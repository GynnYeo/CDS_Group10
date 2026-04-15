"""Run a binned max-aftershock-magnitude severity experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.feature_sets import EXTENDED_TABULAR_FEATURES
from src.models.input_layer import InputConfig, EXPECTED_SPLITS, load_modeling_splits, prepare_tabular_inputs
from src.models.task3_maxmag.binned import (
    BIN_LABELS,
    MajorityClassBaseline,
    BinnedSeverityModel,
    binned_target_col,
    build_binned_targets,
    evaluate_multiclass_predictions,
)
from src.models.task3_maxmag.labels import build_or_load_max_aftershock_magnitude_labels
from src.utils.paths import METRICS_DIR


DEFAULT_DATASET_NAME = "earthquake_aftershock_v2_gcmt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run binned max-magnitude severity modeling.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--backend", choices=("auto", "xgboost", "logreg"), default="auto")
    parser.add_argument("--min-aftershock-magnitude", type=float, default=2.5)
    parser.add_argument("--force-recompute-labels", action="store_true")
    return parser


def _prediction_table(split_name: str, horizon: int, model_name: str, ids: pd.Series, y_true: pd.Series, proba: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trigger_event_id": ids.to_numpy(),
            "split": split_name,
            "horizon": horizon,
            "model_name": model_name,
            "y_true": y_true.astype(int).to_numpy(),
        }
    )
    for idx, label in enumerate(BIN_LABELS):
        frame[f"prob_{label}"] = proba.iloc[:, idx].to_numpy()
    frame["y_pred"] = np.argmax(proba.to_numpy(), axis=1)
    return frame


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    splits = load_modeling_splits(dataset_name=args.dataset_name)
    label_df = build_or_load_max_aftershock_magnitude_labels(
        dataset_name=args.dataset_name,
        min_aftershock_magnitude=args.min_aftershock_magnitude,
        force_recompute=args.force_recompute_labels,
        verbose=True,
    )
    label_df = build_binned_targets(label_df)
    merged_splits = {
        split_name: split_df.merge(label_df, how="left", on="trigger_event_id")
        for split_name, split_df in splits.items()
    }

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []

    for horizon in (24, 72):
        target_col = binned_target_col(horizon)
        config = InputConfig(
            feature_cols=list(EXTENDED_TABULAR_FEATURES),
            target_col=target_col,
            missing_strategy="median",
            scale=False,
            drop_rows_with_missing_target=True,
            allow_missing_optional=True,
        )
        prepared = prepare_tabular_inputs(config=config, splits=merged_splits)

        majority = MajorityClassBaseline().fit(prepared.y_train)
        model = BinnedSeverityModel(backend=args.backend).fit(prepared.X_train, prepared.y_train)

        split_payloads = [
            ("train", prepared.X_train, prepared.y_train),
            ("val", prepared.X_val, prepared.y_val),
            ("test", prepared.X_test, prepared.y_test),
        ]
        for split_name, X_split, y_split in split_payloads:
            split_source = merged_splits[split_name].loc[y_split.index]
            majority_proba = pd.DataFrame(majority.predict_proba(X_split))
            model_proba = pd.DataFrame(model.predict_proba(X_split))
            for model_name, proba in [
                ("task3_maxmag_binned_majority", majority_proba),
                (f"task3_maxmag_binned_{model.backend_}", model_proba),
            ]:
                metrics = evaluate_multiclass_predictions(y_split, proba.to_numpy())
                all_metrics.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "horizon": horizon,
                        **metrics,
                    }
                )
                all_predictions.append(
                    _prediction_table(
                        split_name=split_name,
                        horizon=horizon,
                        model_name=model_name,
                        ids=split_source["trigger_event_id"],
                        y_true=y_split,
                        proba=proba,
                    )
                )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics).sort_values(["model_name", "horizon", "split"]).reset_index(drop=True)

    pred_path = METRICS_DIR / "task3_maxmag_binned_predictions.csv"
    metrics_path = METRICS_DIR / "task3_maxmag_binned_metrics.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Predictions saved to: {pred_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics_df.loc[metrics_df['split'] == 'test'].to_string(index=False))


if __name__ == "__main__":
    main()
