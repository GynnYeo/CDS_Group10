"""Run the existing baseline pipeline and save outputs to reports/metrics/."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.run_baselines import run_all_baselines
from src.utils.paths import METRICS_DIR


DATASET_NAME = "earthquake_aftershock_v2_gcmt"
PREDICTIONS_PATH = METRICS_DIR / "baselines_predictions.csv"
METRICS_PATH = METRICS_DIR / "baselines_metrics.csv"


def main() -> None:
    predictions_df, metrics_df = run_all_baselines(dataset_name=DATASET_NAME)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Prediction rows saved: {len(predictions_df)}")
    print(f"Metric rows saved: {len(metrics_df)}")


if __name__ == "__main__":
    main()
