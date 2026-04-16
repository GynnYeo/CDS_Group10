"""Calibrate saved neural count predictions using validation-set calibration.

This script does not retrain the neural network. It reads an existing
<run_name>_count_predictions.csv file, fits a simple post-processing
calibration model on the validation split only, applies it to all splits,
and saves calibrated count predictions plus calibrated count metrics.

Recommended first calibration:

    log1p(y_true) = intercept + slope * log1p(y_pred)

The calibrated prediction is then:

    y_pred_calibrated = expm1(intercept + slope * log1p(y_pred))

This is useful when the count model has ranking signal but its raw predictions
are systematically compressed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.neural_metrics import (
    compute_capped_count_metrics,
    compute_count_bucket_diagnostics,
    compute_count_metrics,
    compute_count_prediction_bucket_diagnostics,
    compute_positive_count_metrics,
)
from src.evaluation.validation import validate_prediction_frame
from src.models.neural.artifacts import build_run_artifact_paths
from src.models.neural.utils import resolve_repo_path


@dataclass(slots=True)
class HorizonCalibrationResult:
    """Stored calibration parameters for one forecast horizon."""

    horizon: int
    intercept: float
    slope: float
    n_val_rows: int
    val_mean_true: float
    val_mean_pred_before: float
    val_mean_pred_after: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate saved neural count predictions using validation data."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run name whose count prediction CSV should be calibrated.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/metrics",
        help="Base metrics directory containing run-scoped prediction CSVs.",
    )
    parser.add_argument(
        "--calibration-name",
        default="loglinear",
        help="Suffix used for calibrated prediction and metric output files.",
    )
    parser.add_argument(
        "--min-calibrated-count",
        type=float,
        default=0.0,
        help="Lower bound applied to calibrated count predictions.",
    )
    parser.add_argument(
        "--max-calibrated-count",
        type=float,
        default=None,
        help=(
            "Optional upper bound applied to calibrated count predictions. "
            "Leave unset to avoid changing the raw count scale."
        ),
    )
    return parser.parse_args()


def _validate_count_predictions(count_predictions: pd.DataFrame) -> None:
    validate_prediction_frame(
        prediction_df=count_predictions,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_true",
            "y_pred",
        ],
        split_col="split",
    )

    if (count_predictions["y_true"] < 0).any():
        raise ValueError("Count truth values must be non-negative.")

    if (count_predictions["y_pred"] < 0).any():
        raise ValueError("Count predictions must be non-negative before calibration.")


def _fit_loglinear_calibrator(
    validation_df: pd.DataFrame,
    horizon: int,
) -> tuple[LinearRegression, HorizonCalibrationResult]:
    """Fit log-linear calibration for one horizon using validation rows only."""

    horizon_val_df = validation_df.loc[validation_df["horizon"] == horizon].copy()

    if horizon_val_df.empty:
        raise ValueError(f"No validation rows found for horizon {horizon}.")

    x = np.log1p(horizon_val_df["y_pred"].to_numpy(dtype=float)).reshape(-1, 1)
    y = np.log1p(horizon_val_df["y_true"].to_numpy(dtype=float))

    calibrator = LinearRegression()
    calibrator.fit(x, y)

    calibrated_log_pred = calibrator.predict(x)
    calibrated_pred = np.expm1(calibrated_log_pred)
    calibrated_pred = np.clip(calibrated_pred, a_min=0.0, a_max=None)

    result = HorizonCalibrationResult(
        horizon=int(horizon),
        intercept=float(calibrator.intercept_),
        slope=float(calibrator.coef_[0]),
        n_val_rows=int(len(horizon_val_df)),
        val_mean_true=float(horizon_val_df["y_true"].mean()),
        val_mean_pred_before=float(horizon_val_df["y_pred"].mean()),
        val_mean_pred_after=float(calibrated_pred.mean()),
    )

    return calibrator, result


def fit_loglinear_calibrators(
    count_predictions: pd.DataFrame,
) -> tuple[dict[int, LinearRegression], pd.DataFrame]:
    """Fit one log-linear calibrator per horizon using validation split only."""

    validation_df = count_predictions.loc[count_predictions["split"] == "val"].copy()
    if validation_df.empty:
        raise ValueError("No validation rows found. Calibration requires split == 'val'.")

    calibrators: dict[int, LinearRegression] = {}
    calibration_rows: list[dict[str, float | int]] = []

    horizons = sorted(count_predictions["horizon"].astype(int).unique())
    for horizon in horizons:
        calibrator, result = _fit_loglinear_calibrator(
            validation_df=validation_df,
            horizon=horizon,
        )
        calibrators[horizon] = calibrator
        calibration_rows.append(
            {
                "horizon": result.horizon,
                "intercept": result.intercept,
                "slope": result.slope,
                "n_val_rows": result.n_val_rows,
                "val_mean_true": result.val_mean_true,
                "val_mean_pred_before": result.val_mean_pred_before,
                "val_mean_pred_after": result.val_mean_pred_after,
            }
        )

    return calibrators, pd.DataFrame(calibration_rows)


def apply_loglinear_calibration(
    count_predictions: pd.DataFrame,
    calibrators: dict[int, LinearRegression],
    min_calibrated_count: float = 0.0,
    max_calibrated_count: float | None = None,
    calibrated_model_suffix: str = "calibrated_loglinear",
) -> pd.DataFrame:
    """Apply horizon-specific log-linear calibration to all splits."""

    calibrated_frames: list[pd.DataFrame] = []

    for horizon, horizon_df in count_predictions.groupby("horizon", sort=True):
        horizon_int = int(horizon)
        if horizon_int not in calibrators:
            raise ValueError(f"No calibrator fitted for horizon {horizon_int}.")

        calibrator = calibrators[horizon_int]
        calibrated_df = horizon_df.copy()

        x = np.log1p(calibrated_df["y_pred"].to_numpy(dtype=float)).reshape(-1, 1)
        calibrated_log_pred = calibrator.predict(x)
        calibrated_pred = np.expm1(calibrated_log_pred)

        upper = np.inf if max_calibrated_count is None else max_calibrated_count
        calibrated_pred = np.clip(
            calibrated_pred,
            a_min=min_calibrated_count,
            a_max=upper,
        )

        calibrated_df["y_pred_original"] = calibrated_df["y_pred"].astype(float)
        calibrated_df["y_pred"] = calibrated_pred.astype(float)
        calibrated_df["model_name"] = (
            calibrated_df["model_name"].astype(str) + f"_{calibrated_model_suffix}"
        )

        calibrated_frames.append(calibrated_df)

    calibrated_predictions = pd.concat(calibrated_frames, ignore_index=True)

    validate_prediction_frame(
        prediction_df=calibrated_predictions,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_true",
            "y_pred",
            "y_pred_original",
        ],
        split_col="split",
    )

    return calibrated_predictions


def main() -> None:
    args = parse_args()

    run_paths = build_run_artifact_paths(
        run_name=args.run_name,
        base_output_dir=resolve_repo_path(args.output_dir),
        base_checkpoint_dir=None,
        create_dirs=True,
    )

    count_predictions_path = run_paths.count_predictions_path
    if not count_predictions_path.exists():
        raise FileNotFoundError(
            f"Count prediction file not found: {count_predictions_path}"
        )

    count_predictions = pd.read_csv(count_predictions_path)
    _validate_count_predictions(count_predictions)

    calibrators, calibration_params = fit_loglinear_calibrators(count_predictions)

    calibrated_suffix = f"calibrated_{args.calibration_name}"
    calibrated_predictions = apply_loglinear_calibration(
        count_predictions=count_predictions,
        calibrators=calibrators,
        min_calibrated_count=args.min_calibrated_count,
        max_calibrated_count=args.max_calibrated_count,
        calibrated_model_suffix=calibrated_suffix,
    )

    output_dir = run_paths.output_dir
    output_prefix = f"{args.run_name}_count_{calibrated_suffix}"

    calibrated_predictions_path = output_dir / f"{output_prefix}_predictions.csv"
    calibration_params_path = output_dir / f"{output_prefix}_parameters.csv"
    calibrated_count_metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    calibrated_capped_metrics_path = output_dir / f"{output_prefix}_capped_metrics.csv"
    calibrated_positive_metrics_path = (
        output_dir / f"{output_prefix}_positive_only_metrics.csv"
    )
    calibrated_bucket_diagnostics_path = (
        output_dir / f"{output_prefix}_bucket_diagnostics.csv"
    )
    calibrated_prediction_bucket_diagnostics_path = (
        output_dir / f"{output_prefix}_prediction_bucket_diagnostics.csv"
    )

    calibrated_count_metrics = compute_count_metrics(calibrated_predictions)
    calibrated_capped_metrics = compute_capped_count_metrics(calibrated_predictions)
    calibrated_positive_metrics = compute_positive_count_metrics(calibrated_predictions)
    calibrated_bucket_diagnostics = compute_count_bucket_diagnostics(
        calibrated_predictions
    )
    calibrated_prediction_bucket_diagnostics = (
        compute_count_prediction_bucket_diagnostics(calibrated_predictions)
    )

    calibrated_predictions.to_csv(calibrated_predictions_path, index=False)
    calibration_params.to_csv(calibration_params_path, index=False)
    calibrated_count_metrics.to_csv(calibrated_count_metrics_path, index=False)
    calibrated_capped_metrics.to_csv(calibrated_capped_metrics_path, index=False)
    calibrated_positive_metrics.to_csv(calibrated_positive_metrics_path, index=False)
    calibrated_bucket_diagnostics.to_csv(
        calibrated_bucket_diagnostics_path,
        index=False,
    )
    calibrated_prediction_bucket_diagnostics.to_csv(
        calibrated_prediction_bucket_diagnostics_path,
        index=False,
    )

    print(f"Loaded count predictions from: {count_predictions_path}")
    print(f"Calibrated predictions saved to: {calibrated_predictions_path}")
    print(f"Calibration parameters saved to: {calibration_params_path}")
    print(f"Calibrated count metrics saved to: {calibrated_count_metrics_path}")
    print(f"Calibrated capped count metrics saved to: {calibrated_capped_metrics_path}")
    print(f"Calibrated positive-only metrics saved to: {calibrated_positive_metrics_path}")
    print(f"Calibrated bucket diagnostics saved to: {calibrated_bucket_diagnostics_path}")
    print(
        "Calibrated prediction-bucket diagnostics saved to: "
        f"{calibrated_prediction_bucket_diagnostics_path}"
    )


if __name__ == "__main__":
    main()