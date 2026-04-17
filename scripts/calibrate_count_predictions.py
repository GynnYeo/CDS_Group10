"""Calibrate saved neural count predictions using validation-set calibration.

This script does not retrain the neural network. It reads an existing
<run_name>_count_predictions.csv file, fits a post-processing calibration
model on the validation split only, applies it to all splits, and saves
calibrated count predictions plus calibrated count metrics.

Supported methods:

1. Basic log-linear calibration:

    log1p(y_true) = intercept + slope * log1p(y_pred)

2. Probability-aware log-linear calibration:

    log1p(y_true) = intercept
                  + count_log_slope * log1p(y_pred)
                  + probability_slope * y_prob

3. Optional probability-aware calibration with interaction:

    log1p(y_true) = intercept
                  + count_log_slope * log1p(y_pred)
                  + probability_slope * y_prob
                  + interaction_slope * log1p(y_pred) * y_prob

The calibrated prediction is converted back to raw-count scale using expm1.
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


SUPPORTED_METHODS = (
    "loglinear",
    "loglinear_probaware",
    "loglinear_probaware_interaction",
)


@dataclass(slots=True)
class HorizonCalibrationResult:
    """Stored calibration parameters for one forecast horizon."""

    horizon: int
    method: str
    intercept: float
    count_log_slope: float
    probability_slope: float | None
    interaction_slope: float | None
    n_val_rows: int
    val_mean_true: float
    val_mean_pred_before: float
    val_mean_pred_after: float
    val_mean_probability: float | None


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
        "--method",
        choices=SUPPORTED_METHODS,
        default="loglinear",
        help="Calibration method to fit.",
    )
    parser.add_argument(
        "--calibration-name",
        default=None,
        help=(
            "Optional suffix used for calibrated output files. If omitted, "
            "the method name is used."
        ),
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


def _validate_probability_predictions(probability_predictions: pd.DataFrame) -> None:
    validate_prediction_frame(
        prediction_df=probability_predictions,
        required_columns=[
            "trigger_event_id",
            "split",
            "horizon",
            "model_name",
            "y_true",
            "y_prob",
        ],
        split_col="split",
    )

    if probability_predictions["y_prob"].isna().any():
        raise ValueError("Probability predictions contain missing y_prob values.")

    if not probability_predictions["y_prob"].between(0.0, 1.0).all():
        raise ValueError("Probability predictions must be within [0, 1].")


def attach_probability_predictions(
    count_predictions: pd.DataFrame,
    probability_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join horizon-matched probability predictions onto count predictions."""

    _validate_count_predictions(count_predictions)
    _validate_probability_predictions(probability_predictions)

    join_keys = ["trigger_event_id", "split", "horizon"]
    probability_features = probability_predictions[
        join_keys + ["model_name", "y_true", "y_prob"]
    ].rename(
        columns={
            "model_name": "probability_model_name",
            "y_true": "probability_y_true",
        }
    )

    duplicate_probability_keys = probability_features.duplicated(join_keys).sum()
    if duplicate_probability_keys > 0:
        raise ValueError(
            "Probability predictions contain duplicate rows for the same "
            f"trigger/split/horizon key: {duplicate_probability_keys} duplicates."
        )

    merged = count_predictions.merge(
        probability_features,
        on=join_keys,
        how="left",
        validate="one_to_one",
    )

    if merged["y_prob"].isna().any():
        missing = int(merged["y_prob"].isna().sum())
        raise ValueError(
            "Some count prediction rows could not be matched to probability "
            f"predictions. Missing probability rows: {missing}."
        )

    probability_truth_mismatch = (
        merged["probability_y_true"].astype(float)
        != (merged["y_true"].astype(float) > 0).astype(float)
    )
    if probability_truth_mismatch.any():
        n_mismatch = int(probability_truth_mismatch.sum())
        raise ValueError(
            "Joined probability y_true does not match count positivity for "
            f"{n_mismatch} rows. Check that both files come from the same run."
        )

    return merged


def _build_calibration_features(
    df: pd.DataFrame,
    method: str,
) -> tuple[np.ndarray, list[str]]:
    """Build the calibration design matrix for the requested method."""

    count_log = np.log1p(df["y_pred"].to_numpy(dtype=float))

    if method == "loglinear":
        return count_log.reshape(-1, 1), ["log1p_y_pred"]

    if "y_prob" not in df.columns:
        raise ValueError(
            f"Method {method!r} requires probability predictions joined as y_prob."
        )

    probability = df["y_prob"].to_numpy(dtype=float)
    features = [count_log, probability]
    feature_names = ["log1p_y_pred", "y_prob"]

    if method == "loglinear_probaware_interaction":
        features.append(count_log * probability)
        feature_names.append("log1p_y_pred_x_y_prob")

    if method not in {"loglinear_probaware", "loglinear_probaware_interaction"}:
        raise ValueError(f"Unsupported calibration method: {method}")

    return np.column_stack(features), feature_names


def _fit_horizon_calibrator(
    validation_df: pd.DataFrame,
    horizon: int,
    method: str,
) -> tuple[LinearRegression, HorizonCalibrationResult]:
    """Fit one calibration model for one horizon using validation rows only."""

    horizon_val_df = validation_df.loc[validation_df["horizon"] == horizon].copy()

    if horizon_val_df.empty:
        raise ValueError(f"No validation rows found for horizon {horizon}.")

    x, feature_names = _build_calibration_features(horizon_val_df, method=method)
    y = np.log1p(horizon_val_df["y_true"].to_numpy(dtype=float))

    calibrator = LinearRegression()
    calibrator.fit(x, y)

    calibrated_log_pred = calibrator.predict(x)
    calibrated_pred = np.expm1(calibrated_log_pred)
    calibrated_pred = np.clip(calibrated_pred, a_min=0.0, a_max=None)

    coefficients = dict(zip(feature_names, calibrator.coef_))
    result = HorizonCalibrationResult(
        horizon=int(horizon),
        method=method,
        intercept=float(calibrator.intercept_),
        count_log_slope=float(coefficients.get("log1p_y_pred", np.nan)),
        probability_slope=(
            float(coefficients["y_prob"]) if "y_prob" in coefficients else None
        ),
        interaction_slope=(
            float(coefficients["log1p_y_pred_x_y_prob"])
            if "log1p_y_pred_x_y_prob" in coefficients
            else None
        ),
        n_val_rows=int(len(horizon_val_df)),
        val_mean_true=float(horizon_val_df["y_true"].mean()),
        val_mean_pred_before=float(horizon_val_df["y_pred"].mean()),
        val_mean_pred_after=float(calibrated_pred.mean()),
        val_mean_probability=(
            float(horizon_val_df["y_prob"].mean())
            if "y_prob" in horizon_val_df.columns
            else None
        ),
    )

    return calibrator, result


def fit_calibrators(
    calibration_input: pd.DataFrame,
    method: str,
) -> tuple[dict[int, LinearRegression], pd.DataFrame]:
    """Fit one calibrator per horizon using validation split only."""

    validation_df = calibration_input.loc[calibration_input["split"] == "val"].copy()
    if validation_df.empty:
        raise ValueError("No validation rows found. Calibration requires split == 'val'.")

    calibrators: dict[int, LinearRegression] = {}
    calibration_rows: list[dict[str, float | int | str | None]] = []

    horizons = sorted(calibration_input["horizon"].astype(int).unique())
    for horizon in horizons:
        calibrator, result = _fit_horizon_calibrator(
            validation_df=validation_df,
            horizon=horizon,
            method=method,
        )
        calibrators[horizon] = calibrator
        calibration_rows.append(
            {
                "horizon": result.horizon,
                "method": result.method,
                "intercept": result.intercept,
                "count_log_slope": result.count_log_slope,
                "probability_slope": result.probability_slope,
                "interaction_slope": result.interaction_slope,
                "n_val_rows": result.n_val_rows,
                "val_mean_true": result.val_mean_true,
                "val_mean_pred_before": result.val_mean_pred_before,
                "val_mean_pred_after": result.val_mean_pred_after,
                "val_mean_probability": result.val_mean_probability,
            }
        )

    return calibrators, pd.DataFrame(calibration_rows)


def apply_calibration(
    calibration_input: pd.DataFrame,
    calibrators: dict[int, LinearRegression],
    method: str,
    min_calibrated_count: float = 0.0,
    max_calibrated_count: float | None = None,
    calibrated_model_suffix: str = "calibrated_loglinear",
) -> pd.DataFrame:
    """Apply horizon-specific calibration to all splits."""

    calibrated_frames: list[pd.DataFrame] = []

    for horizon, horizon_df in calibration_input.groupby("horizon", sort=True):
        horizon_int = int(horizon)
        if horizon_int not in calibrators:
            raise ValueError(f"No calibrator fitted for horizon {horizon_int}.")

        calibrator = calibrators[horizon_int]
        calibrated_df = horizon_df.copy()

        x, _ = _build_calibration_features(calibrated_df, method=method)
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

    required_columns = [
        "trigger_event_id",
        "split",
        "horizon",
        "model_name",
        "y_true",
        "y_pred",
        "y_pred_original",
    ]
    if method != "loglinear":
        required_columns.append("y_prob")

    validate_prediction_frame(
        prediction_df=calibrated_predictions,
        required_columns=required_columns,
        split_col="split",
    )

    return calibrated_predictions


def _save_calibrated_outputs(
    calibrated_predictions: pd.DataFrame,
    calibration_params: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Compute and save calibrated predictions, parameters, metrics, and diagnostics."""

    calibrated_predictions_path = output_dir / f"{output_prefix}_predictions.csv"
    calibration_params_path = output_dir / f"{output_prefix}_parameters.csv"
    calibrated_count_metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    calibrated_capped_metrics_path = output_dir / f"{output_prefix}_capped_metrics.csv"
    calibrated_positive_metrics_path = output_dir / f"{output_prefix}_positive_only_metrics.csv"
    calibrated_bucket_diagnostics_path = output_dir / f"{output_prefix}_bucket_diagnostics.csv"
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


def main() -> None:
    args = parse_args()

    run_paths = build_run_artifact_paths(
        run_name=args.run_name,
        base_output_dir=resolve_repo_path(args.output_dir),
        base_checkpoint_dir=None,
        create_dirs=True,
    )

    calibration_name = args.calibration_name or args.method
    calibrated_suffix = f"calibrated_{calibration_name}"
    output_prefix = f"{args.run_name}_count_{calibrated_suffix}"

    count_predictions_path = run_paths.count_predictions_path
    if not count_predictions_path.exists():
        raise FileNotFoundError(
            f"Count prediction file not found: {count_predictions_path}"
        )

    count_predictions = pd.read_csv(count_predictions_path)
    _validate_count_predictions(count_predictions)
    calibration_input = count_predictions.copy()

    if args.method != "loglinear":
        probability_predictions_path = run_paths.probability_predictions_path
        if not probability_predictions_path.exists():
            raise FileNotFoundError(
                "Probability-aware calibration requires probability predictions, "
                f"but the file was not found: {probability_predictions_path}"
            )
        probability_predictions = pd.read_csv(probability_predictions_path)
        calibration_input = attach_probability_predictions(
            count_predictions=count_predictions,
            probability_predictions=probability_predictions,
        )
        print(f"Loaded probability predictions from: {probability_predictions_path}")

    calibrators, calibration_params = fit_calibrators(
        calibration_input=calibration_input,
        method=args.method,
    )

    calibrated_predictions = apply_calibration(
        calibration_input=calibration_input,
        calibrators=calibrators,
        method=args.method,
        min_calibrated_count=args.min_calibrated_count,
        max_calibrated_count=args.max_calibrated_count,
        calibrated_model_suffix=calibrated_suffix,
    )

    output_dir = run_paths.output_dir
    _save_calibrated_outputs(
        calibrated_predictions=calibrated_predictions,
        calibration_params=calibration_params,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    print(f"Loaded count predictions from: {count_predictions_path}")
    print(f"Calibration method: {args.method}")


if __name__ == "__main__":
    main()
