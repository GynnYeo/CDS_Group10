from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


SUPPORTED_HORIZONS = (24, 72)
TRIGGER_MAGNITUDE_COLUMN = "trigger_magnitude"


def _binary_target_col(horizon: int) -> str:
    return f"y_{horizon}h"


def _count_target_col(horizon: int) -> str:
    return f"n_aftershocks_{horizon}h"


@dataclass(slots=True)
class RJHorizonParameters:
    """Train-fitted parameters for one RJ-style horizon model."""

    intercept: float
    slope: float
    mean_log_count: float
    n_train: int


@dataclass(slots=True)
class SimplifiedRJBaseline:
    """
    Simplified RJ-style baseline fit globally on the training split.

    This is intentionally not a full operational Reasenberg-Jones model. The
    implementation fits a separate horizon-specific relationship:

    log(1 + lambda_h) = intercept_h + slope_h * trigger_magnitude

    where lambda_h is the expected aftershock count in the horizon. Predicted
    counts are transformed to probabilities via:

    p_h = 1 - exp(-lambda_h)
    """

    model_name: str = "simplified_rj"
    magnitude_col: str = TRIGGER_MAGNITUDE_COLUMN
    horizons: tuple[int, ...] = SUPPORTED_HORIZONS
    params_: dict[int, RJHorizonParameters] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "RJBaseline":
        """Fit horizon-specific log-count regressions on the training split only."""
        if self.magnitude_col not in train_df.columns:
            raise ValueError(
                f"train_df is missing required magnitude column: {self.magnitude_col}"
            )

        self.params_ = {}
        for horizon in self.horizons:
            count_col = _count_target_col(horizon)
            if count_col not in train_df.columns:
                raise ValueError(f"train_df is missing required count target column: {count_col}")

            fit_df = train_df[[self.magnitude_col, count_col]].dropna().copy()
            if fit_df.empty:
                raise ValueError(f"No non-missing training rows available for horizon {horizon}h.")
            if (fit_df[count_col] < 0).any():
                raise ValueError(f"Count target column '{count_col}' must be non-negative.")

            x = fit_df[self.magnitude_col].astype(float).to_numpy()
            log_counts = np.log1p(fit_df[count_col].astype(float).to_numpy())

            if len(np.unique(x)) < 2:
                intercept = float(log_counts.mean())
                slope = 0.0
            else:
                design = np.column_stack([np.ones_like(x), x])
                coefficients, *_ = np.linalg.lstsq(design, log_counts, rcond=None)
                intercept = float(coefficients[0])
                slope = float(coefficients[1])

            self.params_[horizon] = RJHorizonParameters(
                intercept=intercept,
                slope=slope,
                mean_log_count=float(log_counts.mean()),
                n_train=int(len(fit_df)),
            )

        return self

    def predict_lambda(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predict expected aftershock counts for one horizon."""
        params = self._validate_horizon(horizon)
        if self.magnitude_col not in df.columns:
            raise ValueError(
                f"df is missing required magnitude column: {self.magnitude_col}"
            )
        if df[self.magnitude_col].isna().any():
            raise ValueError(
                f"df contains missing values in required magnitude column: {self.magnitude_col}"
            )

        magnitude = df[self.magnitude_col].astype(float).to_numpy()
        linear_term = params.intercept + params.slope * magnitude
        linear_term = np.clip(linear_term, a_min=-20.0, a_max=20.0)
        lambda_pred = np.expm1(linear_term)
        return np.clip(lambda_pred, a_min=0.0, a_max=None)

    def predict_proba(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Convert predicted expected counts to event probabilities."""
        lambda_pred = self.predict_lambda(df, horizon)
        return 1.0 - np.exp(-lambda_pred)

    def predict_table(
        self,
        df: pd.DataFrame,
        split_name: str,
        horizon: int,
        id_col: str = "trigger_event_id",
    ) -> pd.DataFrame:
        """Return standardized long-form prediction rows for one split/horizon."""
        self._validate_horizon(horizon)
        target_col = _binary_target_col(horizon)
        if target_col not in df.columns:
            raise ValueError(f"df is missing required target column: {target_col}")

        lambda_pred = self.predict_lambda(df, horizon)
        prediction_table = pd.DataFrame(
            {
                "split": split_name,
                "horizon": horizon,
                "model_name": self.model_name,
                "y_true": df[target_col].astype(float).to_numpy(),
                "y_prob": 1.0 - np.exp(-lambda_pred),
                "lambda_pred": lambda_pred,
            },
            index=df.index,
        )
        if id_col in df.columns:
            prediction_table.insert(0, id_col, df[id_col].to_numpy())

        return prediction_table.reset_index(drop=True)

    def _validate_horizon(self, horizon: int) -> RJHorizonParameters:
        if horizon not in self.horizons:
            raise ValueError(f"Unsupported horizon: {horizon}. Expected one of {self.horizons}.")
        if horizon not in self.params_:
            raise ValueError("SimplifiedRJBaseline must be fit before prediction.")
        return self.params_[horizon]
