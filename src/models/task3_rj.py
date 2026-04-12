from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TASK3_SUPPORTED_HORIZONS = (24, 72)
TRIGGER_MAGNITUDE_COLUMN = "trigger_magnitude"


def _task3_binary_target_col(horizon: int) -> str:
    return f"y_large_{horizon}h"


def _task3_count_target_col(horizon: int) -> str:
    return f"n_large_aftershocks_{horizon}h"


@dataclass(slots=True)
class Task3RJParameters:
    intercept: float
    slope: float
    mean_log_count: float
    n_train: int


@dataclass(slots=True)
class SimplifiedTask3RJBaseline:
    """
    Simplified RJ-style baseline for task 3.

    It fits a global horizon-specific relationship between trigger magnitude
    and the count of larger aftershocks, then converts the expected count into
    the probability of at least one larger aftershock.
    """

    model_name: str = "task3_simplified_rj"
    magnitude_col: str = TRIGGER_MAGNITUDE_COLUMN
    horizons: tuple[int, ...] = TASK3_SUPPORTED_HORIZONS
    params_: dict[int, Task3RJParameters] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "SimplifiedTask3RJBaseline":
        if self.magnitude_col not in train_df.columns:
            raise ValueError(
                f"train_df is missing required magnitude column: {self.magnitude_col}"
            )

        self.params_ = {}
        for horizon in self.horizons:
            count_col = _task3_count_target_col(horizon)
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

            self.params_[horizon] = Task3RJParameters(
                intercept=intercept,
                slope=slope,
                mean_log_count=float(log_counts.mean()),
                n_train=int(len(fit_df)),
            )

        return self

    def predict_lambda(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
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
        lambda_pred = self.predict_lambda(df, horizon)
        return 1.0 - np.exp(-lambda_pred)

    def predict_table(
        self,
        df: pd.DataFrame,
        split_name: str,
        horizon: int,
        id_col: str = "trigger_event_id",
    ) -> pd.DataFrame:
        self._validate_horizon(horizon)
        target_col = _task3_binary_target_col(horizon)
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

    def _validate_horizon(self, horizon: int) -> Task3RJParameters:
        if horizon not in self.horizons:
            raise ValueError(f"Unsupported horizon: {horizon}. Expected one of {self.horizons}.")
        if horizon not in self.params_:
            raise ValueError("SimplifiedTask3RJBaseline must be fit before prediction.")
        return self.params_[horizon]
