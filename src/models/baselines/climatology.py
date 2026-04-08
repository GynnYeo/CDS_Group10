from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


SUPPORTED_HORIZONS = (24, 72)


def _binary_target_col(horizon: int) -> str:
    return f"y_{horizon}h"


@dataclass(slots=True)
class ClimatologyBaseline:
    """
    Constant-probability baseline fit from training labels only.

    This baseline intentionally ignores event-specific covariates and predicts
    the training positive rate for each requested horizon.
    """

    model_name: str = "climatology"
    horizons: tuple[int, ...] = SUPPORTED_HORIZONS
    positive_rates_: dict[int, float] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "ClimatologyBaseline":
        """Fit horizon-specific positive rates from the training split only."""
        self.positive_rates_ = {}
        for horizon in self.horizons:
            target_col = _binary_target_col(horizon)
            if target_col not in train_df.columns:
                raise ValueError(f"train_df is missing required target column: {target_col}")
            self.positive_rates_[horizon] = float(train_df[target_col].mean())
        return self

    def predict_proba(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Return the fitted constant probability for every row."""
        self._validate_horizon(horizon)
        return np.full(len(df), self.positive_rates_[horizon], dtype=float)

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

        prediction_table = pd.DataFrame(
            {
                "split": split_name,
                "horizon": horizon,
                "model_name": self.model_name,
                "y_true": df[target_col].astype(float).to_numpy(),
                "y_prob": self.predict_proba(df, horizon),
                "lambda_pred": np.nan,
            },
            index=df.index,
        )
        if id_col in df.columns:
            prediction_table.insert(0, id_col, df[id_col].to_numpy())

        return prediction_table.reset_index(drop=True)

    def _validate_horizon(self, horizon: int) -> None:
        if horizon not in self.horizons:
            raise ValueError(f"Unsupported horizon: {horizon}. Expected one of {self.horizons}.")
        if horizon not in self.positive_rates_:
            raise ValueError("ClimatologyBaseline must be fit before prediction.")
