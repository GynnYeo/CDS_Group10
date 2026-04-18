from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


SUPPORTED_HORIZONS = (24, 72)
TRIGGER_MAGNITUDE_COLUMN = "trigger_magnitude"


def _binary_target_col(horizon: int) -> str:
    return f"y_{horizon}h"


@dataclass(slots=True)
class MagnitudeBinnedBaseline:
    """
    Baseline that predicts horizon-specific positive rates by trigger magnitude bin.

    Bins are learned from training-set magnitude quantiles and each bin predicts the
    observed training positive rate for that horizon. Sparse or missing-bin cases
    fall back to the global training positive rate.
    """

    model_name: str = "magnitude_binned_climatology"
    magnitude_col: str = TRIGGER_MAGNITUDE_COLUMN
    horizons: tuple[int, ...] = SUPPORTED_HORIZONS
    n_bins: int = 5
    min_bin_count: int = 30
    bin_edges_: np.ndarray | None = None
    global_positive_rates_: dict[int, float] = field(default_factory=dict)
    bin_positive_rates_: dict[int, dict[int, float]] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "MagnitudeBinnedBaseline":
        """Fit magnitude bins and horizon-specific positive rates on the train split."""
        if self.magnitude_col not in train_df.columns:
            raise ValueError(
                f"train_df is missing required magnitude column: {self.magnitude_col}"
            )

        magnitude = train_df[self.magnitude_col].dropna().astype(float)
        if magnitude.empty:
            raise ValueError(
                f"train_df has no non-missing values in magnitude column: {self.magnitude_col}"
            )

        self.bin_edges_ = self._compute_bin_edges(magnitude.to_numpy())
        self.global_positive_rates_ = {}
        self.bin_positive_rates_ = {}

        for horizon in self.horizons:
            target_col = _binary_target_col(horizon)
            if target_col not in train_df.columns:
                raise ValueError(f"train_df is missing required target column: {target_col}")

            self.global_positive_rates_[horizon] = float(train_df[target_col].mean())

            fit_df = train_df[[self.magnitude_col, target_col]].dropna().copy()
            bin_codes = pd.cut(
                fit_df[self.magnitude_col].astype(float),
                bins=self.bin_edges_,
                include_lowest=True,
                labels=False,
            )
            fit_df["bin_code"] = bin_codes

            grouped = fit_df.dropna(subset=["bin_code"]).groupby("bin_code")[target_col]
            stats = grouped.agg(["mean", "count"]).reset_index()

            rate_by_bin: dict[int, float] = {}
            for row in stats.itertuples(index=False):
                bin_code = int(row.bin_code)
                if int(row.count) >= self.min_bin_count:
                    rate_by_bin[bin_code] = float(row.mean)

            self.bin_positive_rates_[horizon] = rate_by_bin

        return self

    def predict_proba(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predict event probability for one horizon using fitted bin rates."""
        self._validate_horizon(horizon)
        if self.magnitude_col not in df.columns:
            raise ValueError(f"df is missing required magnitude column: {self.magnitude_col}")

        probs = np.full(len(df), self.global_positive_rates_[horizon], dtype=float)
        magnitude = df[self.magnitude_col].astype(float)
        bin_codes = pd.cut(
            magnitude,
            bins=self.bin_edges_,
            include_lowest=True,
            labels=False,
        )

        for bin_code, rate in self.bin_positive_rates_[horizon].items():
            probs[bin_codes == bin_code] = rate

        return probs

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
        if horizon not in self.global_positive_rates_:
            raise ValueError("MagnitudeBinnedBaseline must be fit before prediction.")
        if self.bin_edges_ is None:
            raise ValueError("MagnitudeBinnedBaseline must be fit before prediction.")

    def _compute_bin_edges(self, magnitude_values: np.ndarray) -> np.ndarray:
        n_bins = max(2, int(self.n_bins))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(magnitude_values, quantiles)
        edges = np.unique(edges.astype(float))

        if len(edges) < 2:
            return np.array([-np.inf, np.inf], dtype=float)

        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges