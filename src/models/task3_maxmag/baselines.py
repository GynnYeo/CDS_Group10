from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


TASK3_MAXMAG_HORIZONS = (24, 72)
TRIGGER_MAGNITUDE_COLUMN = "trigger_magnitude"
LINEAR_BASELINE_FEATURES = [
    "trigger_magnitude",
    "trigger_depth_km",
    "prior_global_event_count_24h",
    "prior_global_event_count_7d",
]
MAG_ONLY_LINEAR_FEATURES = [
    "trigger_magnitude",
]


def _target_col(horizon: int) -> str:
    return f"max_aftershock_mag_{horizon}h"


@dataclass(slots=True)
class MeanMaxMagnitudeBaseline:
    model_name: str = "task3_maxmag_mean"
    horizons: tuple[int, ...] = TASK3_MAXMAG_HORIZONS
    mean_targets_: dict[int, float] | None = None

    def fit(self, train_df: pd.DataFrame) -> "MeanMaxMagnitudeBaseline":
        self.mean_targets_ = {}
        for horizon in self.horizons:
            target = _target_col(horizon)
            if target not in train_df.columns:
                raise ValueError(f"Missing target column: {target}")
            fit_series = train_df[target].dropna()
            if fit_series.empty:
                raise ValueError(f"No non-missing rows available for horizon {horizon}h.")
            self.mean_targets_[horizon] = float(fit_series.mean())
        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        if self.mean_targets_ is None or horizon not in self.mean_targets_:
            raise ValueError("MeanMaxMagnitudeBaseline must be fit before prediction.")
        return np.full(len(df), self.mean_targets_[horizon], dtype=float)

    def predict_table(
        self,
        df: pd.DataFrame,
        split_name: str,
        horizon: int,
        id_col: str = "trigger_event_id",
    ) -> pd.DataFrame:
        target = _target_col(horizon)
        eval_df = df.loc[df[target].notna()].copy()
        prediction_table = pd.DataFrame(
            {
                "split": split_name,
                "horizon": horizon,
                "model_name": self.model_name,
                "y_true": eval_df[target].astype(float).to_numpy(),
                "y_pred": self.predict(eval_df, horizon),
            },
            index=eval_df.index,
        )
        if id_col in eval_df.columns:
            prediction_table.insert(0, id_col, eval_df[id_col].to_numpy())
        return prediction_table.reset_index(drop=True)


@dataclass(slots=True)
class BathsLawBaseline:
    model_name: str = "task3_maxmag_baths_law"
    magnitude_col: str = TRIGGER_MAGNITUDE_COLUMN
    delta_m: float = 1.2
    horizons: tuple[int, ...] = TASK3_MAXMAG_HORIZONS

    def fit(self, train_df: pd.DataFrame) -> "BathsLawBaseline":
        if self.magnitude_col not in train_df.columns:
            raise ValueError(f"Missing magnitude column: {self.magnitude_col}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.magnitude_col not in df.columns:
            raise ValueError(f"Missing magnitude column: {self.magnitude_col}")
        return df[self.magnitude_col].astype(float).to_numpy() - self.delta_m

    def predict_table(
        self,
        df: pd.DataFrame,
        split_name: str,
        horizon: int,
        id_col: str = "trigger_event_id",
    ) -> pd.DataFrame:
        target = _target_col(horizon)
        eval_df = df.loc[df[target].notna()].copy()
        prediction_table = pd.DataFrame(
            {
                "split": split_name,
                "horizon": horizon,
                "model_name": self.model_name,
                "y_true": eval_df[target].astype(float).to_numpy(),
                "y_pred": self.predict(eval_df),
            },
            index=eval_df.index,
        )
        if id_col in eval_df.columns:
            prediction_table.insert(0, id_col, eval_df[id_col].to_numpy())
        return prediction_table.reset_index(drop=True)


@dataclass(slots=True)
class LinearMaxMagnitudeBaseline:
    """
    Small classical regression baseline for max-aftershock magnitude.

    This keeps the comparison interpretable by fitting one linear model per
    horizon on a short list of numeric trigger-level predictors.
    """

    model_name: str = "task3_maxmag_linear"
    feature_cols: list[str] | None = None
    horizons: tuple[int, ...] = TASK3_MAXMAG_HORIZONS
    models_: dict[int, LinearRegression] | None = None
    imputers_: dict[int, SimpleImputer] | None = None
    fitted_feature_cols_: list[str] | None = None

    def fit(self, train_df: pd.DataFrame) -> "LinearMaxMagnitudeBaseline":
        requested_features = self.feature_cols or list(LINEAR_BASELINE_FEATURES)
        missing_features = [column for column in requested_features if column not in train_df.columns]
        if missing_features:
            raise ValueError(f"Missing linear-baseline feature columns: {missing_features}")

        self.models_ = {}
        self.imputers_ = {}
        self.fitted_feature_cols_ = list(requested_features)
        for horizon in self.horizons:
            target = _target_col(horizon)
            if target not in train_df.columns:
                raise ValueError(f"Missing target column: {target}")

            fit_df = train_df.loc[train_df[target].notna(), self.fitted_feature_cols_ + [target]].copy()
            if fit_df.empty:
                raise ValueError(f"No non-missing rows available for horizon {horizon}h.")

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(fit_df[self.fitted_feature_cols_])
            y_train = fit_df[target].astype(float).to_numpy()

            model = LinearRegression()
            model.fit(X_train, y_train)

            self.imputers_[horizon] = imputer
            self.models_[horizon] = model

        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        if self.models_ is None or self.imputers_ is None or self.fitted_feature_cols_ is None:
            raise ValueError("LinearMaxMagnitudeBaseline must be fit before prediction.")
        if horizon not in self.models_:
            raise ValueError(f"Unsupported horizon: {horizon}")

        missing_features = [column for column in self.fitted_feature_cols_ if column not in df.columns]
        if missing_features:
            raise ValueError(f"Missing linear-baseline feature columns: {missing_features}")

        X = self.imputers_[horizon].transform(df[self.fitted_feature_cols_])
        return self.models_[horizon].predict(X)

    def predict_table(
        self,
        df: pd.DataFrame,
        split_name: str,
        horizon: int,
        id_col: str = "trigger_event_id",
    ) -> pd.DataFrame:
        target = _target_col(horizon)
        eval_df = df.loc[df[target].notna()].copy()
        prediction_table = pd.DataFrame(
            {
                "split": split_name,
                "horizon": horizon,
                "model_name": self.model_name,
                "y_true": eval_df[target].astype(float).to_numpy(),
                "y_pred": self.predict(eval_df, horizon),
            },
            index=eval_df.index,
        )
        if id_col in eval_df.columns:
            prediction_table.insert(0, id_col, eval_df[id_col].to_numpy())
        return prediction_table.reset_index(drop=True)


@dataclass(slots=True)
class MagnitudeOnlyLinearMaxMagnitudeBaseline(LinearMaxMagnitudeBaseline):
    """
    Simpler one-feature linear regression baseline using trigger magnitude only.
    """

    model_name: str = "task3_maxmag_linear_magonly"
    feature_cols: list[str] | None = None

    def fit(self, train_df: pd.DataFrame) -> "MagnitudeOnlyLinearMaxMagnitudeBaseline":
        self.feature_cols = list(MAG_ONLY_LINEAR_FEATURES)
        return super().fit(train_df)
