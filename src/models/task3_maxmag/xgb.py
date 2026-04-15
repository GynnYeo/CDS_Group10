from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(slots=True)
class Task3MaxMagRegressor:
    backend: str = "auto"
    random_state: int = 42
    max_iter: int = 300
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: float = 3.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 3.0
    reg_alpha: float = 0.5
    tune: bool = False
    early_stopping_rounds: int | None = None
    model_: Any = None
    backend_: str | None = None
    best_params_: dict[str, float | int] | None = None
    best_val_metrics_: dict[str, float] | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "Task3MaxMagRegressor":
        backend = self._resolve_backend()
        if backend == "xgboost":
            self.model_ = self._fit_xgboost(X_train, y_train, X_val, y_val)
        else:
            self.model_ = self._fit_gradient_boosting(X_train, y_train)
        self.backend_ = backend
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Task3MaxMagRegressor must be fit before prediction.")
        return np.asarray(self.model_.predict(X), dtype=float)

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "xgboost" if _xgboost_available() else "gb_reg"
        if self.backend not in {"xgboost", "gb_reg"}:
            raise ValueError("backend must be one of: 'auto', 'xgboost', 'gb_reg'.")
        if self.backend == "xgboost" and not _xgboost_available():
            raise ImportError("xgboost is not installed in this environment.")
        return self.backend

    def _fit_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> Any:
        import xgboost as xgb
        if self.tune and X_val is not None and y_val is not None:
            return self._fit_tuned_xgboost(X_train, y_train, X_val, y_val, xgb)
        params = self._current_xgb_params()
        if self.early_stopping_rounds is not None and X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = int(self.early_stopping_rounds)
        model = xgb.XGBRegressor(**params)
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)
        self.best_params_ = params
        if X_val is not None and y_val is not None:
            self.best_val_metrics_ = self._evaluate_regression(y_val, model.predict(X_val))
        return model

    def _fit_tuned_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        xgb_module: Any,
    ) -> Any:
        candidate_grid = {
            "n_estimators": [200, 300, 400],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3],
            "min_child_weight": [3.0, 5.0],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "reg_lambda": [3.0, 5.0],
            "reg_alpha": [0.5, 1.0],
        }
        candidate_params = self._build_candidate_params(candidate_grid)

        best_model = None
        best_rmse = float("inf")
        best_params: dict[str, float | int] | None = None
        best_metrics: dict[str, float] | None = None

        for params in candidate_params:
            if self.early_stopping_rounds is not None:
                params["early_stopping_rounds"] = int(self.early_stopping_rounds)
            model = xgb_module.XGBRegressor(**params)
            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
            model.fit(X_train, y_train, **fit_kwargs)
            val_pred = model.predict(X_val)
            metrics = self._evaluate_regression(y_val, val_pred)
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_params = params
                best_metrics = metrics

        if best_model is None or best_params is None or best_metrics is None:
            raise ValueError("XGBoost tuning failed to produce a fitted model.")

        self.best_params_ = best_params
        self.best_val_metrics_ = best_metrics
        return best_model

    def _current_xgb_params(self) -> dict[str, float | int | str]:
        return {
            "n_estimators": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": self.random_state,
        }

    def _build_candidate_params(
        self,
        candidate_grid: dict[str, list[float | int]],
    ) -> list[dict[str, float | int | str]]:
        candidates: list[dict[str, float | int | str]] = []
        for n_estimators in candidate_grid["n_estimators"]:
            for learning_rate in candidate_grid["learning_rate"]:
                for max_depth in candidate_grid["max_depth"]:
                    for min_child_weight in candidate_grid["min_child_weight"]:
                        for subsample in candidate_grid["subsample"]:
                            for colsample_bytree in candidate_grid["colsample_bytree"]:
                                for reg_lambda in candidate_grid["reg_lambda"]:
                                    for reg_alpha in candidate_grid["reg_alpha"]:
                                        candidates.append(
                                            {
                                                "n_estimators": int(n_estimators),
                                                "learning_rate": float(learning_rate),
                                                "max_depth": int(max_depth),
                                                "min_child_weight": float(min_child_weight),
                                                "subsample": float(subsample),
                                                "colsample_bytree": float(colsample_bytree),
                                                "reg_lambda": float(reg_lambda),
                                                "reg_alpha": float(reg_alpha),
                                                "objective": "reg:squarederror",
                                                "eval_metric": "rmse",
                                                "random_state": self.random_state,
                                            }
                                        )
        return candidates

    def _evaluate_regression(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return {
            "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
            "bias": float(np.mean(y_pred_arr - y_true_arr)),
        }

    def _fit_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
        model = GradientBoostingRegressor(
            n_estimators=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        return model
