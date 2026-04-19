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


def _tabnet_available() -> bool:
    try:
        import pytorch_tabnet  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(slots=True)
class Task3MaxMagRegressor:
    backend: str = "auto"
    random_state: int = 42
    max_iter: int = 300
    tabnet_max_epochs: int | None = None
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: float = 3.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 3.0
    reg_alpha: float = 0.5
    tune: bool = False
    early_stopping_rounds: int | None = None
    tabnet_patience: int = 20
    tabnet_batch_size: int = 1024
    tabnet_virtual_batch_size: int = 128
    tabnet_lr: float = 0.02
    tabnet_n_d: int = 8
    tabnet_n_a: int = 8
    tabnet_n_steps: int = 3
    tabnet_gamma: float = 1.3
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
        elif backend == "tabnet":
            self.model_ = self._fit_tabnet(X_train, y_train, X_val, y_val)
        else:
            self.model_ = self._fit_gradient_boosting(X_train, y_train)
        self.backend_ = backend
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Task3MaxMagRegressor must be fit before prediction.")
        X_input: pd.DataFrame | np.ndarray = X
        if self.backend_ == "tabnet":
            X_input = X.to_numpy(dtype=np.float32)
        return np.asarray(self.model_.predict(X_input), dtype=float).ravel()

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            if _xgboost_available():
                return "xgboost"
            if _tabnet_available():
                return "tabnet"
            return "gb_reg"
        if self.backend not in {"xgboost", "gb_reg", "tabnet"}:
            raise ValueError("backend must be one of: 'auto', 'xgboost', 'gb_reg', 'tabnet'.")
        if self.backend == "xgboost" and not _xgboost_available():
            raise ImportError("xgboost is not installed in this environment.")
        if self.backend == "tabnet" and not _tabnet_available():
            raise ImportError("pytorch-tabnet is not installed in this environment.")
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

    def _fit_tabnet(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> Any:
        from pytorch_tabnet.tab_model import TabNetRegressor

        model = TabNetRegressor(
            n_d=self.tabnet_n_d,
            n_a=self.tabnet_n_a,
            n_steps=self.tabnet_n_steps,
            gamma=self.tabnet_gamma,
            optimizer_params={"lr": self.tabnet_lr},
            seed=self.random_state,
            verbose=0,
        )
        fit_kwargs: dict[str, Any] = {
            "X_train": X_train.to_numpy(dtype=np.float32),
            "y_train": y_train.to_numpy(dtype=np.float32).reshape(-1, 1),
            "max_epochs": self.tabnet_max_epochs or self.max_iter,
            "patience": self.tabnet_patience,
            "batch_size": self.tabnet_batch_size,
            "virtual_batch_size": self.tabnet_virtual_batch_size,
            "num_workers": 0,
            "drop_last": False,
        }
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [
                (
                    X_val.to_numpy(dtype=np.float32),
                    y_val.to_numpy(dtype=np.float32).reshape(-1, 1),
                )
            ]
            fit_kwargs["eval_name"] = ["val"]
            fit_kwargs["eval_metric"] = ["rmse"]
        model.fit(**fit_kwargs)
        self.best_params_ = {
            "backend": "tabnet",
            "max_epochs": self.tabnet_max_epochs or self.max_iter,
            "patience": self.tabnet_patience,
            "batch_size": self.tabnet_batch_size,
            "virtual_batch_size": self.tabnet_virtual_batch_size,
            "lr": self.tabnet_lr,
            "n_d": self.tabnet_n_d,
            "n_a": self.tabnet_n_a,
            "n_steps": self.tabnet_n_steps,
            "gamma": self.tabnet_gamma,
            "seed": self.random_state,
        }
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val.to_numpy(dtype=np.float32)).ravel()
            self.best_val_metrics_ = self._evaluate_regression(y_val, val_pred)
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
