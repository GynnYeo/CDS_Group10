from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


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
class Task3BinaryModel:
    """
    Binary classifier wrapper for task 3.

    It prefers XGBoost when installed and falls back to sklearn's gradient
    boosting classifier so the sidecar pipeline remains runnable in a minimal
    environment. TabNet is also supported as an optional tabular deep-learning
    backend when `pytorch-tabnet` is installed.
    """

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

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "Task3BinaryModel":
        backend = self._resolve_backend()
        if backend == "xgboost":
            self.model_ = self._fit_xgboost(X_train, y_train, X_val, y_val)
        elif backend == "tabnet":
            self.model_ = self._fit_tabnet(X_train, y_train, X_val, y_val)
        else:
            self.model_ = self._fit_hist_gb(X_train, y_train)
        self.backend_ = backend
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Task3BinaryModel must be fit before prediction.")

        X_input: pd.DataFrame | np.ndarray = X
        if self.backend_ == "tabnet":
            X_input = X.to_numpy(dtype=np.float32)

        if hasattr(self.model_, "predict_proba"):
            probs = self.model_.predict_proba(X_input)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs

        logits = self.model_.decision_function(X_input)
        return 1.0 / (1.0 + np.exp(-logits))

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            if _xgboost_available():
                return "xgboost"
            if _tabnet_available():
                return "tabnet"
            return "hist_gb"
        if self.backend not in {"xgboost", "hist_gb", "tabnet"}:
            raise ValueError("backend must be one of: 'auto', 'xgboost', 'hist_gb', 'tabnet'.")
        if self.backend == "xgboost" and not _xgboost_available():
            raise ImportError(
                "xgboost is not installed in this environment. "
                "Install it or use backend='hist_gb'."
            )
        if self.backend == "tabnet" and not _tabnet_available():
            raise ImportError(
                "pytorch-tabnet is not installed in this environment. "
                "Install it or use backend='xgboost' or 'hist_gb'."
            )
        return self.backend

    def _fit_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> Any:
        import xgboost as xgb

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        positive_rate = float(np.mean(y_train))
        scale_pos_weight = 1.0
        if 0.0 < positive_rate < 1.0:
            scale_pos_weight = float((1.0 - positive_rate) / positive_rate)

        model = xgb.XGBClassifier(
            n_estimators=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
        )
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def _fit_tabnet(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> Any:
        from pytorch_tabnet.tab_model import TabNetClassifier

        model = TabNetClassifier(
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
            "y_train": y_train.to_numpy(dtype=np.int64),
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
                    y_val.to_numpy(dtype=np.int64),
                )
            ]
            fit_kwargs["eval_name"] = ["val"]
            fit_kwargs["eval_metric"] = ["auc"]
        model.fit(**fit_kwargs)
        return model

    def _fit_hist_gb(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        model = GradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        return model
