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


@dataclass(slots=True)
class Task3BinaryModel:
    """
    Binary classifier wrapper for task 3.

    It prefers XGBoost when installed and falls back to sklearn's gradient
    boosting classifier so the sidecar pipeline remains runnable in a minimal
    environment.
    """

    backend: str = "auto"
    random_state: int = 42
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
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
        else:
            self.model_ = self._fit_hist_gb(X_train, y_train)
        self.backend_ = backend
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Task3BinaryModel must be fit before prediction.")

        if hasattr(self.model_, "predict_proba"):
            probs = self.model_.predict_proba(X)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs

        logits = self.model_.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "xgboost" if _xgboost_available() else "hist_gb"
        if self.backend not in {"xgboost", "hist_gb"}:
            raise ValueError("backend must be one of: 'auto', 'xgboost', 'hist_gb'.")
        if self.backend == "xgboost" and not _xgboost_available():
            raise ImportError(
                "xgboost is not installed in this environment. "
                "Install it or use backend='hist_gb'."
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
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
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

    def _fit_hist_gb(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        model = GradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        return model
