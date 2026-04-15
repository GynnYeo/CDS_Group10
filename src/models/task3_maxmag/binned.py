from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss


BIN_EDGES = (0.0, 4.5, 5.0, 5.5, 10.0)
BIN_LABELS = ("<4.5", "4.5-5.0", "5.0-5.5", ">=5.5")


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


def binned_target_col(horizon: int) -> str:
    return f"max_aftershock_mag_bin_{horizon}h"


def build_binned_targets(df: pd.DataFrame, horizons: tuple[int, ...] = (24, 72)) -> pd.DataFrame:
    out = df.copy()
    for horizon in horizons:
        source_col = f"max_aftershock_mag_{horizon}h"
        target_col = binned_target_col(horizon)
        cats = pd.cut(
            out[source_col],
            bins=BIN_EDGES,
            labels=False,
            right=False,
            include_lowest=True,
        )
        out[target_col] = cats.astype("float")
    return out


def evaluate_multiclass_predictions(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    proba_arr = np.asarray(proba, dtype=float)
    pred = np.argmax(proba_arr, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true_arr, pred)),
        "macro_f1": float(f1_score(y_true_arr, pred, average="macro")),
        "log_loss": float(log_loss(y_true_arr, proba_arr, labels=list(range(proba_arr.shape[1])))),
    }


@dataclass(slots=True)
class MajorityClassBaseline:
    n_classes: int = len(BIN_LABELS)
    class_proba_: np.ndarray | None = None

    def fit(self, y_train: pd.Series) -> "MajorityClassBaseline":
        counts = (
            pd.Series(y_train)
            .astype(int)
            .value_counts(normalize=True)
            .reindex(range(self.n_classes), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        self.class_proba_ = counts
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.class_proba_ is None:
            raise ValueError("MajorityClassBaseline must be fit before prediction.")
        return np.tile(self.class_proba_, (len(X), 1))


@dataclass(slots=True)
class BinnedSeverityModel:
    backend: str = "auto"
    random_state: int = 42
    n_classes: int = len(BIN_LABELS)
    max_iter: int = 300
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: float = 3.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 3.0
    reg_alpha: float = 0.5
    model_: Any = None
    backend_: str | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BinnedSeverityModel":
        backend = self._resolve_backend()
        if backend == "xgboost":
            self.model_ = self._fit_xgboost(X_train, y_train)
        else:
            self.model_ = self._fit_logistic(X_train, y_train)
        self.backend_ = backend
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("BinnedSeverityModel must be fit before prediction.")
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "xgboost" if _xgboost_available() else "logreg"
        if self.backend not in {"xgboost", "logreg"}:
            raise ValueError("backend must be one of: 'auto', 'xgboost', 'logreg'.")
        if self.backend == "xgboost" and not _xgboost_available():
            raise ImportError("xgboost is not installed in this environment.")
        return self.backend

    def _fit_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            objective="multi:softprob",
            num_class=self.n_classes,
            eval_metric="mlogloss",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train.astype(int))
        return model

    def _fit_logistic(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        model = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train.astype(int))
        return model
