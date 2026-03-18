from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


DATA_PATH = "data/processed/comcat/baseline_mainshock_model_dataset_2015_2024.parquet"
MODEL_DIR = Path("models/comcat/two_stage")
METRICS_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

feature_cols = [
    "mainshock_latitude",
    "mainshock_longitude",
    "mainshock_depth_km",
    "mainshock_magnitude",
    "mainshock_gap",
    "mainshock_dmin",
    "mainshock_rms",
]

target_cls_col = "has_aftershock_within_720h"
target_reg_col = "target_hours_raw"


def build_numeric_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_columns),
        ]
    )
    return preprocessor


def evaluate_classifier(y_true: pd.Series, pred_label: np.ndarray, pred_proba: np.ndarray) -> dict:
    return {
        "cls_accuracy": float(accuracy_score(y_true, pred_label)),
        "cls_precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "cls_recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "cls_f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "cls_auc": float(roc_auc_score(y_true, pred_proba)),
    }


def evaluate_regressor(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    df = pd.read_parquet(DATA_PATH).copy()
    df["mainshock_time"] = pd.to_datetime(df["mainshock_time"], utc=True, errors="coerce")
    df = df.sort_values("mainshock_time").reset_index(drop=True)

    # Chronological split
    train_df = df[df["mainshock_time"].dt.year <= 2023].copy()
    valid_df = df[df["mainshock_time"].dt.year == 2024].copy()

    # Stage 1 data: all rows
    X_train_cls = train_df[feature_cols]
    y_train_cls = train_df[target_cls_col].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid_cls = valid_df[target_cls_col].astype(int)

    # Stage 2 data: only rows where aftershock actually occurs within 720h
    train_pos_df = train_df[train_df[target_cls_col] == True].copy()
    valid_pos_df = valid_df[valid_df[target_cls_col] == True].copy()

    X_train_reg = train_pos_df[feature_cols]
    y_train_reg = train_pos_df[target_reg_col]

    X_valid_reg_truepos = valid_pos_df[feature_cols]
    y_valid_reg_truepos = valid_pos_df[target_reg_col]

    print("Train rows:", len(train_df))
    print("Valid rows:", len(valid_df))
    print("Train positive rows:", len(train_pos_df))
    print("Valid positive rows:", len(valid_pos_df))

    preprocessor_cls = build_numeric_preprocessor(feature_cols)
    preprocessor_reg = build_numeric_preprocessor(feature_cols)

    classifier_models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    regressor_models = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for cls_name, classifier in classifier_models.items():
        cls_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_cls),
                ("model", classifier),
            ]
        )

        cls_pipeline.fit(X_train_cls, y_train_cls)

        valid_cls_pred = cls_pipeline.predict(X_valid)
        valid_cls_proba = cls_pipeline.predict_proba(X_valid)[:, 1]

        cls_metrics = evaluate_classifier(
            y_true=y_valid_cls,
            pred_label=valid_cls_pred,
            pred_proba=valid_cls_proba,
        )

        joblib.dump(cls_pipeline, MODEL_DIR / f"{cls_name}.joblib")

        for reg_name, regressor in regressor_models.items():
            reg_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_reg),
                    ("model", regressor),
                ]
            )

            reg_pipeline.fit(X_train_reg, y_train_reg)

            # Stage 2 standalone timing performance on true positive validation rows
            valid_reg_pred_truepos = reg_pipeline.predict(X_valid_reg_truepos)
            valid_reg_pred_truepos = np.clip(valid_reg_pred_truepos, 0, 720)

            reg_metrics = evaluate_regressor(
                y_true=y_valid_reg_truepos,
                y_pred=valid_reg_pred_truepos,
                prefix="stage2_truepos",
            )

            # Combined final prediction on all validation rows
            combined_pred = np.full(len(valid_df), 720.0)

            # For rows predicted positive by classifier, use regressor prediction
            pred_positive_mask = valid_cls_pred == 1
            if pred_positive_mask.sum() > 0:
                reg_pred_for_positive = reg_pipeline.predict(X_valid.loc[pred_positive_mask, feature_cols])
                reg_pred_for_positive = np.clip(reg_pred_for_positive, 0, 720)
                combined_pred[pred_positive_mask] = reg_pred_for_positive

            y_valid_final = valid_df["target_hours_capped"].values

            final_metrics = evaluate_regressor(
                y_true=y_valid_final,
                y_pred=combined_pred,
                prefix="combined",
            )

            # Optional: combined threshold classification check
            combined_cls_pred = (combined_pred < 720).astype(int)
            combined_cls_metrics = {
                "combined_accuracy_within_720h": float(accuracy_score(y_valid_cls, combined_cls_pred)),
                "combined_precision_within_720h": float(precision_score(y_valid_cls, combined_cls_pred, zero_division=0)),
                "combined_recall_within_720h": float(recall_score(y_valid_cls, combined_cls_pred, zero_division=0)),
                "combined_f1_within_720h": float(f1_score(y_valid_cls, combined_cls_pred, zero_division=0)),
            }

            result_row = {
                "classifier_model": cls_name,
                "regressor_model": reg_name,
                "n_train": int(len(train_df)),
                "n_valid": int(len(valid_df)),
                "n_train_positive": int(len(train_pos_df)),
                "n_valid_positive": int(len(valid_pos_df)),
            }
            result_row.update(cls_metrics)
            result_row.update(reg_metrics)
            result_row.update(final_metrics)
            result_row.update(combined_cls_metrics)

            results.append(result_row)

            joblib.dump(reg_pipeline, MODEL_DIR / f"{cls_name}__{reg_name}.joblib")

    results_df = pd.DataFrame(results).sort_values(
        by=["combined_mae", "combined_rmse"]
    )

    print("\nTwo-stage results:")
    print(results_df.to_string(index=False))

    results_df.to_csv(METRICS_DIR / "two_stage_results.csv", index=False)

    with open(METRICS_DIR / "two_stage_results.json", "w") as f:
        json.dump(results, f, indent=2)

    best_row = results_df.iloc[0].to_dict()
    print("\nBest combination:")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()