from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = "data/processed/comcat/baseline_mainshock_model_dataset_2015_2024.parquet"
MODEL_DIR = Path("models/comcat")
METRICS_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

target_col = "target_hours_capped"

feature_cols = [
    "mainshock_latitude",
    "mainshock_longitude",
    "mainshock_depth_km",
    "mainshock_magnitude",
    "mainshock_gap",
    "mainshock_dmin",
    "mainshock_rms",
]

numeric_features = feature_cols.copy()

df = pd.read_parquet(DATA_PATH).copy()
df["mainshock_time"] = pd.to_datetime(df["mainshock_time"], utc=True, errors="coerce")
df = df.sort_values("mainshock_time").reset_index(drop=True)

# Use mainshock_time for chronological split
train_df = df[df["mainshock_time"].dt.year <= 2023].copy()
valid_df = df[df["mainshock_time"].dt.year == 2024].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_valid = valid_df[feature_cols]
y_valid = valid_df[target_col]

# handle missing values with median value
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ]
)

models = {
    "dummy_median": DummyRegressor(strategy="median"),
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for model_name, regressor in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    preds = np.clip(preds, 0, 720)

    # A. Overall capped regression metrics
    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    # B. Occurrence classification at 720h threshold
    # 1 = aftershock within 720h, 0 = no aftershock within 720h
    y_valid_cls = (y_valid < 720).astype(int)
    preds_cls = (preds < 720).astype(int)

    accuracy_720 = accuracy_score(y_valid_cls, preds_cls)
    precision_720 = precision_score(y_valid_cls, preds_cls, zero_division=0)
    recall_720 = recall_score(y_valid_cls, preds_cls, zero_division=0)
    f1_720 = f1_score(y_valid_cls, preds_cls, zero_division=0)

    # C. Conditional timing regression on true aftershock cases only
    mask_true_aftershock = y_valid < 720

    conditional_mae = mean_absolute_error(
        y_valid[mask_true_aftershock],
        preds[mask_true_aftershock],
    )
    conditional_rmse = np.sqrt(
        mean_squared_error(
            y_valid[mask_true_aftershock],
            preds[mask_true_aftershock],
        )
    )

    results.append(
        {
            "model": model_name,
            "mae": float(mae),
            "rmse": float(rmse),
            "accuracy_within_720h": float(accuracy_720),
            "precision_within_720h": float(precision_720),
            "recall_within_720h": float(recall_720),
            "f1_within_720h": float(f1_720),
            "conditional_mae_true_aftershock": float(conditional_mae),
            "conditional_rmse_true_aftershock": float(conditional_rmse),
            "n_train": int(len(train_df)),
            "n_valid": int(len(valid_df)),
            "n_valid_true_aftershock": int(mask_true_aftershock.sum()),
        }
    )

    joblib.dump(pipeline, MODEL_DIR / f"{model_name}.joblib")

results_df = pd.DataFrame(results).sort_values("mae")
print(results_df)

results_df.to_csv(METRICS_DIR / "baseline_results.csv", index=False)

with open(METRICS_DIR / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)