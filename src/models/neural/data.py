from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models.feature_sets import NN_CORE_V1, NN_ENRICHED_V1, NN_ENRICHED_V2
from src.models.input_layer import (
    EXPECTED_SPLITS,
    InputConfig,
    TabularInputLayer,
    load_modeling_splits,
    resolve_feature_columns,
)
from src.utils.paths import PROCESSED_DATASETS_DIR, PROCESSED_SPLITS_DIR


PROBABILITY_TARGET_COLUMNS = ["y_24h", "y_72h"]
COUNT_TARGET_COLUMNS = ["n_aftershocks_24h", "n_aftershocks_72h"]
MAGNITUDE_TARGET_COLUMNS = [
    "max_aftershock_magnitude_24h",
    "max_aftershock_magnitude_72h",
]

ALL_NEURAL_TARGET_COLUMNS = (
    PROBABILITY_TARGET_COLUMNS
    + COUNT_TARGET_COLUMNS
    + MAGNITUDE_TARGET_COLUMNS
)

_NEURAL_FEATURE_SET_REGISTRY = {
    "nn_core_v1": NN_CORE_V1,
    "nn_enriched_v1": NN_ENRICHED_V1,
    "nn_enriched_v2": NN_ENRICHED_V2,   # V1 + 35 engineered features (Task 1)
}


@dataclass(slots=True)
class PreparedNeuralInputs:
    """Prepared multitask neural inputs with shared preprocessing state."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_prob_train: pd.DataFrame
    y_prob_val: pd.DataFrame
    y_prob_test: pd.DataFrame
    y_count_train: pd.DataFrame
    y_count_val: pd.DataFrame
    y_count_test: pd.DataFrame
    y_magnitude_train: pd.DataFrame
    y_magnitude_val: pd.DataFrame
    y_magnitude_test: pd.DataFrame
    feature_cols: list[str]
    train_ids: pd.Series
    val_ids: pd.Series
    test_ids: pd.Series
    train_splits: pd.Series
    val_splits: pd.Series
    test_splits: pd.Series
    imputer: SimpleImputer | None
    scaler: StandardScaler | None


def _validate_required_targets(split_name: str, df: pd.DataFrame) -> None:
    missing_targets = [
        column for column in ALL_NEURAL_TARGET_COLUMNS if column not in df.columns
    ]
    if missing_targets:
        raise ValueError(
            f"Split '{split_name}' is missing required neural target columns: "
            f"{missing_targets}"
        )


def _validate_required_id_column(
    split_name: str,
    df: pd.DataFrame,
    id_col: str,
) -> None:
    if id_col not in df.columns:
        raise ValueError(
            f"Split '{split_name}' is missing required id column '{id_col}'."
        )


def _validate_required_split_column(
    split_name: str,
    df: pd.DataFrame,
    split_col: str | None,
) -> None:
    if split_col is None:
        return
    if split_col not in df.columns:
        raise ValueError(
            f"Split '{split_name}' is missing required split column '{split_col}'."
        )


def _validate_split_frames(
    splits: dict[str, pd.DataFrame],
    id_col: str,
    split_col: str | None,
) -> None:
    for split_name in EXPECTED_SPLITS:
        if split_name not in splits:
            raise ValueError(f"Missing expected split dataframe: {split_name}")
        if splits[split_name].empty:
            raise ValueError(f"Split '{split_name}' is empty.")
        _validate_required_targets(split_name=split_name, df=splits[split_name])
        _validate_required_id_column(
            split_name=split_name,
            df=splits[split_name],
            id_col=id_col,
        )
        _validate_required_split_column(
            split_name=split_name,
            df=splits[split_name],
            split_col=split_col,
        )


def _extract_targets(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    return df.loc[:, target_cols].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 35 engineered seismic features used for Task 1 probability prediction.

    These features extend the raw ``nn_enriched_v1`` columns.  The function is
    safe to call on any split dataframe; it only adds columns whose source
    columns are present, so it degrades gracefully on non-enriched datasets.

    Returns a copy — the input dataframe is never modified.
    """
    import numpy as np

    df = df.copy()
    has_gcmt = (
        df["has_gcmt"].fillna(False)
        if "has_gcmt" in df.columns
        else pd.Series(False, index=df.index)
    )

    # ── Depth regime ──────────────────────────────────────────────────────
    if "trigger_depth_km" in df.columns:
        df["depth_shallow"]      = (df["trigger_depth_km"] < 70).astype(float)
        df["depth_intermediate"] = df["trigger_depth_km"].between(70, 300).astype(float)
        df["depth_deep"]         = (df["trigger_depth_km"] > 300).astype(float)

    # ── Magnitude / activity transforms ──────────────────────────────────
    if "trigger_magnitude" in df.columns:
        df["log_magnitude"] = np.log10(df["trigger_magnitude"].clip(lower=1e-6))

    if "trigger_magnitude" in df.columns and "trigger_depth_km" in df.columns:
        df["mag_depth_ratio"] = df["trigger_magnitude"] / (df["trigger_depth_km"] + 1)

    if "prior_global_event_count_24h" in df.columns:
        df["log_prior_24h"] = np.log1p(df["prior_global_event_count_24h"])

    if "prior_global_event_count_7d" in df.columns:
        df["log_prior_7d"] = np.log1p(df["prior_global_event_count_7d"])

    if (
        "prior_global_event_count_24h" in df.columns
        and "prior_global_event_count_7d" in df.columns
    ):
        denom = (df["prior_global_event_count_7d"] / 7.0).replace(0, np.nan)
        df["seismicity_acceleration"] = df["prior_global_event_count_24h"] / denom

    if "trigger_magnitude" in df.columns and "prior_global_event_count_24h" in df.columns:
        df["mag_x_log_prior"] = (
            df["trigger_magnitude"] * np.log1p(df["prior_global_event_count_24h"])
        )

    if "trigger_magnitude" in df.columns and "depth_shallow" in df.columns:
        df["mag_x_shallow"] = df["trigger_magnitude"] * df["depth_shallow"]

    # ── Temporal cyclical encodings ───────────────────────────────────────
    if "trigger_month" in df.columns:
        df["sin_month"] = np.sin(2 * np.pi * df["trigger_month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["trigger_month"] / 12)

    if "trigger_hour" in df.columns:
        df["sin_hour"] = np.sin(2 * np.pi * df["trigger_hour"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["trigger_hour"] / 24)

    if "trigger_dayofyear" in df.columns:
        df["sin_dayofyear"] = np.sin(2 * np.pi * df["trigger_dayofyear"] / 365)
        df["cos_dayofyear"] = np.cos(2 * np.pi * df["trigger_dayofyear"] / 365)

    # ── Spatial ───────────────────────────────────────────────────────────
    if "trigger_longitude" in df.columns and "trigger_latitude" in df.columns:
        df["ring_of_fire"] = (
            (np.abs(df["trigger_longitude"]) > 130)
            & df["trigger_latitude"].between(-60, 60)
        ).astype(float)

    # ── GCMT scalar moment features ───────────────────────────────────────
    if "gcmt_scalar_moment" in df.columns:
        df["log_scalar_moment"] = np.where(
            has_gcmt & df["gcmt_scalar_moment"].notna(),
            np.log10(df["gcmt_scalar_moment"].clip(lower=1e-10)),
            np.nan,
        )

    if "gcmt_moment_exponent" in df.columns:
        df["moment_exponent_centered"] = (
            df["gcmt_moment_exponent"] - 24.0
        ).where(has_gcmt)

    if "gcmt_scalar_moment" in df.columns and "trigger_magnitude" in df.columns:
        df["gcmt_mw"] = np.where(
            has_gcmt & df["gcmt_scalar_moment"].notna(),
            (2 / 3) * np.log10(df["gcmt_scalar_moment"].clip(lower=1e-10)) - 10.7,
            np.nan,
        )
        df["mw_trigger_diff"] = (df["gcmt_mw"] - df["trigger_magnitude"]).where(has_gcmt)

    # ── Rupture geometry ──────────────────────────────────────────────────
    if "dip" in df.columns:
        df["sin_dip"] = np.sin(np.radians(df["dip"].where(has_gcmt)))

    if {"gcmt_eig1", "gcmt_eig2", "gcmt_eig3"}.issubset(df.columns):
        e1 = df["gcmt_eig1"].where(has_gcmt)
        e2 = df["gcmt_eig2"].where(has_gcmt)
        e3 = df["gcmt_eig3"].where(has_gcmt)
        denom_eig = (e1.abs() + e3.abs()).replace(0, np.nan)
        df["clvd_fraction"] = (2 * e2.abs() / denom_eig).where(has_gcmt)
        df["eig_ratio"]     = (e1.abs() / denom_eig).where(has_gcmt)

    if {"gcmt_eig1_plunge", "gcmt_eig3_plunge"}.issubset(df.columns):
        p1 = df["gcmt_eig1_plunge"].where(has_gcmt)
        p3 = df["gcmt_eig3_plunge"].where(has_gcmt)
        df["sin_eig1_plunge"] = np.sin(np.radians(p1))
        df["cos_eig1_plunge"] = np.cos(np.radians(p1))
        df["sin_eig3_plunge"] = np.sin(np.radians(p3))
        df["cos_eig3_plunge"] = np.cos(np.radians(p3))
        df["tp_plunge_diff"]  = (p1 - p3).where(has_gcmt)

    if "gcmt_depth_km" in df.columns and "trigger_depth_km" in df.columns:
        df["centroid_depth_diff"] = (
            df["gcmt_depth_km"] - df["trigger_depth_km"]
        ).where(has_gcmt)

    if "gcmt_half_duration_sec" in df.columns:
        df["log_half_duration"] = np.log1p(
            df["gcmt_half_duration_sec"].where(has_gcmt)
        )

    if "gcmt_mag_diff" in df.columns:
        df["mag_diff_abs"] = df["gcmt_mag_diff"].abs().where(has_gcmt)

    # ── Tectonic regime from rake ─────────────────────────────────────────
    if "rake" in df.columns:
        def _rake_regimes(rake: float) -> tuple[float, float, float]:
            if pd.isna(rake):
                return np.nan, np.nan, np.nan
            r = rake % 360
            ss = min(abs(r), abs(r - 180), abs(r - 360))
            return float(ss < 45), float(45 <= r <= 135), float(225 <= r <= 315)

        regimes = df["rake"].apply(_rake_regimes)
        df["is_strike_slip"] = regimes.apply(lambda x: x[0]).where(has_gcmt)
        df["is_reverse"]     = regimes.apply(lambda x: x[1]).where(has_gcmt)
        df["is_normal"]      = regimes.apply(lambda x: x[2]).where(has_gcmt)

    return df


def get_feature_set_by_name(feature_set_name: str) -> list[str]:
    """Return a named neural feature set."""

    normalized_name = feature_set_name.strip().lower()
    if normalized_name not in _NEURAL_FEATURE_SET_REGISTRY:
        raise ValueError(
            "Unknown neural feature set name. Expected one of "
            f"{sorted(_NEURAL_FEATURE_SET_REGISTRY)}."
        )
    return list(_NEURAL_FEATURE_SET_REGISTRY[normalized_name])


def prepare_multitask_neural_inputs(
    feature_cols: list[str],
    splits: dict[str, pd.DataFrame] | None = None,
    dataset_name: str | None = None,
    missing_strategy: str = "median",
    scale: bool = True,
    allow_missing_optional: bool = True,
    id_col: str = "trigger_event_id",
    split_col: str = "split",
    processed_datasets_dir: Path = PROCESSED_DATASETS_DIR,
    processed_splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> PreparedNeuralInputs:
    """
    Prepare shared feature matrices and raw multitask targets for neural models.

    This helper reuses the existing tabular input layer so train-only imputation
    and optional scaling are fit once and then applied consistently across
    train, validation, and test splits.
    """

    if splits is None:
        splits = load_modeling_splits(
            dataset_name=dataset_name,
            processed_datasets_dir=processed_datasets_dir,
            processed_splits_dir=processed_splits_dir,
        )

    _validate_split_frames(splits=splits, id_col=id_col, split_col=split_col)

    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    # If the requested feature set includes engineered features, derive them
    # now so they are present as columns before resolve_feature_columns runs.
    needs_engineering = any(
        f in feature_cols
        for f in ["depth_shallow", "log_magnitude", "seismicity_acceleration",
                  "ring_of_fire", "is_strike_slip", "clvd_fraction"]
    )
    if needs_engineering:
        train_df = engineer_features(train_df)
        val_df   = engineer_features(val_df)
        test_df  = engineer_features(test_df)

    resolved_feature_cols = resolve_feature_columns(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        requested_cols=feature_cols,
        allow_missing_optional=allow_missing_optional,
    )

    config = InputConfig(
        feature_cols=resolved_feature_cols,
        target_col=PROBABILITY_TARGET_COLUMNS[0],
        missing_strategy=missing_strategy,
        scale=scale,
        drop_rows_with_missing_target=False,
        allow_missing_optional=allow_missing_optional,
    )

    input_layer = TabularInputLayer()
    X_train = input_layer.fit_transform(train_df, config)
    X_val = input_layer.transform(val_df)
    X_test = input_layer.transform(test_df)

    return PreparedNeuralInputs(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_prob_train=_extract_targets(train_df, PROBABILITY_TARGET_COLUMNS),
        y_prob_val=_extract_targets(val_df, PROBABILITY_TARGET_COLUMNS),
        y_prob_test=_extract_targets(test_df, PROBABILITY_TARGET_COLUMNS),
        y_count_train=_extract_targets(train_df, COUNT_TARGET_COLUMNS),
        y_count_val=_extract_targets(val_df, COUNT_TARGET_COLUMNS),
        y_count_test=_extract_targets(test_df, COUNT_TARGET_COLUMNS),
        y_magnitude_train=_extract_targets(train_df, MAGNITUDE_TARGET_COLUMNS),
        y_magnitude_val=_extract_targets(val_df, MAGNITUDE_TARGET_COLUMNS),
        y_magnitude_test=_extract_targets(test_df, MAGNITUDE_TARGET_COLUMNS),
        feature_cols=resolved_feature_cols,
        train_ids=train_df[id_col].copy(),
        val_ids=val_df[id_col].copy(),
        test_ids=test_df[id_col].copy(),
        train_splits=train_df[split_col].copy(),
        val_splits=val_df[split_col].copy(),
        test_splits=test_df[split_col].copy(),
        imputer=input_layer.imputer_,
        scaler=input_layer.scaler_,
    )
