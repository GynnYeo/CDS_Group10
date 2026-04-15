from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models.feature_sets import NN_CORE_V1, NN_ENRICHED_V1
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
ALL_NEURAL_TARGET_COLUMNS = PROBABILITY_TARGET_COLUMNS + COUNT_TARGET_COLUMNS

_NEURAL_FEATURE_SET_REGISTRY = {
    "nn_core_v1": NN_CORE_V1,
    "nn_enriched_v1": NN_ENRICHED_V1,
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
