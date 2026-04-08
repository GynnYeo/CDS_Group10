from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models.feature_sets import NON_FEATURE_COLUMNS
from src.utils.io import load_dataframe
from src.utils.paths import PROCESSED_DATASETS_DIR, PROCESSED_SPLITS_DIR


SUPPORTED_MISSING_STRATEGIES = {"median", "zero", "none"}
EXPECTED_SPLITS = ("train", "val", "test")


def _slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("dataset_name must contain at least one letter or number.")
    return slug


def _infer_dataset_slug(
    dataset_name: str | None,
    processed_datasets_dir: Path,
) -> str:
    if dataset_name is not None:
        return _slugify_dataset_name(dataset_name)

    dataset_paths = sorted(processed_datasets_dir.glob("*.parquet"))
    if len(dataset_paths) != 1:
        raise ValueError(
            "dataset_name is required when multiple processed datasets exist."
        )
    return dataset_paths[0].stem


@dataclass(slots=True)
class InputConfig:
    """Configuration for tabular modeling inputs."""

    feature_cols: list[str]
    target_col: str
    missing_strategy: str = "median"
    scale: bool = False
    drop_rows_with_missing_target: bool = True
    allow_missing_optional: bool = True

    def __post_init__(self) -> None:
        if not self.feature_cols:
            raise ValueError("feature_cols must contain at least one feature name.")
        if self.target_col in self.feature_cols:
            raise ValueError("target_col cannot also appear in feature_cols.")
        if self.missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
            raise ValueError(
                f"missing_strategy must be one of {sorted(SUPPORTED_MISSING_STRATEGIES)}."
            )
        forbidden = set(self.feature_cols).intersection(NON_FEATURE_COLUMNS)
        if forbidden:
            raise ValueError(
                "feature_cols contains non-feature columns: "
                f"{sorted(forbidden)}"
            )


@dataclass(slots=True)
class PreparedInputs:
    """Prepared train/validation/test arrays plus fitted preprocessors."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: list[str]
    target_col: str
    imputer: SimpleImputer | None
    scaler: StandardScaler | None


def load_modeling_splits(
    dataset_name: str | None = None,
    processed_datasets_dir: Path = PROCESSED_DATASETS_DIR,
    processed_splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Load train/val/test modeling splits from processed outputs.

    Preference order:
    1. split-specific parquet files in data/processed/splits/<dataset_slug>/
    2. a single processed dataset parquet with a required `split` column
    """
    dataset_slug = _infer_dataset_slug(dataset_name, processed_datasets_dir)
    split_dir = processed_splits_dir / dataset_slug

    split_paths = {
        split_name: split_dir / f"{dataset_slug}__{split_name}.parquet"
        for split_name in EXPECTED_SPLITS
    }
    if all(path.exists() for path in split_paths.values()):
        splits = {
            split_name: load_dataframe(path).copy()
            for split_name, path in split_paths.items()
        }
        _validate_non_empty_splits(splits)
        return splits

    dataset_path = processed_datasets_dir / f"{dataset_slug}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Could not find processed dataset or split exports for "
            f"dataset '{dataset_slug}'."
        )

    full_df = load_dataframe(dataset_path).copy()
    if "split" not in full_df.columns:
        raise ValueError(
            f"Processed dataset is missing required split column: {dataset_path}"
        )

    available_splits = set(full_df["split"].dropna().astype(str).unique())
    missing_splits = [split for split in EXPECTED_SPLITS if split not in available_splits]
    if missing_splits:
        raise ValueError(
            "Processed dataset is missing expected split values: "
            f"{missing_splits}"
        )

    splits = {
        split_name: full_df.loc[full_df["split"] == split_name].copy()
        for split_name in EXPECTED_SPLITS
    }
    _validate_non_empty_splits(splits)
    return splits


def _validate_non_empty_splits(splits: dict[str, pd.DataFrame]) -> None:
    for split_name in EXPECTED_SPLITS:
        if split_name not in splits:
            raise ValueError(f"Missing expected split dataframe: {split_name}")
        if splits[split_name].empty:
            raise ValueError(f"Split '{split_name}' is empty.")


def resolve_feature_columns(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    requested_cols: list[str],
    allow_missing_optional: bool = True,
) -> list[str]:
    """
    Resolve a requested feature list against all split schemas.

    When allow_missing_optional is True, columns missing from any split are
    dropped from the returned list. When False, missing columns raise a
    ValueError. Columns listed in NON_FEATURE_COLUMNS are always rejected.
    """
    if not requested_cols:
        raise ValueError("requested_cols must contain at least one feature.")

    requested_unique = list(dict.fromkeys(requested_cols))
    forbidden = sorted(set(requested_unique).intersection(NON_FEATURE_COLUMNS))
    if forbidden:
        raise ValueError(
            "Requested features include forbidden non-feature columns: "
            f"{forbidden}"
        )

    split_columns = {
        "train": set(train_df.columns),
        "val": set(val_df.columns),
        "test": set(test_df.columns),
    }
    resolved: list[str] = []
    missing_by_column: dict[str, list[str]] = {}

    for column in requested_unique:
        missing_splits = [
            split_name
            for split_name, columns in split_columns.items()
            if column not in columns
        ]
        if missing_splits:
            missing_by_column[column] = missing_splits
            continue
        resolved.append(column)

    if missing_by_column and not allow_missing_optional:
        details = ", ".join(
            f"{column} missing from {missing_splits}"
            for column, missing_splits in missing_by_column.items()
        )
        raise ValueError(f"Requested feature columns are unavailable: {details}")

    if not resolved:
        raise ValueError("No requested feature columns were available in all splits.")

    return resolved


class TabularInputLayer:
    """
    Fit and apply simple numeric tabular preprocessing.

    This version supports numeric or boolean columns only. It returns pandas
    DataFrames so feature names and column order remain visible downstream.
    """

    def __init__(self) -> None:
        self.config: InputConfig | None = None
        self.feature_cols_: list[str] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, train_df: pd.DataFrame, config: InputConfig) -> "TabularInputLayer":
        """Fit train-only preprocessing state."""
        self._validate_target_column(train_df, config.target_col)
        if config.drop_rows_with_missing_target:
            train_df = train_df.loc[train_df[config.target_col].notna()].copy()

        self._validate_feature_columns(train_df, config.feature_cols, config.target_col)
        X_train = self._coerce_features(train_df[config.feature_cols])

        self.config = config
        self.feature_cols_ = list(config.feature_cols)
        self.imputer_ = self._build_imputer(config.missing_strategy)
        if self.imputer_ is not None:
            self.imputer_.fit(X_train)

        self.scaler_ = StandardScaler() if config.scale else None
        if self.scaler_ is not None:
            transformed = self._apply_imputer(X_train)
            self.scaler_.fit(transformed)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a split with fitted train-only preprocessing state."""
        if self.config is None or self.feature_cols_ is None:
            raise ValueError("TabularInputLayer must be fit before transform().")

        X = self._coerce_features(df[self.feature_cols_])
        X = self._apply_imputer(X)
        if self.scaler_ is not None:
            X_values = self.scaler_.transform(X)
            X = pd.DataFrame(X_values, index=X.index, columns=self.feature_cols_)
        return X

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        config: InputConfig,
    ) -> pd.DataFrame:
        """Fit on training data and return transformed training features."""
        return self.fit(train_df, config).transform(train_df)

    def _validate_target_column(self, df: pd.DataFrame, target_col: str) -> None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' is missing from dataframe.")

    def _validate_feature_columns(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> None:
        missing = [column for column in feature_cols if column not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if target_col in feature_cols:
            raise ValueError("target_col cannot be used as a feature.")

    def _coerce_features(self, X: pd.DataFrame) -> pd.DataFrame:
        invalid_cols = [
            column
            for column in X.columns
            if not (
                pd.api.types.is_numeric_dtype(X[column])
                or pd.api.types.is_bool_dtype(X[column])
            )
        ]
        if invalid_cols:
            raise ValueError(
                "Current input layer supports numeric tabular features only. "
                f"Unsupported columns: {invalid_cols}"
            )

        numeric_X = X.copy()
        for column in numeric_X.columns:
            if pd.api.types.is_bool_dtype(numeric_X[column]):
                numeric_X[column] = numeric_X[column].astype(float)

        return numeric_X.astype(float)

    def _build_imputer(self, strategy: str) -> SimpleImputer | None:
        if strategy == "none":
            return None
        if strategy == "median":
            return SimpleImputer(strategy="median")
        if strategy == "zero":
            return SimpleImputer(strategy="constant", fill_value=0.0)
        raise ValueError(f"Unsupported missing strategy: {strategy}")

    def _apply_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            return X.copy()
        X_values = self.imputer_.transform(X)
        return pd.DataFrame(X_values, index=X.index, columns=self.feature_cols_)


def prepare_tabular_inputs(
    config: InputConfig,
    splits: dict[str, pd.DataFrame] | None = None,
    dataset_name: str | None = None,
    processed_datasets_dir: Path = PROCESSED_DATASETS_DIR,
    processed_splits_dir: Path = PROCESSED_SPLITS_DIR,
) -> PreparedInputs:
    """
    Prepare train/val/test tabular inputs from processed split outputs.

    If `splits` is not provided, processed files are loaded from data/processed/.
    """
    if splits is None:
        splits = load_modeling_splits(
            dataset_name=dataset_name,
            processed_datasets_dir=processed_datasets_dir,
            processed_splits_dir=processed_splits_dir,
        )

    _validate_non_empty_splits(splits)
    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    resolved_feature_cols = resolve_feature_columns(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        requested_cols=config.feature_cols,
        allow_missing_optional=config.allow_missing_optional,
    )

    resolved_config = InputConfig(
        feature_cols=resolved_feature_cols,
        target_col=config.target_col,
        missing_strategy=config.missing_strategy,
        scale=config.scale,
        drop_rows_with_missing_target=config.drop_rows_with_missing_target,
        allow_missing_optional=config.allow_missing_optional,
    )

    train_df = _prepare_target_rows(train_df, resolved_config)
    val_df = _prepare_target_rows(val_df, resolved_config)
    test_df = _prepare_target_rows(test_df, resolved_config)

    input_layer = TabularInputLayer()
    X_train = input_layer.fit_transform(train_df, resolved_config)
    X_val = input_layer.transform(val_df)
    X_test = input_layer.transform(test_df)

    y_train = train_df[resolved_config.target_col].copy()
    y_val = val_df[resolved_config.target_col].copy()
    y_test = test_df[resolved_config.target_col].copy()

    return PreparedInputs(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_cols=resolved_feature_cols,
        target_col=resolved_config.target_col,
        imputer=input_layer.imputer_,
        scaler=input_layer.scaler_,
    )


def _prepare_target_rows(df: pd.DataFrame, config: InputConfig) -> pd.DataFrame:
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' is missing from split.")

    prepared = df.copy()
    if config.drop_rows_with_missing_target:
        prepared = prepared.loc[prepared[config.target_col].notna()].copy()
    return prepared
