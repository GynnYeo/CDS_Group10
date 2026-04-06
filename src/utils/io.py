from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = ensure_parent_dir(path)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported file extension for {path}")

    return path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file extension for {path}")
