from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.input_layer import InputConfig, TabularInputLayer, load_modeling_splits
from src.models.neural.model import build_multitask_mlp


def _to_tuple_or_none(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None


class MultiTaskInferenceRunner:
    """
    Loads task-specific checkpoints from one multitask run and uses:
    - best_prob checkpoint for Task 1
    - best_count checkpoint for Task 2
    - best_magnitude checkpoint for Task 3
    """

    def __init__(
        self,
        dataset_name: str,
        run_name: str,
        checkpoint_dir: str = "reports/checkpoints",
        device: str = "cpu",
    ) -> None:
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.device = torch.device(device)

        self.project_root = PROJECT_ROOT
        self.checkpoint_dir = self.project_root / checkpoint_dir / run_name

        self.ckpt_paths = {
            "prob": self.checkpoint_dir / f"{run_name}_best_prob.pt",
            "count": self.checkpoint_dir / f"{run_name}_best_count.pt",
            "magnitude": self.checkpoint_dir / f"{run_name}_best_magnitude.pt",
        }

        for task, path in self.ckpt_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {task} checkpoint: {path}")

        reference_ckpt = torch.load(self.ckpt_paths["prob"], map_location=self.device)
        self.reference_args = reference_ckpt.get("args", {})
        self.feature_cols = reference_ckpt.get("feature_cols")
        if not self.feature_cols:
            raise ValueError("Checkpoint is missing feature_cols metadata.")

        self.count_modeling_mode = (
            reference_ckpt.get("training_config", {}).get("count_modeling_mode", "standard")
        )

        self.input_layer = self._fit_preprocessor()
        self.models = {
            "prob": self._load_model_from_checkpoint(self.ckpt_paths["prob"]),
            "count": self._load_model_from_checkpoint(self.ckpt_paths["count"]),
            "magnitude": self._load_model_from_checkpoint(self.ckpt_paths["magnitude"]),
        }

    def _fit_preprocessor(self) -> TabularInputLayer:
        splits = load_modeling_splits(dataset_name=self.dataset_name)
        train_df = splits["train"].copy()

        missing_strategy = self.reference_args.get("missing_strategy", "median")
        scale = not bool(self.reference_args.get("no_scale", False))

        config = InputConfig(
            feature_cols=self.feature_cols,
            target_col="y_24h",
            missing_strategy=missing_strategy,
            scale=scale,
            drop_rows_with_missing_target=False,
            allow_missing_optional=True,
        )

        layer = TabularInputLayer()
        layer.fit(train_df, config)
        return layer

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> torch.nn.Module:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        args = ckpt.get("args", {})

        hidden_dims = tuple(args.get("hidden_dims", [128, 64]))
        dropout = float(args.get("dropout", 0.2))
        probability_head_hidden_dims = _to_tuple_or_none(
            args.get("probability_head_hidden_dims")
        )
        count_head_hidden_dims = _to_tuple_or_none(args.get("count_head_hidden_dims"))
        magnitude_head_hidden_dims = _to_tuple_or_none(
            args.get("magnitude_head_hidden_dims")
        )

        model = build_multitask_mlp(
            input_dim=len(self.feature_cols),
            hidden_dims=hidden_dims,
            dropout=dropout,
            probability_head_hidden_dims=probability_head_hidden_dims,
            count_head_hidden_dims=count_head_hidden_dims,
            magnitude_head_hidden_dims=magnitude_head_hidden_dims,
        ).to(self.device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _payload_to_feature_df(self, payload: dict[str, Any]) -> pd.DataFrame:
        """
        Build one-row DataFrame with exact training feature columns.
        Any missing fields stay NaN and are handled by the fitted preprocessor.
        """
        row = {col: np.nan for col in self.feature_cols}

        # Friendly aliases
        aliases = {
            "latitude": "trigger_latitude",
            "longitude": "trigger_longitude",
            "depth_km": "trigger_depth_km",
            "magnitude": "trigger_magnitude",
        }

        for input_key, feature_key in aliases.items():
            if input_key in payload and feature_key in row:
                row[feature_key] = payload[input_key]

        # Direct passthrough if JSON already uses exact feature names
        for key, value in payload.items():
            if key in row:
                row[key] = value

        # Derive time-based fields if trigger_time is present and exact columns exist
        trigger_time = payload.get("trigger_time")
        if trigger_time:
            ts = pd.to_datetime(trigger_time, utc=True, errors="coerce")
            if pd.notna(ts):
                if "trigger_month" in row and pd.isna(row["trigger_month"]):
                    row["trigger_month"] = int(ts.month)
                if "trigger_dayofyear" in row and pd.isna(row["trigger_dayofyear"]):
                    row["trigger_dayofyear"] = int(ts.dayofyear)
                if "trigger_hour" in row and pd.isna(row["trigger_hour"]):
                    row["trigger_hour"] = int(ts.hour)
                if "trigger_year_feature" in row and pd.isna(row["trigger_year_feature"]):
                    row["trigger_year_feature"] = int(ts.year)

        # Basic GCMT fallback
        if "has_gcmt" in row and pd.isna(row["has_gcmt"]):
            gcmt_present = any(str(k).startswith("gcmt_") for k in payload.keys())
            row["has_gcmt"] = 1.0 if gcmt_present else 0.0

        return pd.DataFrame([row], columns=self.feature_cols)

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        feature_df = self._payload_to_feature_df(payload)
        transformed = self.input_layer.transform(feature_df)
        x = torch.tensor(
            transformed.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            prob_outputs = self.models["prob"](x)
            count_outputs = self.models["count"](x)
            magnitude_outputs = self.models["magnitude"](x)

        prob = torch.sigmoid(prob_outputs["prob_logits"]).cpu().numpy().reshape(-1)
        count_log = count_outputs["count_pred"].cpu().numpy().reshape(-1)
        magnitude = magnitude_outputs["magnitude_pred"].cpu().numpy().reshape(-1)

        if self.count_modeling_mode == "standard":
            count = np.expm1(count_log)
            count = np.clip(count, a_min=0.0, a_max=None)
        else:
            positive_count = np.expm1(count_log)
            positive_count = np.clip(positive_count, a_min=0.0, a_max=None)
            count = prob * positive_count

        return {
            "trigger_event_id": payload.get("trigger_event_id", "unknown_event"),
            "latitude": payload.get("latitude", payload.get("trigger_latitude")),
            "longitude": payload.get("longitude", payload.get("trigger_longitude")),
            "prob_24h": float(prob[0]),
            "prob_72h": float(prob[1]),
            "count_24h": float(count[0]),
            "count_72h": float(count[1]),
            "mag_24h": float(magnitude[0]),
            "mag_72h": float(magnitude[1]),
        }

    def predict_many(self, payloads: list[dict[str, Any]]) -> pd.DataFrame:
        rows = [self.predict_one(payload) for payload in payloads]
        return pd.DataFrame(rows)