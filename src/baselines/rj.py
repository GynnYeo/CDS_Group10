from __future__ import annotations

import pandas as pd


def fit_rj_baseline(train_df: pd.DataFrame) -> dict:
    """
    Scaffold for a Reasenberg-Jones style baseline.

    TODO:
    - Decide whether you want a literature-inspired approximation or a simpler
    school-project proxy calibrated from training data.
    - Define which trigger-side inputs are available at prediction time.
    - Fit separate outputs for 24h and 72h probability/count targets.
    """
    if train_df.empty:
        raise ValueError("train_df must not be empty.")

    return {
        "status": "todo",
        "message": "RJ baseline scaffold created. Implementation still needed.",
        "n_train": int(len(train_df)),
    }


def predict_rj_baseline(model: dict, scoring_df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder prediction function so the module has a stable interface.
    """
    predictions = pd.DataFrame({"trigger_event_id": scoring_df["trigger_event_id"].values})

    # TODO:
    # - Replace placeholder outputs with calibrated RJ estimates.
    predictions["pred_y_24h"] = 0.0
    predictions["pred_y_72h"] = 0.0
    predictions["pred_n_aftershocks_24h"] = 0.0
    predictions["pred_n_aftershocks_72h"] = 0.0

    return predictions
