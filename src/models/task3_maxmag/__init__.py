from src.models.task3_maxmag.baselines import (
    BathsLawBaseline,
    LinearMaxMagnitudeBaseline,
    MagnitudeOnlyLinearMaxMagnitudeBaseline,
    MeanMaxMagnitudeBaseline,
)
from src.models.task3_maxmag.labels import build_or_load_max_aftershock_magnitude_labels
from src.models.task3_maxmag.run import (
    evaluate_task3_maxmag_predictions,
    run_task3_maxmag_pipeline,
)
from src.models.task3_maxmag.xgb import Task3MaxMagRegressor

__all__ = [
    "BathsLawBaseline",
    "LinearMaxMagnitudeBaseline",
    "MagnitudeOnlyLinearMaxMagnitudeBaseline",
    "MeanMaxMagnitudeBaseline",
    "build_or_load_max_aftershock_magnitude_labels",
    "evaluate_task3_maxmag_predictions",
    "run_task3_maxmag_pipeline",
    "Task3MaxMagRegressor",
]
