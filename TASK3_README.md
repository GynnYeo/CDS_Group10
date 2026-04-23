# Task 3 README

## Overview

Task 3 focuses on **aftershock severity modelling** and extends the earlier project tasks beyond:

- aftershock occurrence
- aftershock count

into two additional targets:

- **Task 3a**: predict whether at least one larger aftershock (`M >= 4.0`) occurs within `24h` and `72h`
- **Task 3b**: predict the maximum aftershock magnitude within `24h` and `72h`

Task 3a is a **binary classification** task.  
Task 3b is a **regression** task.

## Dataset and Split

The final Task 3 experiments use the processed dataset:

- `data/processed/datasets/earthquake_aftershock_v2_gcmt.parquet`

Chronological split:

- train: `2010-2022`
- validation: `2023`
- test: `2024-2025`

## Task Definitions

### Task 3a

Target columns:

- `y_large_24h`
- `y_large_72h`

Definition:

- label `1` if at least one aftershock with `M >= 4.0` occurs within the relevant forecast horizon
- label `0` otherwise

### Task 3b

Target columns:

- `max_aftershock_magnitude_24h`
- `max_aftershock_magnitude_72h`

Definition:

- candidate aftershocks are filtered by:
  - forecast horizon
  - `50 km` spatial window
  - minimum aftershock magnitude `M >= 2.5`
- the target is the largest qualifying aftershock magnitude within the horizon
- if no qualifying aftershock exists, the regression target is missing for that sample

## Models Evaluated

### Task 3a

- climatology baseline
- simplified RJ-style baseline
- XGBoost classifier
- TabNet classifier

### Task 3b

- mean baseline
- Båth's law baseline
- linear regression (magnitude-only)
- multivariable linear regression
- XGBoost regressor
- TabNet regressor

## Main Scripts

### Task 3a

Run XGBoost:

```powershell
python scripts/run_task3_pipeline.py --backend xgboost
```

Run TabNet:

```powershell
python scripts/run_task3_pipeline.py --backend tabnet
```

### Task 3b

Run XGBoost:

```powershell
python scripts/run_task3_maxmag_pipeline.py --backend xgboost
```

Run TabNet:

```powershell
python scripts/run_task3_maxmag_pipeline.py --backend tabnet
```

## Outputs

Task 3 metrics, predictions, and summary files are written under:

- `reports/metrics/`

Common output patterns:

- `reports/metrics/task3_large_aftershock_*`
- `reports/metrics/task3_maxmag_*`

Figures for slides or visual summaries can be written under:

- `reports/figures/`

## Plotting Helpers

Task 3a ROC curves:

```powershell
python scripts/plot_task3a_roc_curves.py --model-name task3_large_aftershock_xgboost
```

If the current predictions file does not contain XGBoost outputs, rerun:

```powershell
python scripts/run_task3_pipeline.py --backend xgboost
```

Task 3b comparison bars using final reported values:

```powershell
python scripts/plot_task3b_bars.py --use-reported-results
```

## Reference Results

Current best final models:

- **Task 3a**: XGBoost
- **Task 3b**: XGBoost

Detailed experimental analysis, tuning notes, and final result discussion are documented in:

- `report.md`
