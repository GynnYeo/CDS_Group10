# CDS_Group10

Earthquake aftershock forecasting repository with a shared trigger-level preprocessing pipeline and a baseline modeling stage built on top of saved processed datasets.

## Problem Setup

Trigger earthquake:

* magnitude `>= 5.0`

Primary targets:

* `y_24h`
* `y_72h`

Secondary targets:

* `n_aftershocks_24h`
* `n_aftershocks_72h`

Aftershock definition:

* `event_time > trigger_time`
* `event_time <= trigger_time + horizon`
* magnitude `>= 2.5`
* distance `<= 50 km`

Time split:

* train = `2015-2023`
* val = `2024`
* test = `2025`

## Current Status

Implemented:

* shared preprocessing pipeline
* processed trigger-level dataset creation
* train / val / test split generation
* model input layer
* shared evaluation utilities
* climatology baseline
* simplified RJ-style baseline

Not yet implemented:

* ETAS-style benchmark
* AI model(s)

---

## Official Workflow

This repository should be used in two main stages:

## Official Workflow

This repository should be used in three main stages:

### 1. Download raw data

For first-time setup, run:

```bash
python scripts/download_raw_data.py
```
This step:
- checks whether raw ComCat data already exists under data/raw/comcat
- checks whether raw GCMT data already exists under data/raw/moment_tensor
- downloads missing raw data only
- skips downloading if the required raw files are already present

### 2. Build the processed dataset

Run the shared preprocessing pipeline first.

```bash
python -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --moment-tensor-input data/raw/moment_tensor \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --train-start-year 2010 \
  --train-end-year 2022 \
  --validation-year 2023 \
  --test-years 2024 2025
```

If you need to recompute the dataset (e.g. after updating raw data or split definitions), run:
```bash
python -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --moment-tensor-input data/raw/moment_tensor \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --train-start-year 2010 \
  --train-end-year 2022 \
  --validation-year 2023 \
  --test-years 2024 2025 \
  --force-recompute
```

This step:

* loads raw source data
* cleans ComCat
* optionally parses and merges Global CMT enrichment
* builds triggers
* builds labels
* builds shared features
* assembles the final trigger-level dataset
* assigns train / val / test splits
* saves reusable outputs under `data/processed/`

### 3. Run and save baselines

After the processed dataset exists, run:

```bash
python scripts/run_and_save_baselines.py
```

This step:

* loads the saved processed train / val / test splits
* runs the climatology baseline
* runs the simplified RJ-style baseline
* evaluates both baselines
* saves outputs to:

```text
reports/metrics/baselines_predictions.csv
reports/metrics/baselines_metrics.csv
```

### 4. Future model training

Future AI model scripts should also operate on the saved processed dataset under `data/processed/`.

Important:

* model scripts do **not** rebuild the dataset automatically
* the shared preprocessing pipeline must be run explicitly first

---

## Pipeline Boundaries

### Shared preprocessing pipeline

`src/dataset/` is the canonical shared preprocessing layer.

It is responsible for:

* cleaning source data
* optional enrichment merge
* trigger construction
* shared labels
* shared features
* final dataset assembly
* time-based split assignment

This stage writes reusable parquet outputs and should remain separate from model-specific code.

### Modeling and evaluation stage

`src/models/` and `src/evaluation/` consume the saved processed outputs.

They are responsible for:

* loading processed train / val / test splits
* preparing model inputs
* running baselines and future models
* computing evaluation metrics
* saving experiment outputs

These modules should not rebuild raw datasets.

---

## Data Sources

### ComCat

ComCat is the master dataset and source of truth for:

* trigger construction
* label generation
* historical seismicity features

### Global CMT

Global CMT is optional left-joined enrichment.

Important rules:

* unmatched ComCat rows are preserved
* no ComCat rows should be dropped due to missing GCMT
* GCMT merge logic is intentionally conservative

---

## Outputs

### Interim outputs

Saved under `data/interim/`:

* cleaned ComCat parquet
* ComCat cleaning log csv
* raw moment tensor parquet when GCMT input is provided
* cleaned moment tensor parquet
* moment tensor cleaning log csv
* enriched event parquet
* trigger parquet
* labels parquet
* features parquet

### Processed outputs

Saved under `data/processed/`:

* final dataset parquet in `data/processed/datasets/`
* split-specific parquet files in `data/processed/splits/`

### Reported metrics outputs

Saved under `reports/metrics/`:

* `baselines_predictions.csv`
* `baselines_metrics.csv`

---

## Modeling Utilities

Reusable modeling-stage code lives under:

* `src/models/`
* `src/evaluation/`

Current utilities include:

* `src/models/feature_sets.py`

  * explicit starter feature lists
* `src/models/input_layer.py`

  * split loading
  * feature resolution
  * train-only imputation
  * optional scaling
* `src/models/baselines/`

  * climatology baseline
  * simplified RJ-style baseline
* `src/models/run_baselines.py`

  * runs current baselines and returns standardized prediction / metrics tables
* `src/evaluation/metrics.py`

  * shared metric helpers
* `src/evaluation/validation.py`

  * prediction validation helpers

These operate on processed outputs under `data/processed/` and do not modify the shared preprocessing pipeline.

---

## Project Layout

```text
src/data/              source-level loading, cleaning, enrichment, download helpers
src/dataset/           trigger-level shared dataset construction
src/models/            model input utilities, baselines, model runners
src/evaluation/        shared evaluation and prediction validation helpers
scripts/               official runnable entrypoints (download, baselines, future model runs)
data/raw/              raw source inputs
data/interim/          saved intermediate pipeline artifacts
data/processed/        final datasets and split-specific outputs
reports/metrics/       saved baseline and future model metrics
archive/v1/            archived older pipeline
```

---

## Main Rule for Collaborators

Use the repository in this order:

1. build the processed dataset
2. run the baselines
3. add future models on top of the saved processed outputs

Do not create separate dataset pipelines for different models.
