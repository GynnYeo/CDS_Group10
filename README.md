# CDS_Group10

Earthquake aftershock forecasting repository with a shared trigger-level preprocessing pipeline, traditional baseline models, and a modular neural-network modeling pipeline built on top of saved processed datasets.

The repository is organized around one main principle:

> Build the processed trigger-level dataset once, then run all baselines and neural models from the same saved train / validation / test splits.

This keeps model comparisons fair and avoids each model creating its own private version of the dataset.

---

## 1. Problem Setup

Each row in the modeling dataset represents a **trigger earthquake**:

```text
trigger earthquake = earthquake with magnitude >= 5.0
```

Aftershocks are associated with a trigger using the shared project definition:

* aftershock event time must be after the trigger time
* aftershock event time must be within the forecast horizon
* aftershock magnitude must be `>= 2.5`
* aftershock distance from the trigger must be `<= 50 km`

Current forecast horizons:

```text
24 hours
72 hours
```

### 1.1 Targets

The current trigger-level targets are:

| Target                         | Meaning                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| `y_24h`                        | Whether at least one associated aftershock occurs within 24h |
| `y_72h`                        | Whether at least one associated aftershock occurs within 72h |
| `n_aftershocks_24h`            | Number of associated aftershocks within 24h                  |
| `n_aftershocks_72h`            | Number of associated aftershocks within 72h                  |
| `max_aftershock_magnitude_24h` | Maximum magnitude among associated aftershocks within 24h    |
| `max_aftershock_magnitude_72h` | Maximum magnitude among associated aftershocks within 72h    |

For maximum-magnitude targets, rows with no aftershock in the horizon use `NaN`. This is intentional because maximum aftershock magnitude is undefined when no aftershock exists.

### 1.2 Current time split

The current processed dataset build command uses:

```text
train: 2010–2022
validation: 2023
test: 2024 and 2025
```

Use the explicit build command below as the source of truth for the active split configuration.

---

## 2. Current Status

Implemented:

* raw ComCat download helper
* raw GCMT / moment-tensor download or input support
* shared trigger-level preprocessing pipeline
* optional GCMT enrichment via left join
* processed dataset creation
* train / validation / test split generation
* shared model input layer
* shared evaluation utilities
* climatology baseline
* simplified RJ-style baseline
* modular PyTorch multi-task MLP pipeline
* probability, count, and conditional maximum-magnitude neural outputs
* evaluation-only neural prediction script

Not yet implemented or still future work:

* ETAS-style benchmark
* more advanced neural architectures
* calibrated probability post-processing
* fully distributional count modeling

---

## 3. Official Workflow

The repository should be used in this order:

1. download or prepare raw data
2. build the processed trigger-level dataset
3. run traditional baselines
4. run neural models or evaluate saved neural predictions

Model scripts should consume the saved processed outputs. They should not rebuild the dataset automatically.

---

## 4. Download Raw Data

For first-time setup, run:

```bash
python scripts/download_raw_data.py
```

This step:

* checks whether raw ComCat data already exists under `data/raw/comcat/`
* checks whether raw GCMT data already exists under `data/raw/moment_tensor/`
* downloads missing raw data only
* skips downloads when the required raw files already exist

---

## 5. Build the Processed Dataset

Run the shared preprocessing pipeline before running any model.

```bash
python -m src.dataset.build_dataset
    --comcat-input data/raw/comcat
    --moment-tensor-input data/raw/moment_tensor
    --dataset-name earthquake_aftershock_v2_gcmt
    --train-start-year 2010   
    --train-end-year 2022   
    --validation-year 2023   
    --test-years 2024 2025
```

To force a rebuild after changing raw data, split definitions, features, or labels:

```bash
python -m src.dataset.build_dataset
    --comcat-input data/raw/comcat
    --moment-tensor-input data/raw/moment_tensor
    --dataset-name earthquake_aftershock_v2_gcmt
    --train-start-year 2010
    --train-end-year 2022
    --validation-year 2023
    --test-years 2024 2025
    --force-recompute
```

This step:

* loads raw source data
* cleans ComCat events
* optionally parses and cleans GCMT / moment-tensor data
* merges optional GCMT enrichment into ComCat events
* constructs trigger earthquakes
* generates binary, count, and maximum-magnitude labels
* builds shared tabular features
* assembles the final trigger-level dataset
* assigns train / validation / test splits
* saves reusable outputs under `data/interim/` and `data/processed/`

---

## 6. Run Traditional Baselines

After the processed dataset exists, run:

```bash
python scripts/run_and_save_baselines.py
```

This step:

* loads the saved processed train / validation / test splits
* runs the climatology baseline
* runs the simplified RJ-style baseline
* evaluates both baselines
* saves baseline outputs under `reports/metrics/`

Expected baseline outputs:

```text
reports/metrics/baselines_predictions.csv
reports/metrics/baselines_metrics.csv
```

---

## 7. Run the Neural Network Pipeline

The neural-network pipeline is documented separately to avoid duplicating detailed commands here.

See:

```text
docs/running_multihead_NN.md
```

or, if using the updated project notes produced during development:

```text
running_multihead_NN.md
```

The neural pipeline currently supports:

* full training + evaluation with `scripts/train_multitask_mlp.py`
* evaluation-only metric regeneration with `scripts/evaluate_multitask_predictions.py`
* modular neural helpers under `src/models/neural/`
* grouped neural metrics under `src/evaluation/neural_metrics.py`

Neural outputs are saved in run-specific folders:

```text
reports/metrics/<run_name>/
reports/checkpoints/<run_name>/
```

---

## 8. Pipeline Boundaries

### 8.1 Shared preprocessing pipeline

`src/dataset/` is the canonical shared preprocessing layer.

It is responsible for:

* source data cleaning
* optional GCMT enrichment merge
* trigger construction
* shared label generation
* shared feature generation
* final dataset assembly
* time-based split assignment

This stage writes reusable parquet outputs and should remain separate from model-specific code.

### 8.2 Modeling and evaluation stage

`src/models/`, `src/evaluation/`, and `scripts/` consume the saved processed outputs.

They are responsible for:

* loading processed train / validation / test splits
* preparing model inputs
* running baselines and neural models
* formatting prediction tables
* computing evaluation metrics
* saving experiment outputs

These modules should not rebuild raw datasets.

---

## 9. Data Sources

### 9.1 ComCat

ComCat is the master event dataset and source of truth for:

* trigger construction
* aftershock label generation
* historical seismicity features

### 9.2 Global CMT / Moment Tensor Data

Global CMT is optional enrichment.

Important rules:

* GCMT is left-joined onto ComCat events
* unmatched ComCat rows are preserved
* no ComCat trigger rows should be dropped because GCMT is missing
* missing GCMT fields are handled later through modeling-stage imputation
* `has_gcmt` is included as an indicator feature for enriched neural experiments

---

## 10. Outputs

### 10.1 Interim outputs

Saved under `data/interim/`.

Typical files include:

* cleaned ComCat parquet
* ComCat cleaning log CSV
* raw moment tensor parquet, when GCMT input is provided
* cleaned moment tensor parquet
* moment tensor cleaning log CSV
* enriched event parquet
* trigger parquet
* labels parquet
* feature parquet

These files are useful for debugging the dataset build process.

### 10.2 Processed outputs

Saved under `data/processed/`.

Typical structure:

```text
data/processed/
├── datasets/
│   └── <dataset_name>.parquet
└── splits/
    └── <dataset_name>/
        ├── train.parquet
        ├── val.parquet
        └── test.parquet
```

These files are the canonical modeling inputs.

### 10.3 Metrics and predictions

Traditional baseline outputs are saved under:

```text
reports/metrics/
```

Neural model outputs are saved under:

```text
reports/metrics/<run_name>/
reports/checkpoints/<run_name>/
```

Typical neural metric and prediction files include:

```text
<run_name>_probability_predictions.csv
<run_name>_count_predictions.csv
<run_name>_magnitude_predictions.csv
<run_name>_probability_metrics.csv
<run_name>_count_metrics.csv
<run_name>_count_capped_metrics.csv
<run_name>_magnitude_metrics.csv
<run_name>_history.csv
```

---

## 11. Modeling Utilities

Reusable modeling-stage code lives under `src/models/` and `src/evaluation/`.

### 11.1 Feature and input utilities

```text
src/models/feature_sets.py
```

Defines modeling-stage feature sets, including:

* `NN_CORE_V1`
* `NN_ENRICHED_V1`

```text
src/models/input_layer.py
```

Handles:

* loading processed splits
* resolving requested feature columns
* train-only imputation
* optional train-only scaling
* numeric tabular input preparation

### 11.2 Baselines

```text
src/models/baselines/
```

Contains traditional baseline model implementations, including:

* climatology-style baseline
* simplified RJ-style baseline

```text
src/models/run_baselines.py
```

Runs current baselines and returns standardized prediction and metric tables.

### 11.3 Neural modules

```text
src/models/neural/
```

Contains the modular neural-network pipeline:

| File            | Responsibility                                                        |
| --------------- | --------------------------------------------------------------------- |
| `data.py`       | Prepares shared neural inputs and target tables from processed splits |
| `model.py`      | Defines the multi-task MLP architecture                               |
| `tensors.py`    | Converts prepared data into tensors and DataLoaders                   |
| `losses.py`     | Computes probability, count, and masked magnitude losses              |
| `training.py`   | Owns training epochs, validation, checkpointing, and history          |
| `prediction.py` | Runs inference and formats prediction tables                          |
| `artifacts.py`  | Builds standardized run output and checkpoint paths                   |
| `utils.py`      | Provides seed, device, and repository path helpers                    |

### 11.4 Evaluation utilities

```text
src/evaluation/metrics.py
```

Contains shared metric primitives such as binary probability metrics and regression-style MAE/RMSE helpers.

```text
src/evaluation/validation.py
```

Contains prediction-table validation helpers.

```text
src/evaluation/neural_metrics.py
```

Contains grouped neural metric computation for:

* probability metrics
* uncapped count metrics
* capped count metrics
* conditional magnitude metrics

Current capped count evaluation uses:

```text
24h cap = 300+
72h cap = 500+
```

The capping is evaluation-only and does not change training targets.

---

## 12. Project Structure

```text
CDS_Group10/
├── README.md
├── docs/
│   ├── codex/
│   │   ├── refactor_multitask_mlp_pipeline.md
│   │   └── ...
│   └── ...
├── scripts/
│   ├── download_raw_data.py
│   ├── run_and_save_baselines.py
│   ├── train_multitask_mlp.py
│   └── evaluate_multitask_predictions.py
├── src/
│   ├── data/
│   │   ├── comcat/ or source-specific loading helpers
│   │   └── moment_tensor/ or GCMT parsing helpers
│   ├── dataset/
│   │   ├── build_dataset.py
│   │   ├── labels.py
│   │   ├── assemble.py
│   │   └── shared dataset construction utilities
│   ├── models/
│   │   ├── feature_sets.py
│   │   ├── input_layer.py
│   │   ├── run_baselines.py
│   │   ├── baselines/
│   │   └── neural/
│   │       ├── data.py
│   │       ├── model.py
│   │       ├── tensors.py
│   │       ├── losses.py
│   │       ├── training.py
│   │       ├── prediction.py
│   │       ├── artifacts.py
│   │       └── utils.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── validation.py
│   │   └── neural_metrics.py
│   └── utils/
│       └── shared project utilities
├── data/
│   ├── raw/
│   │   ├── comcat/
│   │   └── moment_tensor/
│   ├── interim/
│   └── processed/
│       ├── datasets/
│       └── splits/
├── reports/
│   ├── metrics/
│   │   ├── baselines_predictions.csv
│   │   ├── baselines_metrics.csv
│   │   └── <run_name>/
│   └── checkpoints/
│       └── <run_name>/
├── tests/
└── archive/
    └── v1/
```

### 12.1 Main directories

| Path                   | Purpose                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `scripts/`             | Runnable entrypoints for downloading data, building/running models, and evaluation |
| `src/data/`            | Source-level loading, cleaning, parsing, and enrichment helpers                    |
| `src/dataset/`         | Canonical trigger-level dataset construction pipeline                              |
| `src/models/`          | Modeling-stage utilities, baselines, neural models, and input preparation          |
| `src/evaluation/`      | Shared metric and prediction-validation helpers                                    |
| `data/raw/`            | Raw downloaded or manually provided source data                                    |
| `data/interim/`        | Intermediate artifacts from dataset construction                                   |
| `data/processed/`      | Final reusable datasets and split-specific parquet files                           |
| `reports/metrics/`     | Saved predictions, metrics, and training histories                                 |
| `reports/checkpoints/` | Saved neural model checkpoints                                                     |
| `docs/codex/`          | Implementation notes and Codex handoff documents                                   |
| `tests/`               | Unit and integration tests                                                         |
| `archive/`             | Older archived versions of the pipeline                                            |

---

## 13. Main Rule for Collaborators

Use the repository in this order:

1. download or prepare raw data
2. build the processed dataset
3. run baselines
4. run neural models or other model experiments using the saved processed splits
5. compare models using shared evaluation utilities

Do not create separate dataset pipelines for different models. All models should consume the same processed train / validation / test splits so that comparisons remain fair.
