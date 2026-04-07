# CDS_Group10

Earthquake aftershock forecasting repository with a shared trigger-level preprocessing pipeline.

## Problem Setup

Trigger earthquake:

- magnitude `>= 5.0`

Primary targets:

- `y_24h`
- `y_72h`

Secondary targets:

- `n_aftershocks_24h`
- `n_aftershocks_72h`

Aftershock definition:

- `event_time > trigger_time`
- `event_time <= trigger_time + horizon`
- magnitude `>= 2.5`
- distance `<= 50 km`

Time split:

- train = `2015-2023`
- val = `2024`
- test = `2025`

## Shared Preprocessing Pipeline

Run the shared preprocessing pipeline before any model training.

The canonical entrypoint is [`src/dataset/build_dataset.py`](/Users/gynnyeo/Documents/GitHub/CDS_Group10/src/dataset/build_dataset.py).

What it does:

1. loads raw ComCat from a file or discovers supported files inside a folder
2. cleans the raw ComCat master table
3. optionally loads and parses Global CMT from a file or folder
4. cleans Global CMT data
5. left-joins GCMT enrichment onto ComCat using the existing fuzzy-match rules
6. builds trigger rows
7. builds shared labels
8. builds shared features
9. assembles the final trigger-level dataset
10. assigns `train` / `val` / `test` splits
11. saves reusable parquet outputs for later modeling

Project architecture:

- `src/data`: source-level loading, cleaning, downloading, and enrichment
- `src/dataset`: trigger-level dataset construction, labels, features, assembly, and splits

Model-specific preprocessing should happen later and remain separate from this shared pipeline.

## Data Sources

ComCat is the master dataset and source of truth for trigger construction, label generation, and historical features.

`--comcat-input` can point to:

- a single `.geojson`, `.csv`, or `.parquet` file
- a directory containing supported ComCat raw files

When a directory is passed, the pipeline discovers supported files recursively, loads them in deterministic sorted order, concatenates them, and then cleans the combined raw table.

Global CMT is optional left-joined enrichment. Unmatched ComCat rows are preserved. The GCMT merge logic is intentionally conservative and should not be weakened.

`--moment-tensor-input` can point to:

- a single `.ndk`, `.csv`, or `.parquet` file
- a directory containing supported GCMT files

When a directory is passed, the pipeline discovers supported files recursively, loads them in deterministic sorted order, concatenates them, and then cleans the combined raw table.

## Outputs

Interim outputs are saved under `data/interim/`:

- cleaned ComCat parquet
- ComCat cleaning log csv
- raw moment tensor parquet when GCMT input is provided
- cleaned moment tensor parquet
- moment tensor cleaning log csv
- enriched event parquet
- trigger parquet
- labels parquet
- features parquet

Processed outputs are saved under `data/processed/`:

- final dataset parquet in [`data/processed/datasets`](/Users/gynnyeo/Documents/GitHub/CDS_Group10/data/processed/datasets)
- split-specific parquet files in [`data/processed/splits`](/Users/gynnyeo/Documents/GitHub/CDS_Group10/data/processed/splits)

Filenames are based on the provided dataset name after slugging to lowercase underscore form.

## How To Run

ComCat folder input:

```bash
python -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --dataset-name earthquake_aftershock_v2
```

ComCat single-file input:

```bash
python -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat/comcat.parquet \
  --dataset-name earthquake_aftershock_v2
```

ComCat plus GCMT folders:

```bash
python -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --moment-tensor-input data/raw/moment_tensor \
  --dataset-name earthquake_aftershock_v2_gcmt
```

Force recompute:

```bash
python -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --moment-tensor-input data/raw/moment_tensor \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --force-recompute
```

Optional flags:

- `--moment-tensor-input` for GCMT enrichment
- `--comcat-input` accepts either a file or a folder
- `--moment-tensor-input` accepts either a file or a folder
- `--dataset-name` to control saved artifact names
- `--force-recompute` to ignore cached outputs
- `--quiet` to reduce logging

## Project Layout

```text
src/data/              source-level loading, cleaning, enrichment
src/dataset/           trigger-level shared dataset construction
data/raw/              raw source inputs
data/interim/          saved intermediate pipeline artifacts
data/processed/        final datasets and split-specific outputs
archive/v1/            archived older pipeline
```


## Current Status

Shared preprocessing pipeline is completed and validated.

- Final dataset successfully built (2015–2025)
- Trigger-level dataset (~18k rows)
- Train / validation / test splits verified

Next step:
- Implement baselines (climatology, Reasenberg–Jones)
- Build model input layer