# Running the Multi-Task Neural Network (MLP)

This guide explains how to train and evaluate the refactored multi-task neural network for earthquake aftershock forecasting.

The current MLP predicts three target families for 24h and 72h horizons:

* probability of at least one aftershock
* number of aftershocks
* maximum aftershock magnitude, conditional on an aftershock existing

The pipeline now supports two workflows:

1. full training + evaluation
2. evaluation-only from saved prediction CSVs

---

## 1. Prerequisites

### 1.1 Environment

Ensure the project environment is activated and dependencies are installed.

Typical dependencies include:

```bash
pip install torch numpy pandas scikit-learn tqdm pyarrow
```

If your project already has a requirements file, prefer installing from that instead of manually installing packages.

### 1.2 Processed Dataset

The processed dataset and train/validation/test splits must already exist.

Example dataset build command with GCMT enrichment:

```bash
python3 -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --moment-tensor-input data/raw/moment_tensor \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --force-recompute
```

Example dataset build command without GCMT enrichment:

```bash
python3 -m src.dataset.build_dataset \
  --comcat-input data/raw/comcat \
  --dataset-name earthquake_aftershock_v2 \
  --force-recompute
```

The neural scripts expect the processed dataset and split files to be available through the repository's existing processed-data paths.

---

## 2. Main Scripts

### 2.1 Full Training + Evaluation

Use:

```bash
python3 scripts/train_multitask_mlp.py
```

This script:

1. loads processed train/validation/test splits
2. prepares neural inputs
3. builds DataLoaders
4. builds the MLP
5. trains the model
6. saves best and last checkpoints
7. reloads the best checkpoint
8. generates prediction CSVs
9. computes metrics
10. saves all outputs

### 2.2 Evaluation Only

Use:

```bash
python3 scripts/evaluate_multitask_predictions.py
```

This script:

1. reads saved prediction CSVs
2. recomputes probability, count, capped-count, and magnitude metrics
3. saves metric CSVs

It does not train a model and does not load checkpoints.

This is useful when:

* metrics code changes
* capped evaluation is added or updated
* predictions from another model need to be evaluated using the same metric functions
* you want to regenerate metrics without spending time retraining

---

## 3. Full Training Usage

### 3.1 Recommended Enriched Run

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_mag_evalcap24h300_72h500 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0
```

This run uses:

* GCMT-enriched feature set
* learning rate `3e-4`
* gradient clipping with max norm `1.0`
* count loss weight `1.0`
* magnitude loss weight `1.0`
* horizon-specific capped count evaluation:

```text
24h cap = 300+
72h cap = 500+
```

### 3.2 Core Model Comparison Run

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_core_v1 \
  --run-name mlp_multitask_v1_core_mag_evalcap24h300_72h500 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0
```

This gives a fair comparison between the core and enriched feature sets under the same stabilized training settings.

### 3.3 One-Epoch Smoke Test

Use this after code changes to confirm the pipeline still runs:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name refactor_smoke_test \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --epochs 1
```

### 3.4 Conditional Count Experiment

Use `--count-modeling-mode conditional_positive` to run the new conditional-count experiment.

In this mode:

* count loss is computed only on rows with positive count targets, separately for each horizon
* zero-count rows do not contribute to the count regression loss
* the count head output is interpreted as positive-case severity at inference time
* the final count prediction written to the standard count CSV becomes:

```text
sigmoid(prob_logits) * clip(expm1(count_pred), min=0)
```

Standard mode remains available and unchanged with:

```text
--count-modeling-mode standard
```

Important interpretation note:

* `train_count_loss` and `val_count_loss` are not directly comparable across `standard` and `conditional_positive` runs, because the count-loss population differs once zero-count rows are masked out

Example command using the current best shared architecture (`256 128 64`, dropout `0.3`):

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_conditional_positive_256_128_64 \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --count-modeling-mode conditional_positive
```

This experiment changes both count-loss masking and count inference semantics. Probability and magnitude outputs keep their existing training and CSV formats.

Suggested 1-epoch smoke test:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name conditional_positive_smoke_test \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --count-modeling-mode conditional_positive \
  --epochs 1
```

After the smoke test, inspect:

* training finishes without crashing, including batches that may contain no positive counts for one or both horizons
* the exported count prediction CSV still has the standard columns: `trigger_event_id`, `split`, `horizon`, `model_name`, `y_true`, `y_pred`
* count predictions in the CSV remain non-negative
* when comparing runs, only compare `train_count_loss` and `val_count_loss` within the same count mode


### 3.5 Probability-Focused Experiment (Reduced Count Weight + Optional Focal Loss)

After the conditional-count experiments, the next targeted direction is to improve:

* probability prediction
* maximum aftershock magnitude

without redesigning the model architecture.

The main finding from the recent loss-weight sweep was:

* reducing `count_loss_weight` from `1.0` to `0.5` gave the best balance
* this slightly improved probability and magnitude
* reducing it too far to `0.25` did not help further

This suggests that the count task was influencing the shared trunk enough to slightly interfere with the probability and magnitude tasks, but still provides useful shared signal.

For the next set of runs, use:

```text
--count-loss-weight 0.5
```

as the new preferred setting when the focus is probability and magnitude rather than count.

#### Recommended baseline command for this phase

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_bce_countw05_enriched_256_128_64_d03_lr3e4 \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --count-loss-weight 0.5
```

This keeps the current best shared architecture unchanged while reducing count-task pressure on the shared representation.

#### Optional focal-loss experiment for probability

If binary focal loss has been added to the training pipeline, the next probability-focused experiment is to replace standard BCE with focal loss while keeping the rest of the setup fixed.

Recommended first focal run:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_focalg15_countw05_enriched_256_128_64_d03_lr3e4 \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --count-loss-weight 0.5 \
  --probability-loss focal \
  --focal-gamma 1.5
```

If needed, a milder variant can also be tested:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_focalg10_countw05_enriched_256_128_64_d03_lr3e4 \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --count-loss-weight 0.5 \
  --probability-loss focal \
  --focal-gamma 1.0
```

#### Interpretation notes

This experiment is intended to improve probability discrimination while keeping magnitude stable.

When comparing runs, focus on:

* probability ROC-AUC
* probability Brier score
* probability log loss
* conditional magnitude MAE / RMSE

Important note:

* once focal loss is used, training-loss values are no longer directly comparable to BCE runs in a like-for-like way
* final judgment should come from the saved probability and magnitude metrics, not just validation total loss

#### Current recommendation

For probability- and magnitude-focused runs, the current preferred configuration is:

* feature set: `nn_enriched_v1`
* hidden dims: `256 128 64`
* dropout: `0.3`
* learning rate: `3e-4`
* count loss weight: `0.5`

The focal-loss experiment should be treated as an incremental extension of this setup, not a new model family.

```

## Small wording tweak I would also make earlier in the file

In the earlier “Recommended Enriched Run” section, I would leave the old command if you want the file to preserve historical defaults, but add one short note under it:

```md
Note: for newer probability- and magnitude-focused experiments, `--count-loss-weight 0.5` is now preferred over `1.0`.
```

That way the document still preserves older runs, but the newer recommendation is clear.

## If you want the shortest possible update

If you do not want to add a full section, then add just this short note:

```md
### Update: Preferred Setting for Probability / Magnitude Experiments

For recent probability- and magnitude-focused runs, the preferred training setup keeps the current best shared architecture but reduces count-task influence:

* feature set: `nn_enriched_v1`
* hidden dims: `256 128 64`
* dropout: `0.3`
* learning rate: `3e-4`
* count loss weight: `0.5`

Recommended command:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_bce_countw05_enriched_256_128_64_d03_lr3e4 \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --count-loss-weight 0.5
```

This follows the recent finding that `count_loss_weight = 0.5` gives a better balance for probability and magnitude than `1.0`, while `0.25` reduces count influence too much.



---

## 4. Evaluation-Only Usage

### 4.1 Recompute Metrics for an Existing Run

If prediction CSVs already exist, run:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_multitask_v1_enriched_mag_evalcap24h300_72h500
```

By default, this reads from:

```text
reports/metrics/<run_name>/
```

and writes updated metric files to the same run folder.

### 4.2 Skip Magnitude Evaluation

If a run does not have magnitude prediction CSVs:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name some_run_without_magnitude_predictions \
  --skip-magnitude
```

### 4.3 Override the Model Name in Metrics

If you want the metric tables to use a different `model_name` value:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_multitask_v1_enriched_mag_evalcap24h300_72h500 \
  --model-name mlp_refactored_eval
```

This changes the `model_name` column before grouped metrics are computed.

---
## 5. Important Training Arguments

### Core Arguments

| Argument          | Description                                       |
| ----------------- | ------------------------------------------------- |
| `--dataset-name`  | Processed dataset name to load                    |
| `--feature-set`   | Feature set (e.g. `nn_core_v1`, `nn_enriched_v1`) |
| `--run-name`      | Name used for output folders and file prefixes    |
| `--epochs`        | Number of training epochs                         |
| `--batch-size`    | Mini-batch size                                   |
| `--learning-rate` | Adam learning rate                                |
| `--weight-decay`  | Adam weight decay                                 |
| `--seed`          | Random seed                                       |

---

### Model Architecture

| Argument                         | Description                                           |
| -------------------------------- | ----------------------------------------------------- |
| `--hidden-dims`                  | Shared trunk layer sizes (e.g. `256 128 64`)          |
| `--dropout`                      | Dropout rate in shared trunk                          |
| `--probability-head-hidden-dims` | Optional probability head tower (e.g. `64`, `128 64`) |
| `--count-head-hidden-dims`       | Optional count head tower                             |
| `--magnitude-head-hidden-dims`   | Optional magnitude head tower                         |

---

### Loss & Training Behavior

| Argument                | Description                          |
| ----------------------- | ------------------------------------ |
| `--count-loss-weight`   | Weight applied to count loss         |
| `--count-modeling-mode` | `standard` or `conditional_positive` |
| `--probability-loss`    | `bce` or `focal`                     |
| `--focal-gamma`         | Gamma parameter for focal loss       |

---

### Data & Preprocessing

| Argument             | Description                     |
| -------------------- | ------------------------------- |
| `--missing-strategy` | Missing-value handling strategy |
| `--no-scale`         | Disable feature scaling         |

---

### System & Paths

| Argument           | Description                                 |
| ------------------ | ------------------------------------------- |
| `--device`         | `auto`, `cpu`, or `cuda`                    |
| `--output-dir`     | Directory for metrics, predictions, history |
| `--checkpoint-dir` | Directory for checkpoints                   |


Default output locations are run-scoped:
```text
reports/metrics/<run_name>/
reports/checkpoints/<run_name>/
```

---

## 6. Output Files

For a run named:

```text
<run_name>
```

outputs are saved to:

```text
reports/metrics/<run_name>/
```

### 6.1 Prediction Files

```text
<run_name>_probability_predictions.csv
<run_name>_count_predictions.csv
<run_name>_magnitude_predictions.csv
```

Probability prediction columns:

```text
trigger_event_id
split
horizon
model_name
y_true
y_prob
```

Count prediction columns:

```text
trigger_event_id
split
horizon
model_name
y_true
y_pred
```

Magnitude prediction columns:

```text
trigger_event_id
split
horizon
model_name
y_true
y_pred
target_available
```

### 6.2 Metric Files

```text
<run_name>_probability_metrics.csv
<run_name>_count_metrics.csv
<run_name>_count_capped_metrics.csv
<run_name>_magnitude_metrics.csv
```

The count metric files have different meanings:

```text
<run_name>_count_metrics.csv
```

contains uncapped raw-count metrics.

```text
<run_name>_count_capped_metrics.csv
```

contains horizon-specific capped-count metrics:

```text
24h: min(count, 300)
72h: min(count, 500)
```

Training still uses uncapped count targets. Capping is evaluation-only.

### 6.3 Training History

```text
<run_name>_history.csv
```

Contains per-epoch:

* training total loss
* training probability loss
* training count loss
* training magnitude loss
* validation total loss
* validation probability loss
* validation count loss
* validation magnitude loss
* epoch duration

### 6.4 Checkpoints

Checkpoints are saved to:

```text
reports/checkpoints/<run_name>/
```

Files:

```text
<run_name>_best.pt
<run_name>_last.pt
```

The best checkpoint is selected by lowest validation total loss.

---

## 7. Model Behavior

The multitask MLP keeps a shared trunk with three output heads:

* probability head
* count head
* magnitude head

By default, the probability head, count head, and magnitude head are all linear output heads.

If head-specific hidden dims are provided, the corresponding head becomes a small task-specific MLP tower on top of the shared trunk.

Available optional head-tower arguments:

* `--probability-head-hidden-dims`
* `--count-head-hidden-dims`
* `--magnitude-head-hidden-dims`

Example:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03_counttower64 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 128 64 \
  --count-head-hidden-dims 64 \
  --dropout 0.3
```

With that configuration, the architecture is:

```text
shared trunk: input -> 256 -> 128 -> 64
probability head: 64 -> 2
count tower: 64 -> 64 -> 2
magnitude head: 64 -> 2
```

### 7.1 Probability Outputs

The probability head outputs logits during training. Sigmoid is applied during inference to produce probabilities.

### 7.2 Count Outputs

Count training uses transformed targets:

```text
log1p(count)
```

During inference:

```text
count_prediction = expm1(model_output)
```

Negative predictions after inverse transformation are clipped to zero.

### 7.3 Magnitude Outputs

The magnitude head predicts maximum aftershock magnitude for each horizon.

Magnitude targets are only defined when at least one aftershock exists. Rows without a magnitude target are masked out of the magnitude loss.

Magnitude metrics are conditional metrics computed only where `target_available == True`.


### 7.4 Optional Probability and Magnitude Towers

The multi-task MLP now also supports optional task-specific towers for:

* probability head
* magnitude head

This extends the same design already used for the optional count tower.

By default, all three heads remain backward compatible:

* probability head: linear output head
* count head: linear output head unless `--count-head-hidden-dims` is provided
* magnitude head: linear output head

If tower hidden dims are provided for a head, that head is replaced by a small MLP tower built on top of the shared trunk output.

New optional training arguments:

* `--probability-head-hidden-dims`
* `--count-head-hidden-dims`
* `--magnitude-head-hidden-dims`

Each accepts zero or more integers. Examples:

```bash
--probability-head-hidden-dims 64
--magnitude-head-hidden-dims 64
--probability-head-hidden-dims 128 64
````

If omitted, the corresponding head remains a single linear layer.

#### Why use head-specific towers?

These towers are a small, targeted way to give a task more task-specific nonlinear capacity without changing:

* dataset definitions
* shared trunk structure
* output dictionary
* training targets
* prediction CSV schemas

This is especially useful when:

* probability is already strong but may benefit from a more expressive task-specific mapping
* magnitude is stable but slightly conservative and may benefit from extra task-specific capacity
* you want to explore task-specific improvements without redesigning the full model

#### Recommended experiment order

To keep experiments interpretable, do not enable both towers immediately in the first run.

Recommended order:

1. magnitude tower only
2. probability tower only
3. both towers together

This makes it easier to tell which task-specific tower is helping.

#### Recommended baseline for this phase

Keep:

* feature set: `nn_enriched_v1`
* hidden dims: `256 128 64`
* dropout: `0.3`
* learning rate: `3e-4`
* count loss weight: `0.5`
* probability loss: `bce`

This remains the current preferred baseline for probability- and magnitude-focused experiments.

#### Example: magnitude tower only

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_magtower64_bce_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5 \
  --probability-loss bce \
  --magnitude-head-hidden-dims 64
```

Architecture in this run:

```text
shared trunk: input -> 256 -> 128 -> 64
probability head: 64 -> 2
count head: 64 -> 2
magnitude tower: 64 -> 64 -> 2
```

#### Example: probability tower only

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_probtower64_bce_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5 \
  --probability-loss bce \
  --probability-head-hidden-dims 64
```

Architecture in this run:

```text
shared trunk: input -> 256 -> 128 -> 64
probability tower: 64 -> 64 -> 2
count head: 64 -> 2
magnitude head: 64 -> 2
```

#### Example: probability + magnitude towers together

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_probtower64_magtower64_bce_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5 \
  --probability-loss bce \
  --probability-head-hidden-dims 64 \
  --magnitude-head-hidden-dims 64
```

Architecture in this run:

```text
shared trunk: input -> 256 -> 128 -> 64
probability tower: 64 -> 64 -> 2
count head: 64 -> 2
magnitude tower: 64 -> 64 -> 2
```

#### Interpretation notes

These tower experiments do not change:

* probability target definition
* count target definition
* magnitude masking logic
* output dict keys
* prediction CSV formats

They only change how the shared trunk features are mapped into the corresponding task head.

When comparing runs, focus on:

* probability ROC-AUC, Brier score, and log loss
* magnitude MAE / RMSE
* whether magnitude mean prediction becomes less conservative
* whether gains from one tower remain when both towers are enabled together


---

## 8. Typical Workflow

### Step 1: Train the enriched model

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_enriched_stable \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0
```

### Step 2: Train the core model

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_core_v1 \
  --run-name mlp_core_stable \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0
```

### Step 3: Regenerate metrics if needed

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_enriched_stable
```

### Step 4: Compare outputs

Compare files under:

```text
reports/metrics/mlp_enriched_stable/
reports/metrics/mlp_core_stable/
```

Important comparisons:

* probability ROC-AUC, Brier score, and log loss
* uncapped count MAE/RMSE
* capped count MAE/RMSE
* cap+ missed cases and false alarms
* conditional magnitude MAE/RMSE

---

## 9. Troubleshooting and Tips

### Training is unstable

Try:

```bash
--learning-rate 3e-4
```

The enriched model is more sensitive than the core model because it contains additional sparse/imputed GCMT features.

### Count predictions are unstable

Current stabilizers already include:

* `log1p(count)` target transform
* gradient clipping
* lower learning rate for enriched runs

Avoid increasing `--count-loss-weight` too aggressively, because earlier experiments with count weight `2.0` worsened count RMSE.

### Model may be overfitting

Try:

```bash
--dropout 0.3
```

or smaller hidden layers:

```bash
--hidden-dims 64 32
```

### GPU is available

Use:

```bash
--device cuda
```

or let the script choose automatically:

```bash
--device auto
```

### You only changed metric code

Do not retrain. Run:

```bash
python3 scripts/evaluate_multitask_predictions.py --run-name <run_name>
```

---

## 10. Notes for Fair Evaluation

* Do not compare neural capped count metrics against baseline uncapped count metrics.
* If capped metrics are reported for the neural model, apply the same cap logic to baseline predictions.
* Do not drop rows with missing GCMT features.
* Do not modify the dataset pipeline during neural training experiments.
* Use the same train/validation/test splits when comparing models.
* Capped count evaluation is diagnostic only; it does not mean the model was trained on capped counts.

---

## 11. Next Suggested Analyses

After training and basic evaluation, the most useful next analyses are:

* count metrics only for rows where true count is positive
* count metrics by true-count bucket
* inspection of missed `300+` and `500+` cases
* comparison against traditional baselines under both uncapped and capped count metrics
* probability calibration analysis
* learning-curve plots from the history CSV
