# Neural Network Modeling for Aftershock Forecasting

## 1. Overview

This document is an intermediate working log for the neural-network part of the earthquake aftershock forecasting project. It records the current model design, experiments attempted so far, results, interpretation, and possible next steps before writing the final report.

The project uses a **trigger-level aftershock forecasting setup**. Each row in the modeling dataset represents a trigger earthquake:

```text
trigger earthquake = earthquake with magnitude >= 5.0
```

For each trigger earthquake, the neural model predicts three target families over two forecast horizons, 24 hours and 72 hours:

| Target family     | 24h target                     | 72h target                     | Meaning                                                                               |
| ----------------- | ------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------- |
| Probability       | `y_24h`                        | `y_72h`                        | Whether at least one associated aftershock occurs                                     |
| Count             | `n_aftershocks_24h`            | `n_aftershocks_72h`            | Number of associated aftershocks                                                      |
| Maximum magnitude | `max_aftershock_magnitude_24h` | `max_aftershock_magnitude_72h` | Maximum magnitude among associated aftershocks, conditional on an aftershock existing |

The current neural model is a **multi-task PyTorch MLP** with a shared trunk and separate heads for probability, count, and maximum magnitude.

The project uses:

* **ComCat** as the primary earthquake event catalogue.
* **GCMT / moment tensor data** as optional enrichment.
* A shared processed dataset and shared train / validation / test split so neural models and traditional baselines can be compared fairly.

GCMT is left-joined onto the ComCat trigger-event dataset. Missing GCMT is allowed and handled later by the modeling-stage imputation pipeline. No ComCat trigger rows are dropped simply because GCMT information is unavailable.

---

## 2. Dataset, Label, and Split Setup

### 2.1 Aftershock association definition

An aftershock is associated with a trigger if it satisfies:

```text
event_time > trigger_time
event_time <= trigger_time + horizon
aftershock magnitude >= 2.5
distance <= 50 km
```

The two forecast horizons are:

```text
24 hours
72 hours
```

### 2.2 Current time split

The current split setup is:

```text
train = 2010–2022
validation = 2023
test = 2024–2025
```

Approximate split sizes:

| Split      | Approximate number of trigger rows |
| ---------- | ---------------------------------: |
| Train      |                            ~23,000 |
| Validation |                             ~1,700 |
| Test       |                             ~3,500 |

The test split is future-year data relative to the training and validation periods. For that reason, the validation set should be used for experiment selection, while the test set should be used mainly for final confirmation.

### 2.3 Feature sets

Two neural feature configurations are currently used.

#### `nn_core_v1`

This is the core ComCat-only feature set. It includes features such as:

* trigger latitude
* trigger longitude
* trigger depth
* trigger magnitude
* time-derived features
* prior global activity features
* event quality features such as `gap`, `dmin`, `rms`, and `nst`

This is the cleanest neural baseline because these features are available for all trigger events.

#### `nn_enriched_v1`

This includes all `nn_core_v1` features plus optional GCMT features, such as:

* `has_gcmt`
* GCMT match distance / time difference / magnitude difference
* GCMT location and depth
* strike, dip, rake
* moment tensor components
* eigenvalue / eigenvector features
* scalar moment and related focal-mechanism features

Important design choice:

```text
GCMT is optional enrichment, not a row filter.
```

This keeps the core and enriched models comparable because both are trained on the same trigger-event population.

---

## 3. Current Neural Model

### 3.1 Architecture

The current neural model is a shared-trunk MLP with three task heads.

The model returns the following dictionary:

```python
{
    "prob_logits": ...,
    "count_pred": ...,
    "magnitude_pred": ...,
}
```

This output structure is important because the training, loss, prediction, and metric modules expect these keys.

The default architecture is:

```text
shared trunk: 128 -> 64
dropout: 0.2
```

The three heads are:

| Head             | Output shape | Meaning                                            |
| ---------------- | -----------: | -------------------------------------------------- |
| Probability head |            2 | 24h and 72h logits                                 |
| Count head       |            2 | 24h and 72h predictions in transformed count space |
| Magnitude head   |            2 | 24h and 72h max-magnitude predictions              |

The architecture can be changed from the command line using:

```bash
--hidden-dims
--dropout
```

This was useful for the later larger-MLP experiments.

### 3.2 Probability task

The probability task predicts whether there is at least one associated aftershock within the horizon.

Targets:

```text
y_24h
y_72h
```

Loss:

```text
BCEWithLogitsLoss
```

The model outputs logits during training. Sigmoid is applied during inference to produce probabilities.

### 3.3 Count task

The count task predicts the number of associated aftershocks within the horizon.

Targets:

```text
n_aftershocks_24h
n_aftershocks_72h
```

The count targets are highly right-skewed and include many zeros. To stabilize training, count targets are transformed using:

```text
log1p(count) = log(1 + count)
```

This is useful because:

```text
log1p(0) = 0
```

so zero-count events remain valid, while very large counts are compressed.

Loss:

```text
SmoothL1Loss
```

During inference, the model's count outputs are converted back to raw-count scale using:

```text
raw_count_prediction = expm1(model_count_output)
```

Negative inverse-transformed predictions are clipped to zero because negative aftershock counts are not meaningful.

### 3.4 Maximum magnitude task

The maximum magnitude task predicts the maximum aftershock magnitude within each horizon, but only when at least one aftershock exists.

Targets:

```text
max_aftershock_magnitude_24h
max_aftershock_magnitude_72h
```

If no aftershock exists within the horizon, the target is stored as:

```text
NaN
```

This is intentional because maximum magnitude is undefined when there is no aftershock.

Magnitude loss is masked so only rows with available magnitude targets contribute to the magnitude loss.

Loss:

```text
SmoothL1Loss, applied only where target_available == True
```

### 3.5 Total loss

The current total training loss is:

```text
total_loss = probability_loss
           + count_loss_weight * count_loss
           + magnitude_loss_weight * magnitude_loss
```

Current settings:

```text
count_loss_weight = 1.0
magnitude_loss_weight = 1.0
```

The count loss weight is configurable from the command line. Magnitude loss weight is currently fixed at `1.0`.

---

## 4. Training Setup

The model is trained using:

* Adam optimizer
* mini-batch training with PyTorch `DataLoader`
* default batch size of 128
* validation-based best checkpoint selection
* gradient clipping with max norm `1.0`
* train-only imputation and optional train-only feature scaling

Each epoch records:

* training total loss
* training probability loss
* training count loss
* training magnitude loss
* validation total loss
* validation probability loss
* validation count loss
* validation magnitude loss
* epoch duration

Two checkpoints are saved:

| Checkpoint           | Meaning                      |
| -------------------- | ---------------------------- |
| `<run_name>_best.pt` | Lowest validation total loss |
| `<run_name>_last.pt` | Final epoch                  |

After training, the best checkpoint is reloaded before generating prediction tables and metrics.

---

## 5. Modular Refactor of the Neural Pipeline

The original neural training script became too long because it handled too many responsibilities in one file. The pipeline has now been refactored so the main script coordinates the workflow while reusable logic lives in modules.

### 5.1 Current module responsibilities

| File                                        | Responsibility                                                             |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `scripts/train_multitask_mlp.py`            | Full training + prediction + metric generation coordinator                 |
| `scripts/evaluate_multitask_predictions.py` | Evaluation-only metric regeneration from saved prediction CSVs             |
| `src/models/neural/data.py`                 | Prepares train / validation / test neural inputs and targets               |
| `src/models/neural/model.py`                | Defines `MultiTaskMLP` and `build_multitask_mlp`                           |
| `src/models/neural/tensors.py`              | Tensor conversion, `log1p` count tensors, magnitude masks, DataLoaders     |
| `src/models/neural/losses.py`               | Multi-task loss computation and masked magnitude loss                      |
| `src/models/neural/training.py`             | Epoch loops, validation, checkpoint saving, training history               |
| `src/models/neural/prediction.py`           | Inference and standardized prediction-table formatting                     |
| `src/models/neural/artifacts.py`            | Standardized run output and checkpoint paths                               |
| `src/models/neural/utils.py`                | Seed, device, and path helpers                                             |
| `src/evaluation/neural_metrics.py`          | Neural probability, count, capped-count, magnitude, and diagnostic metrics |

### 5.2 Why the refactor matters

The refactor did not change the scientific task. Its purpose was to make the code easier to extend, audit, and reuse.

Benefits:

* Saved prediction CSVs can be re-evaluated without retraining.
* New diagnostics can be added to evaluation code without changing training.
* Future architectures can reuse the same tensor, loss, prediction, and metric modules.
* The training script is now easier to understand because it coordinates rather than implements everything directly.

---

## 6. Baseline Neural Experiments Before Larger Architectures

### 6.1 Initial core model

Run name:

```text
mlp_multitask_v1_core
```

Purpose:

* Establish a neural baseline using ComCat/core features only.

Observation:

* Probability prediction learned meaningful signal.
* Count prediction was difficult but somewhat more stable than early enriched runs.

### 6.2 Initial enriched model

Run name:

```text
mlp_multitask_v1_enriched
```

Purpose:

* Test whether optional GCMT enrichment improved the MLP.

Observation:

* Probability performance remained reasonable.
* Count predictions were less stable.
* GCMT enrichment did not clearly outperform the core feature set.

### 6.3 Lower learning rate for enriched model

The enriched model was rerun with:

```bash
--learning-rate 3e-4
```

instead of:

```bash
--learning-rate 1e-3
```

Reasoning:

* The enriched model has many more input features.
* Many GCMT features are sparse or imputed.
* Lower learning rate reduces aggressive parameter updates and can stabilize training.

Result:

* Count stability improved.
* Probability performance remained reasonable.

### 6.4 Gradient clipping

Gradient clipping was added:

```python
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Reasoning:

* Earlier count predictions showed instability.
* Count targets are heavy-tailed, so some batches can cause unusually large gradients.
* Gradient clipping limits the size of parameter updates.

Result:

* Improved stability.
* No major degradation to probability performance.

### 6.5 Count loss weight experiment

A command-line argument was added:

```bash
--count-loss-weight
```

Experiment tried:

```bash
--count-loss-weight 2.0
```

Result:

* Count performance worsened.
* Increasing the count loss weight appeared to amplify instability rather than solve the count underprediction problem.

Current setting:

```bash
--count-loss-weight 1.0
```

---

## 7. Capped Count Evaluation

### 7.1 Motivation

The count targets are extremely right-skewed. Most trigger events have few or zero aftershocks, but a small number have very high aftershock counts.

Training-set count distribution:

```text
n_aftershocks_24h
count    23031
mean         6.67
std         25.34
min          0
25%          0
50%          0
75%          3
90%         17
95%         38
99%         87
99.5%      132.85
99.9%      291.82
max       1253
```

```text
n_aftershocks_72h
count    23031
mean        10.86
std         41.18
min          0
25%          0
50%          1
75%          5
90%         29
95%         59
99%        140
99.5%      214.70
99.9%      479.97
max       1643
```

This means raw count RMSE can be dominated by rare extreme sequences.

### 7.2 Evaluation-only capping

Capping was added only during evaluation, not training.

Training still uses:

```text
log1p(raw_count)
```

and saved predictions remain raw-count-scale predictions after `expm1`.

Capped evaluation uses:

```text
24h cap = 300+
72h cap = 500+
```

These caps are close to the 99.9th percentile of the training distributions.

For capped evaluation:

```text
y_true_capped = min(y_true, cap)
y_pred_capped = min(y_pred, cap)
```

The capped metric file also tracks:

* number of true cap+ cases
* number of predicted cap+ cases
* number of correctly predicted cap+ cases
* number of missed cap+ cases
* number of false cap+ alarms

### 7.3 Interpretation of capped evaluation

Capped evaluation reduced RMSE only slightly. It also showed that the model usually predicts no cap+ cases.

This means:

```text
The count problem is not only caused by a few extreme outliers.
The model also underpredicts moderate-to-high count events below the cap.
```

Important fairness rule:

```text
Do not compare neural capped metrics against baseline uncapped metrics.
```

If capped metrics are reported for the neural model, the same capped evaluation must be applied to baseline models.

---

## 8. Larger Shared-MLP Architecture Experiments

### 8.1 Motivation

After stabilizing the enriched MLP with lower learning rate and gradient clipping, the next question was whether model capacity was limiting performance.

The default architecture was:

```text
hidden_dims = 128 64
dropout = 0.2
```

Because the dataset has about 23k training rows and the enriched feature set has many GCMT-derived features, a moderately larger network was a reasonable next experiment. However, this was treated as a controlled capacity test, not a full redesign.

The goal was:

```text
Try larger networks while keeping the tabular trigger-level setup, output keys, losses, prediction schemas, and evaluation metrics unchanged.
```

### 8.2 Experiments run

Three larger shared-MLP experiments were run using existing CLI options only.

#### Experiment 1: moderate larger MLP

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do02 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 128 64 \
  --dropout 0.2
```

Purpose:

* Test a moderate capacity increase while keeping dropout at the previous value.

#### Experiment 2: moderate larger MLP with stronger dropout

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 128 64 \
  --dropout 0.3
```

Purpose:

* Test the same moderate larger MLP with stronger regularization.

#### Experiment 3: wider MLP

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_256_128_do03 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 256 128 \
  --dropout 0.3
```

Purpose:

* Test whether a wider shared trunk improves performance beyond the moderate model.

### 8.3 Probability results for larger architectures

Test probability metrics:

| Run                 | Horizon | Test ROC-AUC | Test Brier | Test log loss |
| ------------------- | ------: | -----------: | ---------: | ------------: |
| `h256_128_64_do02`  |     24h |       0.7484 |     0.2030 |        0.5874 |
| `h256_128_64_do02`  |     72h |       0.7307 |     0.2094 |        0.6032 |
| `h256_128_64_do03`  |     24h |   **0.7664** | **0.1972** |    **0.5746** |
| `h256_128_64_do03`  |     72h |   **0.7485** | **0.2042** |    **0.5913** |
| `h256_256_128_do03` |     24h |       0.7558 |     0.1991 |        0.5796 |
| `h256_256_128_do03` |     72h |       0.7411 |     0.2053 |        0.5948 |

Interpretation:

* The moderate larger model with dropout `0.3` was best for probability prediction.
* Stronger dropout helped generalization.
* The wider model did not outperform the moderate model on test.

### 8.4 Count results for larger architectures

Uncapped test count metrics:

| Run                 | Horizon |    Test MAE |   Test RMSE | Mean true | Mean pred |
| ------------------- | ------: | ----------: | ----------: | --------: | --------: |
| `h256_128_64_do02`  |     24h |      9.2071 |     28.8488 |    9.6268 |    1.6661 |
| `h256_128_64_do02`  |     72h |     13.8777 |     44.5540 |   14.4236 |    2.7166 |
| `h256_128_64_do03`  |     24h |  **9.0700** | **28.4269** |    9.6268 |    1.4777 |
| `h256_128_64_do03`  |     72h | **13.5435** | **42.2253** |   14.4236 |    2.2448 |
| `h256_256_128_do03` |     24h |      9.1604 |     28.6206 |    9.6268 |    1.6653 |
| `h256_256_128_do03` |     72h |     13.7368 |     43.1996 |   14.4236 |    2.5432 |

Interpretation:

* `h256_128_64_do03` had the best test count MAE and RMSE among the three larger models.
* Improvements were modest.
* Mean predicted counts remained far below mean true counts.
* Larger capacity helped slightly but did not solve the count problem.

### 8.5 Capped count results for larger architectures

Test capped count metrics:

| Run                 | Horizon | Test capped MAE | Test capped RMSE | True cap+ | Pred cap+ | Missed cap+ |
| ------------------- | ------: | --------------: | ---------------: | --------: | --------: | ----------: |
| `h256_128_64_do02`  |     24h |          9.0692 |          27.3073 |         6 |         1 |           6 |
| `h256_128_64_do02`  |     72h |         13.6420 |          41.5260 |         5 |         1 |           5 |
| `h256_128_64_do03`  |     24h |      **8.9493** |      **27.0131** |         6 |         0 |           6 |
| `h256_128_64_do03`  |     72h |     **13.4316** |      **40.8251** |         5 |         0 |           5 |
| `h256_256_128_do03` |     24h |          9.0397 |          27.2072 |         6 |         0 |           6 |
| `h256_256_128_do03` |     72h |         13.5859 |          41.4163 |         5 |         1 |           5 |

Interpretation:

* `h256_128_64_do03` again performed best on capped RMSE.
* However, it still predicted no test cap+ cases.
* The model still missed all true cap+ cases.
* The count issue remains broader than just extreme outliers.

### 8.6 Magnitude results for larger architectures

Test magnitude metrics:

| Run                 | Horizon |   Test MAE |  Test RMSE | Mean true | Mean pred |
| ------------------- | ------: | ---------: | ---------: | --------: | --------: |
| `h256_128_64_do02`  |     24h |     0.4930 |     0.6568 |    5.0155 |    5.0533 |
| `h256_128_64_do02`  |     72h |     0.4951 |     0.6537 |    5.0412 |    5.0875 |
| `h256_128_64_do03`  |     24h | **0.4779** | **0.6264** |    5.0155 |    4.9826 |
| `h256_128_64_do03`  |     72h | **0.4824** | **0.6285** |    5.0412 |    5.0190 |
| `h256_256_128_do03` |     24h |     0.4899 |     0.6533 |    5.0155 |    5.0087 |
| `h256_256_128_do03` |     72h |     0.4902 |     0.6491 |    5.0412 |    5.0381 |

Interpretation:

* The moderate larger model with dropout `0.3` also had the best magnitude metrics.
* Magnitude prediction remained stable.
* The larger model did not harm the magnitude task.

### 8.7 Larger-architecture conclusion

Best larger shared MLP:

```text
mlp_multitask_v1_enriched_h256_128_64_do03
```

Current recommended architecture:

```text
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 1.0
```

Main conclusion:

```text
A moderate increase in shared MLP capacity helped slightly, especially with dropout 0.3, but simply making the shared trunk larger did not solve count underprediction.
```

This means the count bottleneck is probably not just model capacity. The target distribution, loss behavior, `log1p` compression, and count-head design are likely important.

---

## 9. Count Diagnostics Added After Larger-MLP Experiments

After the larger architecture experiments, the main remaining question was:

```text
Where exactly is the count head failing?
```

To answer this, new evaluation-only diagnostics were added. These diagnostics do not retrain the model and do not change existing metric files. They only generate additional CSV outputs from saved count prediction CSVs.

### 9.1 New diagnostic outputs

For a run such as:

```text
mlp_multitask_v1_enriched_h256_128_64_do03
```

the evaluation-only script can now save additional files:

```text
<run_name>_count_positive_only_metrics.csv
<run_name>_count_bucket_diagnostics.csv
<run_name>_count_prediction_bucket_diagnostics.csv
```

These are diagnostic-only files.

### 9.2 Positive-count-only metrics

Purpose:

```text
Evaluate count performance only where y_true > 0.
```

This asks:

```text
When aftershocks actually occur, can the model estimate how many?
```

Results for the best current run, `h256_128_64_do03`:

| Split      | Horizon | Positive rows | Mean true | Mean pred |     MAE |    RMSE |
| ---------- | ------: | ------------: | --------: | --------: | ------: | ------: |
| Train      |     24h |        10,199 |   15.0643 |    4.3353 | 11.8867 | 35.6616 |
| Validation |     24h |           834 |   28.1451 |    2.2235 | 26.3820 | 61.3579 |
| Test       |     24h |         1,646 |   20.5462 |    2.4774 | 18.6814 | 41.5171 |
| Train      |     72h |        11,597 |   21.5652 |    6.3617 | 17.0589 | 54.0199 |
| Validation |     72h |           933 |   38.3848 |    3.0891 | 36.1314 | 85.7501 |
| Test       |     72h |         1,863 |   27.1981 |    3.5119 | 24.8174 | 57.9711 |

Interpretation:

* On positive test cases, the model predicts only about 12–13% of the true mean count.
* The count head is very conservative when aftershocks actually occur.
* All-row count metrics hide some of this severity problem because many rows have zero counts.

### 9.3 True-count bucket diagnostics

Purpose:

```text
Group rows by true count bucket and inspect average predicted count.
```

This answers:

```text
For true low-count, moderate-count, and high-count events, what does the model predict?
```

Buckets used:

```text
0
1
2–5
6–10
11–20
21–50
51–100
101–300
300+
```

#### Test 24h true-count bucket results

| True-count bucket |  Rows | Mean true | Mean pred | Pred / true |
| ----------------- | ----: | --------: | --------: | ----------: |
| 0                 | 1,867 |    0.0000 |    0.5963 |           — |
| 1                 |   463 |    1.0000 |    1.2095 |      1.2095 |
| 2–5               |   454 |    3.0220 |    1.5286 |      0.5058 |
| 6–10              |   164 |    7.7683 |    2.3782 |      0.3061 |
| 11–20             |   155 |   14.8452 |    3.0983 |      0.2087 |
| 21–50             |   179 |   33.6089 |    3.3272 |      0.0990 |
| 51–100            |   161 |   73.0000 |    4.7768 |      0.0654 |
| 101–300           |    64 |  131.5000 |    8.0214 |      0.0610 |
| 300+              |     6 |  370.6667 |   12.6031 |      0.0340 |

#### Test 72h true-count bucket results

| True-count bucket |  Rows | Mean true | Mean pred | Pred / true |
| ----------------- | ----: | --------: | --------: | ----------: |
| 0                 | 1,650 |    0.0000 |    0.8142 |           — |
| 1                 |   496 |    1.0000 |    1.4652 |      1.4652 |
| 2–5               |   501 |    2.9701 |    2.0342 |      0.6849 |
| 6–10              |   182 |    7.6593 |    2.7396 |      0.3577 |
| 11–20             |   152 |   14.9079 |    4.4211 |      0.2966 |
| 21–50             |   208 |   34.2115 |    4.8058 |      0.1405 |
| 51–100            |   172 |   72.8198 |    5.6040 |      0.0770 |
| 101–300           |   142 |  143.8028 |   10.6084 |      0.0738 |
| 300+              |    10 |  496.5000 |   15.6227 |      0.0315 |

Interpretation:

* Mean prediction generally increases as the true-count bucket increases.
* However, it increases far too slowly.
* The model is not random, but its predictions are heavily compressed.
* The count head is reasonable for zero and one-aftershock cases, but it severely underpredicts moderate and high counts.
* The problem starts well below the cap thresholds. For example, the 24h `21–50` bucket has mean true count 33.61 but mean prediction only 3.33.

Main conclusion:

```text
The count head has weak severity learning but severe scale compression.
```

### 9.4 Prediction-count bucket diagnostics

A second diagnostic was added to group by predicted count instead of true count.

Purpose:

```text
Group rows by predicted count bucket and inspect actual true count.
```

This answers:

```text
When the model predicts a higher count, are those rows actually higher-count events?
```

Prediction buckets used:

```text
0–1
1–2
2–5
5–10
10–20
20–50
50+
```

These buckets are lower than the true-count buckets because the model rarely predicts very large counts.

#### Test 24h prediction-bucket results

| Predicted-count bucket |  Rows | True positive rate | Mean true | Mean pred | Max true |
| ---------------------- | ----: | -----------------: | --------: | --------: | -------: |
| 0–1                    | 2,451 |             0.3464 |    4.0861 |    0.4274 |      130 |
| 1–2                    |   486 |             0.6399 |   11.6831 |    1.4018 |      151 |
| 2–5                    |   366 |             0.7760 |   20.2158 |    3.0249 |      370 |
| 5–10                   |   131 |             0.9389 |   45.7939 |    6.9872 |      389 |
| 10–20                  |    56 |             1.0000 |   49.5357 |   13.2021 |      159 |
| 20–50                  |    22 |             1.0000 |   83.8636 |   27.1917 |      381 |
| 50+                    |     1 |             1.0000 |  109.0000 |  102.4310 |      109 |

#### Test 72h prediction-bucket results

| Predicted-count bucket |  Rows | True positive rate | Mean true | Mean pred | Max true |
| ---------------------- | ----: | -----------------: | --------: | --------: | -------: |
| 0–1                    | 2,056 |             0.3711 |    5.0005 |    0.4952 |      200 |
| 1–2                    |   691 |             0.6671 |   13.4038 |    1.3744 |      228 |
| 2–5                    |   452 |             0.7611 |   24.9845 |    3.1065 |      585 |
| 5–10                   |   165 |             0.8909 |   45.6727 |    6.9470 |      495 |
| 10–20                  |   101 |             0.9901 |   70.9406 |   14.0036 |      603 |
| 20–50                  |    39 |             1.0000 |  110.3333 |   31.5101 |      605 |
| 50+                    |     9 |             1.0000 |   92.2222 |   80.5151 |      194 |

Interpretation:

* Higher predicted-count buckets generally correspond to higher true counts.
* True positive rate also increases strongly as predicted-count bucket increases.
* This means the count head has useful ranking/severity signal.
* However, the predicted values are still much lower than the true means in most higher predicted-count buckets.
* The count head is better at ranking severity than estimating the absolute count scale.

Important nuance:

* Some high true-count cases still appear in low predicted-count buckets.
* Calibration cannot fix cases that the model ranks too low.
* Calibration may help compressed predictions, but it will not solve all high-count misses.

---

## 10. Updated Interpretation of the Count Problem

The count problem is now better understood.

Earlier, the issue looked like:

```text
The count head underpredicts and misses cap+ cases.
```

After the new diagnostics, the more precise interpretation is:

```text
The count head learns a meaningful but compressed severity score.
It can rank some events by aftershock-count severity, but the raw predicted counts are severely under-scaled.
```

### 10.1 Evidence for useful severity signal

Prediction-bucket diagnostics show that when the model predicts a larger count, the actual mean true count is generally larger.

For example, test 72h:

| Predicted bucket | Mean pred | Mean true |
| ---------------- | --------: | --------: |
| 0–1              |    0.4952 |    5.0005 |
| 1–2              |    1.3744 |   13.4038 |
| 2–5              |    3.1065 |   24.9845 |
| 5–10             |    6.9470 |   45.6727 |
| 10–20            |   14.0036 |   70.9406 |
| 20–50            |   31.5101 |  110.3333 |

This suggests the model has learned a useful ordering.

### 10.2 Evidence for scale compression

True-count bucket diagnostics show that the model does not expand its predictions enough for moderate and high counts.

For example, test 24h:

| True bucket | Mean true | Mean pred |
| ----------- | --------: | --------: |
| 21–50       |   33.6089 |    3.3272 |
| 51–100      |   73.0000 |    4.7768 |
| 101–300     |  131.5000 |    8.0214 |
| 300+        |  370.6667 |   12.6031 |

This shows strong under-scaling.

### 10.3 Why this likely happens

Several factors likely contribute:

* The target is zero-heavy.
* The target is heavy-tailed.
* `log1p(count)` compresses large counts.
* `SmoothL1Loss` encourages conservative predictions.
* High-count events are rare.
* Trigger-level tabular features may not contain enough information to distinguish all high-count sequences.
* The current count head is only a linear layer after the shared trunk.

This does not mean the count head is useless. It means the current raw count output should be interpreted carefully.

---


## 11. Basic Log-Linear Count Calibration Experiment

After the count diagnostics showed that the count head had useful severity-ranking signal but severely compressed raw-count predictions, a first post-processing calibration experiment was run.

The goal was not to retrain the neural network or change the model architecture. Instead, the aim was to test whether the existing count predictions could be rescaled after prediction using validation data.

### 11.1 Calibration setup

The basic calibration model was fitted using the validation split only, separately for 24h and 72h:

```text
log1p(true_count) = intercept + slope * log1p(predicted_count)
```

Then calibrated predictions were converted back to raw-count scale using:

```text
calibrated_count = expm1(intercept + slope * log1p(predicted_count))
```

This was intentionally kept as post-processing:

* the neural model was not retrained;
* the original count prediction CSV was not overwritten;
* probability and magnitude predictions were unchanged;
* calibration was fitted on validation only, not on test;
* calibrated outputs were saved as separate files for comparison.

The calibration was applied to the current best shared-MLP run:

```text
mlp_multitask_v1_enriched_h256_128_64_do03
```

### 11.2 Calibration parameters

The fitted log-linear calibration parameters were:

| Horizon | Intercept |  Slope | Val mean true | Val mean pred before | Val mean pred after |
| ------: | --------: | -----: | ------------: | -------------------: | ------------------: |
|     24h |    0.2336 | 1.2689 |       13.4593 |               1.3934 |              3.3152 |
|     72h |    0.2907 | 1.1854 |       20.5350 |               2.0807 |              4.6720 |

Interpretation:

* The positive intercepts lifted predictions upward.
* Slopes greater than `1.0` stretched larger predictions more than smaller predictions.
* This matched the previous diagnosis that the count outputs were too low and too compressed.
* However, even after calibration, validation mean predictions remained far below validation mean true counts.

So calibration helped the scale problem, but did not fully solve it.

### 11.3 Uncapped count metrics before and after calibration

Test-set uncapped count metrics changed as follows:

| Horizon | Metric    | Original | Calibrated | Interpretation                     |
| ------: | --------- | -------: | ---------: | ---------------------------------- |
|     24h | MAE       |   9.0700 |     9.1784 | Worse                              |
|     24h | RMSE      |  28.4269 |    27.5740 | Better                             |
|     24h | Mean pred |   1.4777 |     3.6306 | Closer to mean true, but still low |
|     72h | MAE       |  13.5435 |    13.7954 | Worse                              |
|     72h | RMSE      |  42.2253 |    41.9279 | Slightly better                    |
|     72h | Mean pred |   2.2448 |     5.1910 | Closer to mean true, but still low |

Interpretation:

* Calibration improved RMSE for both horizons.
* Calibration worsened MAE for both horizons.
* Mean predicted counts moved closer to mean true counts, but remained substantially underpredicted.
* The RMSE improvement suggests calibration helped some larger count errors.
* The MAE worsening suggests calibration hurt many common zero-count or low-count rows by lifting them too much.

This is a mixed result rather than a clean improvement.

### 11.4 Capped count metrics before and after calibration

Test-set capped metrics showed a similar tradeoff:

| Horizon | Metric      | Original | Calibrated | Interpretation |
| ------: | ----------- | -------: | ---------: | -------------- |
|     24h | Capped MAE  |   8.9493 |     9.0139 | Worse          |
|     24h | Capped RMSE |  27.0131 |    25.7815 | Better         |
|     72h | Capped MAE  |  13.4316 |    13.6006 | Worse          |
|     72h | Capped RMSE |  40.8251 |    39.5697 | Better         |

Calibration improved capped RMSE more clearly than uncapped RMSE, which suggests it helped reduce some larger errors in the moderate-to-high count range.

However, cap+ detection did not meaningfully improve:

| Horizon | True cap+ | Predicted cap+ | Correct cap+ | Missed cap+ | False cap+ |
| ------: | --------: | -------------: | -----------: | ----------: | ---------: |
|     24h |         6 |              1 |            0 |           6 |          1 |
|     72h |         5 |              1 |            0 |           5 |          1 |

Interpretation:

* The original model predicted no test cap+ cases.
* Calibration produced one predicted cap+ case for each horizon.
* Both were false alarms.
* All true cap+ cases were still missed.

So calibration did not solve extreme-count detection.

### 11.5 Positive-count-only metrics before and after calibration

Calibration was more helpful when evaluation was restricted to rows where aftershocks actually occurred.

Test positive-only metrics:

| Horizon | Metric    | Original | Calibrated | Interpretation        |
| ------: | --------- | -------: | ---------: | --------------------- |
|     24h | MAE       |  18.6814 |    18.0563 | Better                |
|     24h | RMSE      |  41.5171 |    40.2238 | Better                |
|     24h | Mean pred |   2.4774 |     6.2158 | Better, but still low |
|     72h | MAE       |  24.8174 |    24.4418 | Better                |
|     72h | RMSE      |  57.9711 |    57.5213 | Better                |
|     72h | Mean pred |   3.5119 |     8.2168 | Better, but still low |

Interpretation:

* Calibration improved positive-only MAE and RMSE for both horizons.
* This means calibration helped cases where aftershocks actually occurred.
* However, calibrated positive-only mean predictions remained far below positive-only mean true counts.

For example:

```text
24h test positive cases:
mean true = 20.5462
calibrated mean pred = 6.2158

72h test positive cases:
mean true = 27.1981
calibrated mean pred = 8.2168
```

This explains the main calibration tradeoff:

```text
Calibration helps positive-count cases, but hurts some zero / low-count cases.
```

### 11.6 True-count bucket behavior after calibration

Calibration raised predictions in moderate and high true-count buckets, but not enough to fully solve underprediction.

#### Test 24h true-count buckets

| True-count bucket | Mean true | Original mean pred | Calibrated mean pred |
| ----------------- | --------: | -----------------: | -------------------: |
| 0                 |    0.0000 |             0.5963 |               1.3500 |
| 1                 |    1.0000 |             1.2095 |               2.8200 |
| 2–5               |    3.0220 |             1.5286 |               3.4200 |
| 6–10              |    7.7683 |             2.3782 |               5.4600 |
| 11–20             |   14.8452 |             3.0983 |               7.6300 |
| 21–50             |   33.6089 |             3.3272 |               8.2100 |
| 51–100            |   73.0000 |             4.7768 |              12.6000 |
| 101–300           |  131.5000 |             8.0214 |              24.6800 |
| 300+              |  370.6667 |            12.6031 |              35.9600 |

#### Test 72h true-count buckets

| True-count bucket | Mean true | Original mean pred | Calibrated mean pred |
| ----------------- | --------: | -----------------: | -------------------: |
| 0                 |    0.0000 |             0.8142 |               1.7700 |
| 1                 |    1.0000 |             1.4652 |               3.2000 |
| 2–5               |    2.9701 |             2.0342 |               4.4000 |
| 6–10              |    7.6593 |             2.7396 |               5.8000 |
| 11–20             |   14.9079 |             4.4211 |               9.9800 |
| 21–50             |   34.2115 |             4.8058 |              11.1500 |
| 51–100            |   72.8198 |             5.6040 |              13.0800 |
| 101–300           |  143.8028 |            10.6084 |              28.0600 |
| 300+              |  496.5000 |            15.6227 |              38.9800 |

Interpretation:

* Calibration raised predictions more for larger predicted counts, as intended.
* Moderate and high true-count buckets became less underpredicted.
* However, zero and one-count buckets also became overpredicted.
* Even after calibration, high-count buckets remained severely underpredicted.

For example, in the 72h `300+` bucket:

```text
mean true = 496.5000
calibrated mean pred = 38.9800
```

This is an improvement over the original prediction of `15.6227`, but it is still far below the true count scale.

### 11.7 Prediction-bucket behavior after calibration

After calibration, prediction-bucket diagnostics still showed a useful severity-ranking pattern.

#### Test 24h calibrated prediction buckets

| Predicted-count bucket | Rows | True positive rate | Mean true | Mean pred |
| ---------------------- | ---: | -----------------: | --------: | --------: |
| 0–1                    | 1340 |             0.2313 |    2.8900 |    0.6500 |
| 1–2                    | 1084 |             0.4820 |    5.5000 |    1.4000 |
| 2–5                    |  626 |             0.6440 |   11.4600 |    3.1100 |
| 5–10                   |  226 |             0.8140 |   23.9700 |    6.9200 |
| 10–20                  |  129 |             0.9220 |   35.3700 |   14.1100 |
| 20–50                  |   77 |             0.9870 |   55.5700 |   28.9000 |
| 50+                    |   31 |             1.0000 |   82.3900 |   90.7100 |

#### Test 72h calibrated prediction buckets

| Predicted-count bucket | Rows | True positive rate | Mean true | Mean pred |
| ---------------------- | ---: | -----------------: | --------: | --------: |
| 0–1                    |  839 |             0.2570 |    1.5000 |    0.7300 |
| 1–2                    | 1198 |             0.4490 |    7.4100 |    1.4500 |
| 2–5                    |  845 |             0.6620 |   14.3300 |    2.9900 |
| 5–10                   |  316 |             0.8040 |   27.0200 |    6.9000 |
| 10–20                  |  151 |             0.8810 |   46.5900 |   14.0000 |
| 20–50                  |  119 |             0.9920 |   68.2400 |   31.3000 |
| 50+                    |   45 |             1.0000 |  105.1600 |  118.8100 |

Interpretation:

* Higher calibrated prediction buckets still corresponded to higher true counts.
* True positive rate increased strongly across prediction buckets.
* Calibration created more spread in the predictions.
* The highest calibrated prediction buckets were not random; they generally corresponded to genuinely more severe cases.

This confirms that the count head has useful ranking signal, but the simple calibration mapping is still too blunt.

### 11.8 Overall conclusion from basic calibration

The basic log-linear calibration experiment was useful but mixed.

It improved:

* test RMSE;
* capped test RMSE;
* positive-count-only MAE and RMSE;
* mean predicted count;
* moderate/high bucket predictions;
* spread across prediction buckets.

It worsened:

* test MAE;
* capped test MAE;
* zero-count bucket predictions;
* one-count bucket predictions;
* false cap+ behavior.

Most important interpretation:

```text
The count head has useful severity-ranking signal, and calibration can partially stretch the compressed count scale. However, a simple global log-linear calibration also lifts too many zero and low-count rows, so it is too blunt to replace the original count predictions outright.
```

Therefore, this calibrated model should be treated as a diagnostic or optional post-processing experiment, not yet as the final count model.

The next natural calibration experiment is probability-aware count calibration:

```text
log1p(true_count)
=
a + b * log1p(predicted_count) + c * probability_prediction
```

Reasoning:

* The basic calibrator only knows the original count prediction.
* It improved positive-count cases but over-lifted many zero / low-count cases.
* The probability head may help the calibrator decide which rows are likely to be true zero cases and which rows deserve count inflation.
* This keeps the neural model unchanged and tests the idea as post-processing before any architecture redesign.

## 12. Probability-Aware Count Calibration Experiment

After the basic log-linear calibration experiment, the next question was whether calibration could use the occurrence-probability prediction to avoid lifting too many likely-zero or low-count rows.

The basic calibrator used only the original count prediction:

```text
log1p(true_count) = a + b * log1p(predicted_count)
```

This helped stretch the compressed count scale, but it also raised many zero and one-count cases too much. Therefore, the next calibration experiment added the probability prediction for the same horizon as an additional post-processing input:

```text
log1p(true_count)
=
a
+ b * log1p(original_count_prediction)
+ c * probability_prediction
```

The calibrated prediction is then:

```text
calibrated_count
= expm1(a + b * log1p(original_count_prediction) + c * probability_prediction)
```

This remains a post-processing calibration model. The neural network itself was not retrained, and the probability head was not fed into the count head during neural training. The original architecture remained:

```text
features -> shared trunk -> probability head
features -> shared trunk -> count head
features -> shared trunk -> magnitude head
```

The calibration script was updated to support multiple calibration methods:

```bash
python3 scripts/calibrate_count_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --method loglinear_probaware
```

The probability-aware method reads both saved prediction files:

```text
<run_name>_count_predictions.csv
<run_name>_probability_predictions.csv
```

and joins probability predictions onto count predictions using:

```text
trigger_event_id
split
horizon
```

This keeps the calibration horizon-specific and row-specific while avoiding any change to the training pipeline. The calibration is still fitted on the validation split only, separately for 24h and 72h, and then applied to train, validation, and test. The test split is not used to fit calibration parameters.

### 12.1 Why probability-aware calibration was tried

The motivation was based directly on the mixed result from basic calibration. Basic calibration showed that the count head has useful severity-ranking signal, because stretching the count predictions improved RMSE and positive-only metrics. However, it also worsened MAE because it lifted too many common zero and low-count rows.

The probability head is trained to answer whether at least one aftershock occurs. Therefore, it may help the calibrator distinguish between two cases that have similar count predictions but different occurrence likelihoods. For example:

```text
count_pred = 1.5, probability = 0.25
count_pred = 1.5, probability = 0.80
```

A count-only calibrator treats these similarly, but a probability-aware calibrator can lift the second case more than the first. This makes it a natural next step before changing the neural architecture.

Important design choice:

```text
This is not a fully conditional neural model.
```

A fully conditional neural model would make the count prediction depend on occurrence probability during neural training, or would train the positive-count severity model separately from occurrence. This experiment does not do that. It only uses already-saved probability and count predictions in a validation-fitted post-processing model.

### 12.2 Calibration parameters

The fitted probability-aware calibration parameters were:

| Horizon | Intercept | Count log slope | Probability slope | Val mean true | Val mean pred before | Val mean pred after | Val mean probability |
| ------: | --------: | --------------: | ----------------: | ------------: | -------------------: | ------------------: | -------------------: |
|     24h |    0.1876 |          1.1923 |            0.2216 |       13.4593 |               1.3934 |              3.2609 |               0.4291 |
|     72h |    0.0424 |          0.9260 |            0.9175 |       20.5350 |               2.0807 |              4.3566 |               0.4962 |

Interpretation:

* For 24h, the count prediction still carries most of the calibration signal. The probability coefficient is positive but modest.
* For 72h, the probability coefficient is much larger, meaning the calibrator relies more strongly on the occurrence-probability prediction to decide how much to lift the count.
* Compared with basic calibration, the probability-aware model uses a less aggressive count slope, especially for 72h. This is consistent with the goal of avoiding a blunt global upward rescaling.

The validation mean prediction still remains far below the validation mean truth after calibration, so this method only partially addresses the scale problem.

### 12.3 Uncapped count metrics

Test-set uncapped count metrics compared with the original and basic calibrated versions:

| Horizon | Version           |     MAE |    RMSE | Mean true | Mean pred |
| ------: | ----------------- | ------: | ------: | --------: | --------: |
|     24h | Original          |  9.0700 | 28.4269 |    9.6268 |    1.4777 |
|     24h | Basic log-linear  |  9.1784 | 27.5740 |    9.6268 |    3.6306 |
|     24h | Probability-aware |  9.1399 | 27.3409 |    9.6268 |    3.5381 |
|     72h | Original          | 13.5435 | 42.2253 |   14.4236 |    2.2448 |
|     72h | Basic log-linear  | 13.7954 | 41.9279 |   14.4236 |    5.1910 |
|     72h | Probability-aware | 13.5958 | 40.7277 |   14.4236 |    4.6615 |

Interpretation:

* Probability-aware calibration improved RMSE compared with both the original model and the basic calibration for both horizons.
* For 24h, MAE remained worse than the original model, but better than the basic calibrated model.
* For 72h, MAE was only slightly worse than the original model and clearly better than the basic calibrated model.
* Mean predicted counts moved closer to the true means than the original model, but still remained far below the true means.

This is a better tradeoff than the basic calibrator. The basic calibrator improved RMSE but worsened MAE more noticeably. The probability-aware calibrator preserved most of the RMSE gain while reducing the MAE penalty.

### 12.4 Capped count metrics

Test-set capped count metrics were:

| Horizon | Version           | Capped MAE | Capped RMSE | True cap+ | Pred cap+ | Correct cap+ | Missed cap+ | False cap+ |
| ------: | ----------------- | ---------: | ----------: | --------: | --------: | -----------: | ----------: | ---------: |
|     24h | Original          |     8.9493 |     27.0131 |         6 |         0 |            0 |           6 |          0 |
|     24h | Basic log-linear  |     9.0139 |     25.7815 |         6 |         1 |            0 |           6 |          1 |
|     24h | Probability-aware |     8.9968 |     25.7777 |         6 |         1 |            0 |           6 |          1 |
|     72h | Original          |    13.4316 |     40.8251 |         5 |         0 |            0 |           5 |          0 |
|     72h | Basic log-linear  |    13.6006 |     39.5697 |         5 |         1 |            0 |           5 |          1 |
|     72h | Probability-aware |    13.4839 |     39.3396 |         5 |         0 |            0 |           5 |          0 |

Interpretation:

* Probability-aware calibration produced the best capped RMSE for both horizons.
* 24h cap+ behavior did not improve; it still produced one false cap+ case and missed all true cap+ cases.
* 72h cap+ behavior improved relative to the basic calibrator because the false cap+ case disappeared.
* Like the earlier models, probability-aware calibration still missed all true cap+ cases.

This means probability-aware calibration helps reduce squared error in the moderate/high count range, but it does not solve extreme-count detection.

### 12.5 Positive-count-only metrics

Positive-count-only test metrics were:

| Horizon | Version           | Positive-only MAE | Positive-only RMSE | Mean true | Mean pred |
| ------: | ----------------- | ----------------: | -----------------: | --------: | --------: |
|     24h | Original          |           18.6814 |            41.5171 |   20.5462 |    2.4774 |
|     24h | Basic log-linear  |           18.0563 |            40.2238 |   20.5462 |    6.2158 |
|     24h | Probability-aware |           17.9706 |            39.8821 |   20.5462 |    6.0149 |
|     72h | Original          |           24.8174 |            57.9711 |   27.1981 |    3.5119 |
|     72h | Basic log-linear  |           24.4418 |            57.5213 |   27.1981 |    8.2168 |
|     72h | Probability-aware |           24.0467 |            55.8687 |   27.1981 |    7.1996 |

Interpretation:

* Probability-aware calibration gave the best positive-only MAE and RMSE for both horizons.
* This is the strongest evidence that probability-aware calibration is useful.
* It improved cases where aftershocks actually occurred, while being less aggressive than the basic calibrator.
* However, positive-only mean predictions were still far below positive-only mean true counts.

For example:

```text
24h positive-only test:
mean true = 20.5462
probability-aware mean pred = 6.0149

72h positive-only test:
mean true = 27.1981
probability-aware mean pred = 7.1996
```

So the calibration improves positive cases, but the positive-count scale remains strongly compressed.

### 12.6 True-count bucket behavior

Probability-aware calibration was expected to reduce the basic calibrator's over-lifting of zero and low-count rows. The bucket diagnostics showed a more mixed result.

#### Test 24h true-count buckets

| True-count bucket | Mean true | Basic mean pred | Probability-aware mean pred |
| ----------------- | --------: | --------------: | --------------------------: |
| 0                 |    0.0000 |          1.3514 |                      1.3544 |
| 1                 |    1.0000 |          2.8225 |                      2.7937 |
| 2–5               |    3.0220 |          3.4188 |                      3.4209 |
| 6–10              |    7.7683 |          5.4593 |                      5.4208 |
| 11–20             |   14.8452 |          7.6343 |                      7.4260 |
| 21–50             |   33.6089 |          8.2107 |                      7.9752 |
| 51–100            |   73.0000 |         12.6003 |                     12.0311 |
| 101–300           |  131.5000 |         24.6808 |                     22.6006 |
| 300+              |  370.6667 |         35.9610 |                     33.8305 |

#### Test 72h true-count buckets

| True-count bucket | Mean true | Basic mean pred | Probability-aware mean pred |
| ----------------- | --------: | --------------: | --------------------------: |
| 0                 |    0.0000 |          1.7746 |                      1.7958 |
| 1                 |    1.0000 |          3.2018 |                      3.1286 |
| 2–5               |    2.9701 |          4.3995 |                      4.3066 |
| 6–10              |    7.6593 |          5.7981 |                      5.7784 |
| 11–20             |   14.9079 |          9.9825 |                      9.1241 |
| 21–50             |   34.2115 |         11.1507 |                      9.7828 |
| 51–100            |   72.8198 |         13.0806 |                     11.3759 |
| 101–300           |  143.8028 |         28.0561 |                     20.8446 |
| 300+              |  496.5000 |         38.9821 |                     31.3530 |

Interpretation:

* Probability-aware calibration did not meaningfully reduce zero-bucket overprediction.
* The 24h zero-count bucket was almost unchanged compared with basic calibration.
* The 72h zero-count bucket was slightly higher than basic calibration.
* For most moderate and high true-count buckets, probability-aware calibration was less aggressive than basic calibration.
* This helped overall MAE/RMSE tradeoffs, but high-count buckets remained severely underpredicted.

Therefore, the improvement is not because the probability-aware model solved the zero-count bucket problem. Instead, it improved the overall calibration tradeoff by reshaping the predictions more cautiously, especially for 72h.

### 12.7 Prediction-bucket behavior after probability-aware calibration

The prediction-bucket diagnostics remained encouraging.

#### Test 24h probability-aware prediction buckets

| Predicted-count bucket | Rows | True positive rate | Mean true | Mean pred |
| ---------------------- | ---: | -----------------: | --------: | --------: |
| 0–1                    | 1323 |             0.2290 |    2.9101 |    0.6255 |
| 1–2                    | 1081 |             0.4810 |    5.4875 |    1.4053 |
| 2–5                    |  643 |             0.6392 |   11.1571 |    3.1204 |
| 5–10                   |  229 |             0.8166 |   24.1834 |    6.9406 |
| 10–20                  |  133 |             0.9173 |   35.7820 |   14.1277 |
| 20–50                  |   75 |             0.9867 |   54.2000 |   28.8221 |
| 50+                    |   29 |             1.0000 |   86.2414 |   84.3465 |

#### Test 72h probability-aware prediction buckets

| Predicted-count bucket | Rows | True positive rate | Mean true | Mean pred |
| ---------------------- | ---: | -----------------: | --------: | --------: |
| 0–1                    |  900 |             0.2689 |    2.1711 |    0.6265 |
| 1–2                    | 1017 |             0.4385 |    7.3609 |    1.4697 |
| 2–5                    |  933 |             0.6517 |   13.6334 |    3.0016 |
| 5–10                   |  337 |             0.7804 |   24.8368 |    6.9291 |
| 10–20                  |  170 |             0.8765 |   44.9353 |   13.8686 |
| 20–50                  |  118 |             0.9915 |   69.7797 |   29.6012 |
| 50+                    |   38 |             1.0000 |  112.2895 |   87.6613 |

Interpretation:

* Higher calibrated prediction buckets still corresponded to higher true positive rates.
* Higher calibrated prediction buckets also corresponded to higher mean true counts.
* This means probability-aware calibration preserved the useful severity-ranking behavior of the count head.
* The highest prediction buckets were not random false inflation; they generally represented genuinely more severe trigger events.

This supports the earlier conclusion that the count head has useful ranking signal, even though its raw scale is compressed.

### 12.8 Overall conclusion from probability-aware calibration

Probability-aware count calibration was a useful improvement over the basic log-linear calibrator.

It improved:

* test RMSE for both horizons compared with the original model;
* test RMSE and MAE for both horizons compared with basic calibration;
* capped RMSE for both horizons;
* positive-count-only MAE and RMSE for both horizons;
* 72h false cap+ behavior compared with basic calibration;
* prediction-bucket severity ranking.

It did not solve:

* zero-count bucket overprediction;
* high-count underprediction;
* cap+ detection;
* the large gap between mean true count and mean predicted count.

The most accurate interpretation is:

```text
Probability-aware calibration improves the calibration tradeoff compared with basic count-only calibration, especially for RMSE and positive-count cases. However, it does not fully resolve zero/low-count overprediction or high-count underprediction. It is a better post-processing option, but not a complete fix for the count task.
```

This result suggests that the next count-focused experiment should move beyond post-processing calibration and test a targeted architecture change, such as a task-specific count tower.



## 13. Count-Tower Architecture Experiment

After the probability-aware calibration experiment, the next question was whether the count task needed more task-specific nonlinear capacity inside the neural model itself.

The motivation was:

```text
The count head has useful ranking/severity signal, but its raw predictions are compressed and under-scaled.
Post-processing calibration helps somewhat, but it does not fully solve the problem.
```

The previous count head was effectively a simple linear mapping from the shared trunk representation to the two count outputs. A targeted architecture change was therefore tested to see whether the count task would benefit from its own extra nonlinear transformation, without changing the probability head, the magnitude head, the losses, the targets, or the output dictionary structure.

### 13.1 Architecture change

The model code was extended so that the count head could optionally be replaced by a small count-specific tower while preserving the existing default behavior.

The original best shared-trunk model remained:

```text
shared trunk: 256 -> 128 -> 64
dropout: 0.3
```

The count-tower experiment used:

```text
probability head: 64 -> 2
count tower:      64 -> 64 -> 2
magnitude head:   64 -> 2
```

Important design constraints were preserved:

* probability head unchanged;
* magnitude head unchanged;
* output keys unchanged:
  ```python
  {
      "prob_logits": ...,
      "count_pred": ...,
      "magnitude_pred": ...,
  }
  ```
* losses unchanged;
* target transforms unchanged;
* prediction CSV schemas unchanged;
* evaluation and calibration scripts unchanged.

This made the experiment a controlled test of one question:

```text
Does the count task improve if it gets a small task-specific nonlinear tower after the shared trunk?
```

The count-tower run used:

```bash
python3 scripts/train_multitask_mlp.py   --dataset-name earthquake_aftershock_v2_gcmt   --feature-set nn_enriched_v1   --run-name mlp_multitask_v1_enriched_h256_128_64_do03_counttower64   --learning-rate 3e-4   --count-loss-weight 1.0   --hidden-dims 256 128 64   --count-head-hidden-dims 64   --dropout 0.3
```

### 13.2 Raw count results from the count tower

Test-set raw count metrics for the count-tower model were:

| Horizon | Test MAE | Test RMSE | Mean true | Mean pred |
| ------: | -------: | --------: | --------: | --------: |
|     24h |   9.1115 |   28.5860 |    9.6268 |    1.6180 |
|     72h |  13.6809 |   43.1871 |   14.4236 |    2.5641 |

Compared with the previous best shared-MLP run, `mlp_multitask_v1_enriched_h256_128_64_do03`:

| Horizon | Version           | Test MAE | Test RMSE | Mean true | Mean pred |
| ------: | ----------------- | -------: | --------: | --------: | --------: |
|     24h | Previous best raw |   9.0700 |   28.4269 |    9.6268 |    1.4777 |
|     24h | Count tower raw   |   9.1115 |   28.5860 |    9.6268 |    1.6180 |
|     72h | Previous best raw |  13.5435 |   42.2253 |   14.4236 |    2.2448 |
|     72h | Count tower raw   |  13.6809 |   43.1871 |   14.4236 |    2.5641 |

Interpretation:

* The count tower made raw mean predictions slightly larger, so the model became somewhat less conservative.
* However, raw count MAE and RMSE both worsened at both horizons.
* This means the count tower did not improve raw count prediction quality in the way hoped.
* The result suggests that simply adding one small count-specific hidden layer is not enough to solve the scale-compression problem.

### 13.3 Raw positive-count-only metrics

Because the count problem is most severe when aftershocks actually occur, positive-count-only metrics were especially important.

Test positive-only metrics for the count-tower model were:

| Horizon | Positive-only MAE | Positive-only RMSE | Mean true | Mean pred |
| ------: | ----------------: | -----------------: | --------: | --------: |
|     24h |           18.7849 |            41.7470 |   20.5462 |    2.7919 |
|     72h |           25.0884 |            59.2889 |   27.1981 |    4.1258 |

Compared with the previous best raw shared-MLP model:

| Horizon | Version           | Positive-only MAE | Positive-only RMSE | Mean true | Mean pred |
| ------: | ----------------- | ----------------: | -----------------: | --------: | --------: |
|     24h | Previous best raw |           18.6814 |            41.5171 |   20.5462 |    2.4774 |
|     24h | Count tower raw   |           18.7849 |            41.7470 |   20.5462 |    2.7919 |
|     72h | Previous best raw |           24.8174 |            57.9711 |   27.1981 |    3.5119 |
|     72h | Count tower raw   |           25.0884 |            59.2889 |   27.1981 |    4.1258 |

Interpretation:

* Positive-case mean predictions increased slightly.
* However, positive-only MAE and RMSE both worsened.
* This means the count tower did not improve the model's ability to estimate aftershock-count severity on rows where aftershocks actually occurred.

So the count-tower change did not produce the hoped-for positive-count improvement.

### 13.4 Raw capped count results

Test capped count metrics for the count-tower model were:

| Horizon | Capped MAE | Capped RMSE | True cap+ | Pred cap+ | Correct cap+ | Missed cap+ | False cap+ |
| ------: | ---------: | ----------: | --------: | --------: | -----------: | ----------: | ---------: |
|     24h |     8.9893 |     27.1612 |         6 |         1 |            0 |           6 |          1 |
|     72h |    13.5174 |     41.2437 |         5 |         1 |            0 |           5 |          1 |

Interpretation:

* The count tower still missed all true cap+ cases on test.
* It also introduced one false cap+ case at each horizon.
* Capped RMSE worsened relative to the previous best raw model.

So the tower did not improve extreme-count behavior either.

### 13.5 True-count bucket behavior with the raw count tower

The true-count bucket diagnostics were useful because they showed where the count tower changed model behavior.

#### Test 24h true-count buckets

| True-count bucket | Mean true | Previous best raw mean pred | Count-tower raw mean pred |
| ----------------- | --------: | --------------------------: | ------------------------: |
| 0                 |    0.0000 |                      0.5963 |                    0.5831 |
| 1                 |    1.0000 |                      1.2095 |                    1.2787 |
| 2–5               |    3.0220 |                      1.5286 |                    1.7779 |
| 6–10              |    7.7683 |                      2.3782 |                    2.5711 |
| 11–20             |   14.8452 |                      3.0983 |                    3.0367 |
| 21–50             |   33.6089 |                      3.3272 |                    3.6580 |
| 51–100            |   73.0000 |                      4.7768 |                    5.4435 |
| 101–300           |  131.5000 |                      8.0214 |                   11.6728 |
| 300+              |  370.6667 |                     12.6031 |                   10.6354 |

#### Test 72h true-count buckets

| True-count bucket | Mean true | Previous best raw mean pred | Count-tower raw mean pred |
| ----------------- | --------: | --------------------------: | ------------------------: |
| 0                 |    0.0000 |                      0.8142 |                    0.8008 |
| 1                 |    1.0000 |                      1.4652 |                    1.5072 |
| 2–5               |    2.9701 |                      2.0342 |                    2.2615 |
| 6–10              |    7.6593 |                      2.7396 |                    3.1501 |
| 11–20             |   14.9079 |                      4.4211 |                    4.6903 |
| 21–50             |   34.2115 |                      4.8058 |                    5.3079 |
| 51–100            |   72.8198 |                      5.6040 |                    6.3993 |
| 101–300           |  143.8028 |                     10.6084 |                   14.9832 |
| 300+              |  496.5000 |                     15.6227 |                   13.2572 |

Interpretation:

* The count tower did increase predictions in several moderate and upper-middle true-count buckets.
* However, this did not translate into better overall count metrics.
* The highest `300+` bucket remained very poorly predicted and even decreased relative to the previous best raw model.
* The tower changed the scale somewhat, but not in a stable or useful enough way to improve test performance.

### 13.6 Prediction-bucket behavior with the raw count tower

The prediction-bucket diagnostics showed that the count tower still preserved the useful ranking structure of the count task.

#### Test 24h count-tower prediction buckets

| Predicted-count bucket | Rows | True positive rate | Mean true | Mean pred |
| ---------------------- | ---: | -----------------: | --------: | --------: |
| 0–1                    | 2386 |             0.3571 |    4.3168 |    0.4261 |
| 1–2                    |  539 |             0.6228 |   10.9569 |    1.4077 |
| 2–5                    |  356 |             0.7584 |   19.4560 |    3.0652 |
| 5–10                   |  130 |             0.9154 |   36.7692 |    6.8207 |
| 10–20                  |   78 |             0.9872 |   68.7308 |   13.7197 |
| 20–50                  |   45 |             1.0000 |   74.4889 |   29.1387 |
| 50+                    |    1 |             1.0000 |   46.0000 |   53.8967 |

#### Test 72h count-tower prediction buckets

| Predicted-count bucket | Rows | True positive rate | Mean true | Mean pred |
| ---------------------- | ---: | -----------------: | --------: | --------: |
| 0–1                    | 1917 |             0.3928 |    5.3490 |    0.4659 |
| 1–2                    |  741 |             0.6561 |   13.2416 |    1.4126 |
| 2–5                    |  485 |             0.7546 |   23.4433 |    3.0943 |
| 5–10                   |  161 |             0.8509 |   34.8944 |    6.9698 |
| 10–20                  |  111 |             0.9550 |   67.7117 |   13.9323 |
| 20–50                  |   54 |             1.0000 |   89.2963 |   30.0170 |
| 50+                    |   18 |             1.0000 |  100.5556 |   69.6965 |

Interpretation:

* Higher predicted-count buckets still corresponded to higher true positive rates and higher mean true counts.
* So the count tower did not destroy the useful severity-ranking behavior.
* However, preserving ranking signal alone was not enough. Raw count accuracy still worsened.

### 13.7 Probability-aware calibration applied to the count-tower model

The same validation-fitted probability-aware calibrator was then applied to the count-tower model:

```text
log1p(true_count)
=
a
+ b * log1p(original_count_prediction)
+ c * probability_prediction
```

Calibration parameters for the count-tower model were:

| Horizon | Intercept | Count log slope | Probability slope | Val mean true | Val mean pred before | Val mean pred after |
| ------: | --------: | --------------: | ----------------: | ------------: | -------------------: | ------------------: |
|     24h |    0.0803 |          1.1949 |            0.4153 |       13.4593 |               1.3451 |              3.1053 |
|     72h |   -0.0428 |          0.8401 |            0.9235 |       20.5350 |               2.1007 |              4.2001 |

Interpretation:

* For 24h, the calibration relied more heavily on probability than the earlier non-tower probability-aware calibrator.
* For 72h, the count slope became even smaller, while the probability slope remained large.
* This suggests the count tower did not make the raw count prediction more trustworthy; the calibrator still needed strong probability-based correction.

### 13.8 Count-tower calibrated results

Test-set uncapped metrics for the calibrated count-tower model were:

| Horizon | Version                       | Test MAE | Test RMSE | Mean true | Mean pred |
| ------: | ----------------------------- | -------: | --------: | --------: | --------: |
|     24h | Raw count tower               |   9.1115 |   28.5860 |    9.6268 |    1.6180 |
|     24h | Count tower + prob-aware cal. |   9.1855 |   27.7525 |    9.6268 |    3.1594 |
|     72h | Raw count tower               |  13.6809 |   43.1871 |   14.4236 |    2.5641 |
|     72h | Count tower + prob-aware cal. |  13.7247 |   41.3893 |   14.4236 |    4.2277 |

Positive-only test metrics for the calibrated count-tower model were:

| Horizon | Version                       | Positive-only MAE | Positive-only RMSE | Mean true | Mean pred |
| ------: | ----------------------------- | ----------------: | -----------------: | --------: | --------: |
|     24h | Raw count tower               |           18.7849 |            41.7470 |   20.5462 |    2.7919 |
|     24h | Count tower + prob-aware cal. |           18.0721 |            40.4806 |   20.5462 |    5.6061 |
|     72h | Raw count tower               |           25.0884 |            59.2889 |   27.1981 |    4.1258 |
|     72h | Count tower + prob-aware cal. |           24.2841 |            56.7743 |   27.1981 |    6.3776 |

Interpretation:

* Calibration still helped the count-tower model relative to its own raw predictions.
* RMSE improved at both horizons.
* Positive-only MAE and RMSE improved at both horizons.
* Mean predictions moved closer to the true mean counts than the raw count-tower outputs.

So the calibrator still provided useful post-processing on top of the tower.

### 13.9 Comparison with the previous calibrated non-tower model

The crucial comparison, however, is not only whether calibration helped the tower model relative to its own raw version. It is whether the tower path became better than the earlier best path.

Test-set uncapped comparison:

| Horizon | Version                              | Test MAE | Test RMSE | Mean pred |
| ------: | ------------------------------------ | -------: | --------: | --------: |
|     24h | Previous prob-aware calibrated model |   9.1399 |   27.3409 |    3.5381 |
|     24h | Count tower + prob-aware calibrated  |   9.1855 |   27.7525 |    3.1594 |
|     72h | Previous prob-aware calibrated model |  13.5958 |   40.7277 |    4.6615 |
|     72h | Count tower + prob-aware calibrated  |  13.7247 |   41.3893 |    4.2277 |

Test capped comparison:

| Horizon | Version                              | Capped MAE | Capped RMSE |
| ------: | ------------------------------------ | ---------: | ----------: |
|     24h | Previous prob-aware calibrated model |     8.9968 |     25.7777 |
|     24h | Count tower + prob-aware calibrated  |     9.0239 |     25.8952 |
|     72h | Previous prob-aware calibrated model |    13.4839 |     39.3396 |
|     72h | Count tower + prob-aware calibrated  |    13.6172 |     39.9715 |

Interpretation:

* The calibrated count-tower model is worse than the earlier calibrated non-tower model at both horizons.
* This means the count tower did not improve the underlying count representation enough to produce a better final calibrated model.
* The count tower is therefore not a new best model.

### 13.10 Overall conclusion from the count-tower experiment

The count-tower experiment was an informative negative result.

What it showed:

* A simple count-specific tower can preserve the useful ranking/severity structure of the count task.
* Calibration can still improve the tower model relative to its own raw outputs.
* However, the simple `64 -> 64 -> 2` count tower did not improve the raw count model.
* It also did not produce a better calibrated model than the earlier probability-aware calibrated shared-MLP baseline.

Most accurate interpretation:

```text
A small count-specific tower is not enough, by itself, to solve the count problem.
The count bottleneck is probably not just that the count head is too shallow.
The issue likely also involves the target formulation, the log1p compression, the zero-heavy / heavy-tailed structure, and the difficulty of learning positive-count severity from the available tabular features.
```

Therefore:

```text
The count-tower run should be treated as an informative negative experiment, not as the new best model.
```

The previous best count path remains:

```text
mlp_multitask_v1_enriched_h256_128_64_do03
+
probability-aware post-processing calibration
```


## 14. Current Best Neural Model

Based on the larger-MLP experiments, diagnostics, calibration experiments, and the negative count-tower result, the current best shared-MLP configuration remains:

```text
run_name = mlp_multitask_v1_enriched_h256_128_64_do03
feature_set = nn_enriched_v1
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 1.0
gradient_clip_max_norm = 1.0
```

Recommended raw-model training command:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 128 64 \
  --dropout 0.3
```

Recommended evaluation-only command:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03
```

Recommended best post-processing count command:

```bash
python3 scripts/calibrate_count_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --method loglinear_probaware
```

So the best practical count result currently comes from the shared-MLP baseline plus probability-aware calibration, not from the count-tower architecture.

---

## 15. Results Summary by Task

### 15.1 Probability prediction

Probability prediction remains the strongest task.

Current best test results from `h256_128_64_do03`:

| Horizon | ROC-AUC | Brier score | Log loss | Positive rate |
| ------- | ------: | ----------: | -------: | ------------: |
| 24h     |  0.7664 |      0.1972 |   0.5746 |        0.4685 |
| 72h     |  0.7485 |      0.2042 |   0.5913 |        0.5303 |

Interpretation:

* The probability head learns meaningful signal.
* Performance is stable across horizons.
* The moderate larger MLP with dropout `0.3` remains the best architecture for this task among the models tried so far.

### 15.2 Count prediction

Count prediction remains the hardest task.

Best raw test count results remain those from `h256_128_64_do03`:

| Horizon |     MAE |    RMSE | Mean true | Mean pred |
| ------- | ------: | ------: | --------: | --------: |
| 24h     |  9.0700 | 28.4269 |    9.6268 |    1.4777 |
| 72h     | 13.5435 | 42.2253 |   14.4236 |    2.2448 |

Best calibrated count results currently come from the probability-aware calibrated version of the same shared-MLP model:

| Horizon | Version               |     MAE |    RMSE | Mean pred |
| ------- | --------------------- | ------: | ------: | --------: |
| 24h     | Prob-aware calibrated |  9.1399 | 27.3409 |    3.5381 |
| 72h     | Prob-aware calibrated | 13.5958 | 40.7277 |    4.6615 |

Interpretation:

* The raw count task remains strongly under-scaled.
* Probability-aware calibration is currently the best count post-processing method tried.
* The count-tower model did not beat either the raw or calibrated non-tower baseline.
* Count remains the main weakness of the neural pipeline.

### 15.3 Capped count prediction

Best capped calibrated results currently come from the probability-aware calibrated shared-MLP model:

| Horizon | Capped MAE | Capped RMSE | True cap+ | Pred cap+ | Missed cap+ |
| ------- | ---------: | ----------: | --------: | --------: | ----------: |
| 24h     |     8.9968 |     25.7777 |         6 |         1 |           6 |
| 72h     |    13.4839 |     39.3396 |         5 |         0 |           5 |

Interpretation:

* Calibration improves capped RMSE relative to the raw model.
* Cap+ detection remains poor.
* The count-tower path did not improve this limitation.

### 15.4 Maximum magnitude prediction

Current best test magnitude results still come from `h256_128_64_do03`:

| Horizon |    MAE |   RMSE | Mean true | Mean pred |
| ------- | -----: | -----: | --------: | --------: |
| 24h     | 0.4779 | 0.6264 |    5.0155 |    4.9826 |
| 72h     | 0.4824 | 0.6285 |    5.0412 |    5.0190 |

Interpretation:

* Magnitude prediction remains reasonably stable.
* The count-tower experiment did not target magnitude and does not change the best magnitude result.

---

## 16. Key Findings So Far

### 16.1 Neural model viability

The multi-task MLP is a useful neural baseline. It learns meaningful occurrence-probability signal and produces reasonable conditional magnitude predictions.

### 16.2 Task difficulty differs strongly

The tasks remain quite different in difficulty:

| Task              | Current status                |
| ----------------- | ----------------------------- |
| Probability       | Strongest and most stable     |
| Maximum magnitude | Promising and stable          |
| Count             | Hardest and most conservative |

### 16.3 GCMT enrichment has not clearly helped yet

The enriched feature set still has not produced a clear improvement over core features in the current MLP setup.

Possible reasons remain:

* incomplete GCMT coverage;
* sparse or imputed GCMT fields;
* optimization difficulty from extra features;
* target difficulty, especially for count.

### 16.4 Larger shared MLPs helped only modestly

The moderate `256 128 64` architecture with dropout `0.3` remains the best overall shared-trunk architecture tested so far.

### 16.5 Count head is under-scaled rather than completely uninformative

The count diagnostics continue to show that the count head has useful severity-ranking information but poor raw-count calibration.

### 16.6 Calibration helps, but only partially

Both the basic log-linear calibrator and the probability-aware calibrator improved some count metrics, especially RMSE and positive-only metrics. Probability-aware calibration is the better of the two. However, calibration still does not solve zero-count overprediction, high-count underprediction, or cap+ detection.

### 16.7 Simple count-tower architecture did not solve the problem

The count-tower experiment showed that a small task-specific count tower does not automatically improve count prediction.

This is an important negative result because it suggests the count bottleneck is not simply that the count head is too shallow. The problem likely also involves the target design and the difficulty of learning positive-count severity under strong zero inflation and heavy tails.

---

## 17. Current Limitations

Current limitations now include:

* no early stopping, although best checkpoint selection is based on validation loss;
* limited task-weight tuning;
* count is modeled as continuous regression rather than with a count-specific distribution;
* count targets remain zero-heavy and heavy-tailed;
* `log1p(count)` still compresses the upper tail;
* high-count events are rare and difficult to learn;
* capped evaluation is diagnostic only;
* probability outputs are not separately calibrated beyond the raw neural output;
* count calibration is helpful but still incomplete;
* GCMT enrichment has not clearly improved performance;
* the simple count-tower experiment did not improve the count task enough;
* evaluation-only diagnostics explain the problem better, but do not themselves fix it.

---

## 18. Recommended Next Experiments

### 18.1 Completed probability-aware count calibration

Probability-aware count calibration has been completed and is currently the best post-processing calibration method tried so far.

It improved the calibration tradeoff relative to both the raw count predictions and the basic count-only calibration. It especially improved RMSE-based and positive-count-only metrics. However, it still did not solve zero-count overprediction, high-count underprediction, or cap+ detection.

So it should be treated as:

```text
the best count calibration result so far,
but not a complete solution
```

### 18.2 Completed simple count-tower experiment

A simple count-tower architecture has also now been tested:

```text
shared trunk: 256 -> 128 -> 64
count tower: 64 -> 64 -> 2
```

This did not improve the raw count model and did not produce a better calibrated model than the earlier non-tower shared-MLP baseline.

Therefore, the count-tower result should be treated as:

```text
an informative negative experiment
```

and not as the new preferred architecture.

### 18.3 Recommended next direction: move beyond small post-hoc or small head-only tweaks

The current evidence suggests that the next count-focused experiment should not just be a slightly larger version of the same count tower.

More promising next directions are:

* conditional count modeling, separating occurrence from positive-count severity;
* alternative count-specific objectives or distributions;
* count-specific threshold models for moderate/extreme count events;
* probability calibration if future count models continue to rely on probability-aware post-processing;
* deeper analysis of whether `log1p(count)` is too compressive for the upper tail.

### 18.4 Other possible future directions

Later, if time allows:

* count-only model for comparison;
* Poisson or negative-binomial style count objectives;
* high-count classifier for moderate / extreme count thresholds;
* probability calibration;
* feature selection or stronger regularization for enriched GCMT features;
* learning-curve plots from history CSVs;
* comparison against traditional baselines under both uncapped and capped count metrics.

---

## 19. What Not To Do Yet

Based on the updated evidence, avoid the following for now:

### Do not keep making the shared MLP bigger blindly

The wider shared trunk did not solve the count problem.

### Do not keep scaling up the same small count tower immediately

Because the first count-tower attempt was not even directionally better, simply trying `128` or `128 64` as the next step is less compelling than moving to a more conceptually different count approach.

### Do not cap training targets yet

Underprediction begins well below the cap thresholds.

### Do not compare capped neural metrics against uncapped baselines

Capped evaluation must still be applied consistently.

### Do not tune repeatedly on the test set

Validation remains the correct place for experiment selection.

### Do not change dataset labels casually

Keep the aftershock definition and splits stable unless a future experiment is specifically about label design.

---

## 20. Current Recommended Workflow

### 20.1 Train current best shared-MLP model

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 128 64 \
  --dropout 0.3
```

### 20.2 Regenerate metrics and diagnostics

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03
```

Expected main outputs:

```text
<run_name>_probability_metrics.csv
<run_name>_count_metrics.csv
<run_name>_count_capped_metrics.csv
<run_name>_magnitude_metrics.csv
<run_name>_count_positive_only_metrics.csv
<run_name>_count_bucket_diagnostics.csv
<run_name>_count_prediction_bucket_diagnostics.csv
```

### 20.3 Apply best current count calibration

```bash
python3 scripts/calibrate_count_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --method loglinear_probaware
```

### 20.4 Compare model behavior

For final report preparation, compare:

* raw probability metrics by horizon;
* raw and calibrated count metrics by horizon;
* raw and calibrated capped count metrics;
* positive-count-only count metrics;
* true-count bucket diagnostics;
* prediction-count bucket diagnostics;
* magnitude metrics by horizon;
* train / validation / test gaps;
* count-tower negative result versus the earlier calibrated shared-MLP baseline.

---

## 21. Report-Ready Summary Paragraph

A multi-task PyTorch MLP was implemented for trigger-based aftershock forecasting, jointly predicting aftershock occurrence probability, aftershock count, and conditional maximum aftershock magnitude for 24h and 72h horizons. The model uses a shared tabular MLP trunk with separate probability, count, and magnitude heads. Probability prediction was the strongest task, with the best shared-trunk configuration reaching test ROC-AUC values of approximately 0.766 for 24h and 0.749 for 72h. Conditional maximum-magnitude prediction was also reasonably stable, with test MAE around 0.48 magnitude units. Count prediction remained the main difficulty because the target is zero-heavy and strongly right-skewed. Larger shared MLPs provided only modest improvement, and count diagnostics showed that the count head was not random but severely under-scaled: higher predicted-count buckets corresponded to higher true counts, yet moderate and high true-count buckets were still strongly underpredicted. A basic validation-fitted log-linear count calibration partially stretched the count scale and improved RMSE-based and positive-count-only metrics, but it worsened MAE by over-lifting zero and low-count rows. A probability-aware calibration model using both the original count prediction and the occurrence-probability prediction improved this tradeoff and became the best count post-processing method tried so far, but it still did not solve zero-count overprediction, high-count underprediction, or cap+ detection. A subsequent count-tower architecture experiment added a small count-specific hidden layer after the shared trunk, but this did not improve the raw count model and did not outperform the earlier probability-aware calibrated shared-MLP baseline. Overall, the evidence suggests that the count task is limited not only by head depth but also by the target formulation and the difficulty of learning positive-count severity under strong zero inflation and heavy tails.

---

## 22. Current Overall Conclusion

The neural pipeline is now a usable and extensible modeling framework.

Current strengths:

* stable modular pipeline;
* fair train / validation / test setup;
* reusable evaluation-only workflow;
* good probability prediction;
* promising conditional magnitude prediction;
* detailed count diagnostics and calibration workflow now available.

Current main weakness:

```text
aftershock count prediction remains under-scaled and conservative,
even after calibration.
```

Most important current findings are:

```text
The count model has useful ranking signal but poor raw-count calibration.
Probability-aware calibration is the best count post-processing method tried so far.
A simple count-specific tower did not solve the count problem.
```

Therefore, the next recommended direction is no longer merely:

```text
task-specific count tower
```

Instead, the updated recommendation is:

```text
move toward a more conceptually different count approach,
such as conditional count modeling or a count-specific objective,
rather than continuing small head-only tweaks.
```


## 14. Conditional-Positive Count Modeling Experiment

After the count-tower experiment, the next question was whether the count problem was caused more by the **training formulation** than by head capacity.

The earlier diagnostics suggested that the count head was learning a useful severity-ranking signal, but was being pulled toward overly conservative predictions because:

* the count target is zero-heavy;
* the target is strongly right-skewed;
* the model was trained to regress counts for both zero and positive rows in one head;
* the count tower did not clearly improve the raw representation.

This motivated a more structural count experiment:

```text
separate occurrence learning from positive-count severity learning
```

The key idea was to keep the existing probability head for event occurrence, while changing the count objective so the count head focuses on **positive-count severity**.

### 14.1 Implementation concept

A new configurable count-modeling mode was added:

```text
standard
conditional_positive
```

In `standard` mode, count behavior is unchanged:

* count loss uses all rows;
* inference uses:
  ```text
  count_prediction = clip(expm1(count_pred), min=0)
  ```

In `conditional_positive` mode:

* count loss is computed only on rows where the transformed count target is greater than zero, separately for each horizon;
* the count head is interpreted as positive-case severity on the log-count scale;
* the final count prediction is:

```text
count_prediction
=
sigmoid(prob_logits) * clip(expm1(count_pred), min=0)
```

Important note:

```text
This is not post-hoc calibration.
This changes how the neural model is trained and how its count output is interpreted.
```

The probability and magnitude heads remained unchanged. Output keys remained:

```python
{
    "prob_logits": ...,
    "count_pred": ...,
    "magnitude_pred": ...,
}
```

A command-line argument was added:

```bash
--count-modeling-mode standard
--count-modeling-mode conditional_positive
```

### 14.2 Why this experiment mattered

This experiment directly tested the hypothesis:

```text
The count problem may come more from zero-vs-positive conflict in the loss than from insufficient head depth.
```

This was important because the simple count-tower result suggested that extra count-head capacity alone was not enough.

Two conditional-positive runs were tested:

```text
mlp_cp_base
mlp_cp_t64
```

Where:

* `mlp_cp_base` = best shared trunk, no count tower;
* `mlp_cp_t64` = best shared trunk plus count tower `64 -> 2`.

Both used:

```text
feature_set = nn_enriched_v1
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 1.0
count_modeling_mode = conditional_positive
```

The commands were:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_cp_base \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --count-modeling-mode conditional_positive
```

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_cp_t64 \
  --hidden-dims 256 128 64 \
  --count-head-hidden-dims 64 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --count-modeling-mode conditional_positive
```

### 14.3 Raw uncapped count results

Test-set uncapped count metrics were:

| Horizon | Version                                | Test MAE | Test RMSE | Mean true | Mean pred |
| ------: | -------------------------------------- | -------: | --------: | --------: | --------: |
|     24h | Previous raw best (`h256_128_64_do03`) |   9.0700 |   28.4269 |    9.6268 |    1.4777 |
|     24h | `mlp_cp_base`                          |   9.2156 |   28.0167 |    9.6268 |    2.3185 |
|     24h | `mlp_cp_t64`                           |   9.2382 |   27.7798 |    9.6268 |    2.6802 |
|     72h | Previous raw best (`h256_128_64_do03`) |  13.5435 |   42.2253 |   14.4236 |    2.2448 |
|     72h | `mlp_cp_base`                          |  13.7949 |   41.7059 |   14.4236 |    3.4286 |
|     72h | `mlp_cp_t64`                           |  13.8391 |   41.4685 |   14.4236 |    4.0064 |

Interpretation:

* Both conditional-positive runs increased the mean prediction substantially relative to the earlier raw baseline.
* Both conditional-positive runs improved RMSE at both horizons relative to the earlier raw baseline.
* However, all-row MAE worsened at both horizons relative to the earlier raw baseline.
* This means the conditional-positive objective reduced some larger errors and reduced conservative scale compression, but it also over-lifted enough low-count / zero-count rows to hurt MAE.

This is a different tradeoff from the earlier raw shared-MLP model:

```text
conditional-positive helps severity and RMSE,
but not all-row MAE
```

### 14.4 Positive-count-only metrics

Because the conditional-positive experiment explicitly targets positive-case severity, the most important metrics are the positive-count-only metrics.

Test positive-only metrics were:

| Horizon | Version                                | Positive-only MAE | Positive-only RMSE | Mean true | Mean pred |
| ------: | -------------------------------------- | ----------------: | -----------------: | --------: | --------: |
|     24h | Previous raw best (`h256_128_64_do03`) |           18.6814 |            41.5171 |   20.5462 |    2.4774 |
|     24h | `mlp_cp_base`                          |           18.3208 |            40.8775 |   20.5462 |    3.6004 |
|     24h | `mlp_cp_t64`                           |           18.1789 |            40.5147 |   20.5462 |    4.1823 |
|     72h | Previous raw best (`h256_128_64_do03`) |           24.8174 |            57.9711 |   27.1981 |    3.5119 |
|     72h | `mlp_cp_base`                          |           24.5238 |            57.2116 |   27.1981 |    4.9765 |
|     72h | `mlp_cp_t64`                           |           24.4124 |            56.8664 |   27.1981 |    5.8711 |

Interpretation:

* This is the clearest success signal of the conditional-positive idea.
* Both conditional-positive runs improved positive-only MAE and RMSE at both horizons relative to the earlier raw baseline.
* Mean predictions on positive rows moved materially upward, especially for 72h.
* The tower version (`mlp_cp_t64`) was slightly better than the base version on positive-only metrics.

This suggests:

```text
conditional-positive training does improve positive-count severity learning
```

This is important because it confirms that the earlier count problem was not only about missing capacity. The training formulation itself mattered.

### 14.5 Capped count metrics

Test capped count metrics were:

| Horizon | Version                                | Capped MAE | Capped RMSE | True cap+ | Pred cap+ | Missed cap+ | False cap+ |
| ------: | -------------------------------------- | ---------: | ----------: | --------: | --------: | ----------: | ---------: |
|     24h | Previous raw best (`h256_128_64_do03`) |     8.9493 |     27.0131 |         6 |         0 |           6 |          0 |
|     24h | `mlp_cp_base`                          |     9.0949 |     26.6122 |         6 |         0 |           6 |          0 |
|     24h | `mlp_cp_t64`                           |     9.1175 |     26.3669 |         6 |         0 |           6 |          0 |
|     72h | Previous raw best (`h256_128_64_do03`) |    13.4316 |     40.8251 |         5 |         0 |           5 |          0 |
|     72h | `mlp_cp_base`                          |    13.6830 |     40.3140 |         5 |         0 |           5 |          0 |
|     72h | `mlp_cp_t64`                           |    13.7272 |     40.0756 |         5 |         0 |           5 |          0 |

Interpretation:

* Both conditional-positive runs improved capped RMSE relative to the earlier raw baseline.
* Both worsened capped MAE.
* Neither produced any true cap+ hits on test.
* So conditional-positive improved the moderate/high-count RMSE tradeoff without solving extreme-count detection.

### 14.6 True-count bucket diagnostics

The true-count bucket diagnostics explain the tradeoff more clearly.

#### `mlp_cp_base` test 24h buckets

| True bucket | Mean true | Mean pred |
| ----------- | --------: | --------: |
| 0           |    0.0000 |    1.1882 |
| 1           |    1.0000 |    2.0291 |
| 2–5         |    3.0220 |    2.5005 |
| 6–10        |    7.7683 |    3.7149 |
| 11–20       |   14.8452 |    4.5900 |
| 21–50       |   33.6089 |    4.7663 |
| 51–100      |   73.0000 |    6.0407 |
| 101–300     |  131.5000 |    9.2080 |
| 300+        |  370.6667 |   19.3147 |

#### `mlp_cp_t64` test 24h buckets

| True bucket | Mean true | Mean pred |
| ----------- | --------: | --------: |
| 0           |    0.0000 |    1.3558 |
| 1           |    1.0000 |    2.3473 |
| 2–5         |    3.0220 |    2.9396 |
| 6–10        |    7.7683 |    4.1982 |
| 11–20       |   14.8452 |    4.9194 |
| 21–50       |   33.6089 |    5.4156 |
| 51–100      |   73.0000 |    7.2474 |
| 101–300     |  131.5000 |   11.7883 |
| 300+        |  370.6667 |   20.1728 |

#### `mlp_cp_base` test 72h buckets

| True bucket | Mean true | Mean pred |
| ----------- | --------: | --------: |
| 0           |    0.0000 |    1.6809 |
| 1           |    1.0000 |    2.5391 |
| 2–5         |    2.9701 |    3.3450 |
| 6–10        |    7.6593 |    4.6970 |
| 11–20       |   14.9079 |    6.4060 |
| 21–50       |   34.2115 |    6.7591 |
| 51–100      |   72.8198 |    7.2125 |
| 101–300     |  143.8028 |   11.4725 |
| 300+        |  496.5000 |   23.1934 |

#### `mlp_cp_t64` test 72h buckets

| True bucket | Mean true | Mean pred |
| ----------- | --------: | --------: |
| 0           |    0.0000 |    1.9009 |
| 1           |    1.0000 |    2.9107 |
| 2–5         |    2.9701 |    3.9401 |
| 6–10        |    7.6593 |    5.3954 |
| 11–20       |   14.9079 |    7.0598 |
| 21–50       |   34.2115 |    7.5998 |
| 51–100      |   72.8198 |    8.8282 |
| 101–300     |  143.8028 |   14.9170 |
| 300+        |  496.5000 |   24.7713 |

Interpretation:

* Both conditional-positive runs lifted predictions across nearly all true-count buckets relative to the earlier raw baseline.
* The tower version generally lifted the moderate/high buckets more strongly than the base version.
* However, both also lifted the zero and one-count buckets sharply.
* This explains why positive-only metrics improved while all-row MAE worsened.

Most accurate interpretation:

```text
conditional-positive reduced scale compression,
but it also increased overprediction on zero / very-low-count rows
```

### 14.7 Comparison against the earlier probability-aware calibrated baseline

A key question is whether conditional-positive modeling improved the raw model enough to compete with the earlier post-hoc calibrated baseline.

Earlier best calibrated non-conditional model:

| Horizon | Version                                     | Test MAE | Test RMSE | Mean pred |
| ------: | ------------------------------------------- | -------: | --------: | --------: |
|     24h | `h256_128_64_do03` + prob-aware calibration |   9.1399 |   27.3409 |    3.5381 |
|     72h | `h256_128_64_do03` + prob-aware calibration |  13.5958 |   40.7277 |    4.6615 |

Comparison:

| Horizon | Version           | Test MAE | Test RMSE | Mean pred |
| ------: | ----------------- | -------: | --------: | --------: |
|     24h | `mlp_cp_base` raw |   9.2156 |   28.0167 |    2.3185 |
|     24h | `mlp_cp_t64` raw  |   9.2382 |   27.7798 |    2.6802 |
|     72h | `mlp_cp_base` raw |  13.7949 |   41.7059 |    3.4286 |
|     72h | `mlp_cp_t64` raw  |  13.8391 |   41.4685 |    4.0064 |

Interpretation:

* Raw conditional-positive models still did not beat the earlier probability-aware calibrated non-conditional pipeline on test MAE or RMSE.
* However, they were much closer than the earlier raw models.
* This suggests that conditional-positive improved the raw neural formulation itself, but not enough to fully replace the best calibrated pipeline.

### 14.8 Overall conclusion from the conditional-positive experiment

The conditional-positive experiment was a meaningful structural improvement.

What it showed:

* training formulation matters, not just head depth;
* positive-only count metrics improved for both the base and tower versions;
* RMSE improved relative to the earlier raw baseline;
* bucket diagnostics showed reduced count-scale compression;
* the tower still did not solve the count problem outright.

The downside:

* all-row MAE worsened;
* zero-count and one-count buckets were lifted too much;
* raw conditional-positive models still did not beat the earlier best probability-aware calibrated non-conditional model.

Most accurate interpretation:

```text
Conditional-positive modeling improved positive-count severity learning and reduced count-scale compression, confirming that the original count problem was partly caused by zero-vs-positive conflict in the count objective. However, it also increased overprediction on zero and low-count rows, so it improved the RMSE / positive-only tradeoff more than the MAE tradeoff.
```

---

## 15. Probability-Aware Calibration Applied To Conditional-Positive Models

After the raw conditional-positive models were evaluated, the same validation-fitted probability-aware log-linear calibration was applied to both:

```text
mlp_cp_base
mlp_cp_t64
```

This allowed two separate comparisons:

1. whether calibration still helps after the structural count change;
2. whether the conditional-positive models can now produce a better final practical pipeline.

### 15.1 Calibration parameters

For `mlp_cp_base`, the fitted probability-aware calibration parameters were:

| Horizon | Intercept | Count log slope | Probability slope | Val mean pred before | Val mean pred after |
| ------: | --------: | --------------: | ----------------: | -------------------: | ------------------: |
|     24h |    0.0932 |          1.2280 |           -0.4170 |               2.3605 |              3.1510 |
|     72h |    0.0463 |          1.0934 |           -0.0955 |               3.4844 |              4.3146 |

For `mlp_cp_t64`, the fitted parameters were:

| Horizon | Intercept | Count log slope | Probability slope | Val mean pred before | Val mean pred after |
| ------: | --------: | --------------: | ----------------: | -------------------: | ------------------: |
|     24h |    0.2079 |          1.5088 |           -1.4918 |               2.7197 |              3.3950 |
|     72h |    0.2320 |          1.3232 |           -1.1644 |               4.0205 |              4.6798 |

Important observation:

```text
The fitted probability coefficients became negative.
```

This is a notable change relative to the earlier non-conditional raw model, where probability-aware calibration used positive probability slopes.

Interpretation:

* In the raw conditional-positive models, probability weighting is already built into the count prediction.
* The calibrator is therefore not using probability to lift likely-positive rows further.
* Instead, it is using probability as an additional control to partially counteract overprediction in some regions of prediction space.
* This is consistent with the observed raw behavior: the conditional-positive models already lifted many zero / low-count rows.

### 15.2 `mlp_cp_base` calibrated results

Test-set uncapped results for `mlp_cp_base` were:

| Horizon | Version                  | Test MAE | Test RMSE | Mean pred |
| ------: | ------------------------ | -------: | --------: | --------: |
|     24h | Raw `mlp_cp_base`        |   9.2156 |   28.0167 |    2.3185 |
|     24h | Calibrated `mlp_cp_base` |   9.3051 |   27.4968 |    3.1050 |
|     72h | Raw `mlp_cp_base`        |  13.7949 |   41.7059 |    3.4286 |
|     72h | Calibrated `mlp_cp_base` |  13.8862 |   41.2063 |    4.2544 |

Positive-only results:

| Horizon | Version                  | Positive-only MAE | Positive-only RMSE | Mean pred |
| ------: | ------------------------ | ----------------: | -----------------: | --------: |
|     24h | Raw `mlp_cp_base`        |           18.3208 |            40.8775 |    3.6004 |
|     24h | Calibrated `mlp_cp_base` |           18.1592 |            40.0759 |    4.9265 |
|     72h | Raw `mlp_cp_base`        |           24.5238 |            57.2116 |    4.9765 |
|     72h | Calibrated `mlp_cp_base` |           24.4155 |            56.4950 |    6.2530 |

Capped results:

| Horizon | Version                  | Capped MAE | Capped RMSE | Pred cap+ | Missed cap+ | False cap+ |
| ------: | ------------------------ | ---------: | ----------: | --------: | ----------: | ---------: |
|     24h | Raw `mlp_cp_base`        |     9.0949 |     26.6122 |         0 |           6 |          0 |
|     24h | Calibrated `mlp_cp_base` |     9.1844 |     26.1172 |         0 |           6 |          0 |
|     72h | Raw `mlp_cp_base`        |    13.6830 |     40.3140 |         0 |           5 |          0 |
|     72h | Calibrated `mlp_cp_base` |    13.7743 |     39.8297 |         0 |           5 |          0 |

Interpretation:

* Calibration still improved RMSE and positive-only metrics for the base conditional-positive model.
* But calibration worsened MAE and capped MAE.
* Unlike the earlier non-conditional calibrated model, no cap+ false alarms were introduced.
* So the base conditional-positive model plus calibration has a somewhat cleaner cap+ profile, but not a better overall count score than the earlier best calibrated non-conditional path.

### 15.3 `mlp_cp_t64` calibrated results

Test-set uncapped results for `mlp_cp_t64` were:

| Horizon | Version                 | Test MAE | Test RMSE | Mean pred |
| ------: | ----------------------- | -------: | --------: | --------: |
|     24h | Raw `mlp_cp_t64`        |   9.2382 |   27.7798 |    2.6802 |
|     24h | Calibrated `mlp_cp_t64` |   9.4190 |   28.6210 |    3.5107 |
|     72h | Raw `mlp_cp_t64`        |  13.8391 |   41.4685 |    4.0064 |
|     72h | Calibrated `mlp_cp_t64` |  14.0798 |   43.0481 |    4.8605 |

Positive-only results:

| Horizon | Version                 | Positive-only MAE | Positive-only RMSE | Mean pred |
| ------: | ----------------------- | ----------------: | -----------------: | --------: |
|     24h | Raw `mlp_cp_t64`        |           18.1789 |            40.5147 |    4.1823 |
|     24h | Calibrated `mlp_cp_t64` |           18.3804 |            41.7075 |    5.7704 |
|     72h | Raw `mlp_cp_t64`        |           24.4124 |            56.8664 |    5.8711 |
|     72h | Calibrated `mlp_cp_t64` |           24.7508 |            59.0131 |    7.3663 |

Capped results:

| Horizon | Version                 | Capped MAE | Capped RMSE | Pred cap+ | Missed cap+ | False cap+ |
| ------: | ----------------------- | ---------: | ----------: | --------: | ----------: | ---------: |
|     24h | Raw `mlp_cp_t64`        |     9.1175 |     26.3669 |         0 |           6 |          0 |
|     24h | Calibrated `mlp_cp_t64` |     9.2065 |     26.1031 |         1 |           6 |          1 |
|     72h | Raw `mlp_cp_t64`        |    13.7272 |     40.0756 |         0 |           5 |          0 |
|     72h | Calibrated `mlp_cp_t64` |    13.8462 |     40.0342 |         1 |           5 |          1 |

Interpretation:

* Calibration harmed the tower conditional-positive model on almost every important metric.
* Uncapped MAE worsened.
* Uncapped RMSE worsened.
* Positive-only metrics worsened.
* False cap+ alarms were introduced at both horizons.

This is very different from the base conditional-positive case and from the earlier non-conditional tower case.

Most accurate interpretation:

```text
The tower conditional-positive model already pushed counts upward enough that the same probability-aware calibration became counterproductive.
```

### 15.4 Calibrated bucket behavior

Calibration still lifted the conditional-positive models further in the upper true-count buckets, but the tradeoff was mixed.

For `mlp_cp_base`, test 24h bucket means moved from:

```text
101–300:  9.21 -> 14.08
300+:    19.31 -> 29.88
```

and test 72h bucket means moved from:

```text
101–300: 11.47 -> 15.07
300+:    23.19 -> 30.73
```

For `mlp_cp_t64`, the lifts were even larger:

```text
24h:
101–300: 11.79 -> 23.90
300+:    20.17 -> 33.24

72h:
101–300: 14.92 -> 23.02
300+:    24.77 -> 33.89
```

However, the zero bucket also remained strongly elevated after calibration.

So calibration still stretches the compressed upper tail, but once conditional-positive training has already lifted counts substantially, the additional calibration becomes much riskier.

### 15.5 Overall conclusion from calibration on conditional-positive models

The calibrated conditional-positive results show a more nuanced picture than the earlier non-conditional calibration results.

Main conclusions:

* `mlp_cp_base` + probability-aware calibration still helps RMSE and positive-only metrics relative to raw `mlp_cp_base`, but not MAE.
* `mlp_cp_t64` + probability-aware calibration is not useful; it worsens most important metrics.
* The fitted negative probability slopes suggest that probability is now being used partly to suppress over-lift, not to amplify count scale.

This means:

```text
Once occurrence is already built into the count prediction through conditional-positive modeling, the old probability-aware calibrator is no longer doing the same job it did before.
```

That is an important methodological finding for the report.

---

## 16. Updated Current Best Neural Model

The notion of “best” now depends on whether the priority is:

* the best raw all-row MAE;
* the best raw RMSE / positive-only severity behavior;
* or the best practical calibrated end-to-end pipeline.

### 16.1 Best probability / magnitude architecture

This is still:

```text
mlp_multitask_v1_enriched_h256_128_64_do03
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 1.0
```

### 16.2 Best raw count MAE

Best raw test MAE still comes from the earlier non-conditional shared-MLP run:

| Horizon | Version            | Test MAE |
| ------: | ------------------ | -------: |
|     24h | `h256_128_64_do03` |   9.0700 |
|     72h | `h256_128_64_do03` |  13.5435 |

### 16.3 Best raw count RMSE and positive-only behavior among the new structural models

Among the conditional-positive runs:

* `mlp_cp_t64` had the best raw RMSE and best positive-only metrics;
* `mlp_cp_base` was slightly worse than `mlp_cp_t64` on those measures, but cleaner conceptually because it isolates the formulation change without mixing in the tower.

### 16.4 Best practical calibrated count pipeline

Best practical calibrated count pipeline still remains:

```text
mlp_multitask_v1_enriched_h256_128_64_do03
+
probability-aware post-processing calibration
```

Test metrics for that earlier best calibrated non-conditional pipeline:

| Horizon | Test MAE | Test RMSE | Positive-only MAE | Positive-only RMSE |
| ------: | -------: | --------: | ----------------: | -----------------: |
|     24h |   9.1399 |   27.3409 |           17.9706 |            39.8821 |
|     72h |  13.5958 |   40.7277 |           24.0467 |            55.8687 |

This still beats:

* calibrated `mlp_cp_base` on MAE and RMSE;
* calibrated `mlp_cp_t64` by a wide margin.

### 16.5 Most accurate “current best” statement

The most accurate summary now is:

```text
The earlier non-conditional shared-MLP plus probability-aware calibration remains the best practical end-to-end count pipeline tried so far. However, the conditional-positive experiments provided important evidence that count underprediction was partly caused by the original count objective, because positive-only severity metrics improved materially under conditional-positive training.
```

This is more precise than simply saying “conditional-positive failed” or “tower failed.”

---

## 17. Updated Results Summary By Task

### 17.1 Probability prediction

Probability prediction remains the strongest task.

Current best test results from `h256_128_64_do03`:

| Horizon | ROC-AUC | Brier score | Log loss | Positive rate |
| ------- | ------: | ----------: | -------: | ------------: |
| 24h     |  0.7664 |      0.1972 |   0.5746 |        0.4685 |
| 72h     |  0.7485 |      0.2042 |   0.5913 |        0.5303 |

### 17.2 Count prediction

Count prediction remains the hardest task.

The results now support three distinct statements:

1. **Best raw all-row MAE** still comes from the earlier non-conditional shared-MLP baseline.
2. **Best raw positive-only and RMSE tradeoff among the structural count experiments** came from the conditional-positive runs, especially `mlp_cp_t64`.
3. **Best final practical calibrated pipeline** still comes from the earlier non-conditional shared-MLP plus probability-aware calibration.

### 17.3 Capped count prediction

Conditional-positive improved capped RMSE relative to the earlier raw baseline, but it still did not solve cap+ detection. The earlier calibrated non-conditional model remains the best capped practical pipeline tried so far.

### 17.4 Maximum magnitude prediction

Magnitude prediction remains reasonably stable and was not materially changed by the count-focused experiments.

---

## 18. Updated Key Findings

### 18.1 Neural model viability

The multi-task MLP remains a useful neural baseline with meaningful probability signal and reasonable conditional magnitude performance.

### 18.2 Count underprediction is partly a training-formulation problem

The conditional-positive experiment is the strongest evidence yet that the original count problem was not only about insufficient head capacity.

Why:

* count tower alone did not help enough;
* conditional-positive training improved positive-only metrics and reduced scale compression.

### 18.3 Head capacity is still not the main bottleneck

The tower remained a mixed result:

* useful when combined with conditional-positive for raw positive-only metrics;
* but still not enough to create the best end-to-end pipeline;
* and clearly harmful once combined with the old probability-aware calibrator.

### 18.4 Calibration depends on the raw model semantics

The earlier probability-aware calibrator worked best when the raw count model was heavily compressed and did not already include occurrence weighting.

Once occurrence weighting was built into the raw count prediction through conditional-positive modeling:

* the fitted probability slopes turned negative;
* calibration helped the base conditional-positive model only modestly;
* calibration actively harmed the tower conditional-positive model.

This is an important methodological detail that would be easy to miss.

### 18.5 The report now has a stronger causal story

You can now say the following with much more confidence:

```text
Count underprediction was caused by both target complexity and the original count formulation.
Extra head depth alone was not enough.
Changing the count objective helped positive-count severity learning, but introduced new zero/low-count overprediction tradeoffs.
```

---

## 19. Updated Current Limitations

Current limitations now include:

* no early stopping, although best checkpoint selection is validation-based;
* count-loss values are not directly comparable across `standard` and `conditional_positive` runs because the supervised population differs;
* count remains modeled as continuous regression rather than with a count-specific distribution;
* the target remains zero-heavy and heavy-tailed;
* `log1p(count)` still compresses the upper tail;
* cap+ detection remains poor;
* the old probability-aware calibrator is not automatically suitable once the raw count model changes semantics;
* GCMT enrichment has still not clearly shown a robust benefit;
* repeated model selection on the test set should still be avoided.

---

## 20. Updated Recommended Next Experiments

### 20.1 Completed conditional-positive modeling

Conditional-positive modeling has now been completed in both:

```text
mlp_cp_base
mlp_cp_t64
```

This experiment produced a meaningful structural insight:

```text
positive-only severity learning improved,
but zero / low-count overprediction also increased
```

So the result should be treated as:

```text
a meaningful partial success,
not a new best end-to-end pipeline
```

### 20.2 Most promising next direction

The current evidence suggests the next count-focused experiment should target:

```text
conditional-positive modeling
+ a more selective way to control low-count / zero-count over-lift
```

Examples include:

* mild weighting or asymmetric penalty inside the conditional-positive objective;
* an alternative target transform less compressive than `log1p`;
* a more suitable calibrator for the new conditional-positive semantics;
* explicit modeling of occurrence and positive severity with separate evaluation logic.

### 20.3 Less promising immediate directions

Less promising immediate next steps are:

* simply making the tower bigger;
* reusing the old probability-aware calibrator unchanged on every new raw count model;
* capping training targets;
* repeatedly scaling the shared trunk larger.

---

## 21. Updated Current Recommended Workflow

### 21.1 If the goal is best practical current pipeline

Use:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --hidden-dims 256 128 64 \
  --dropout 0.3
```

Then:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03
```

Then:

```bash
python3 scripts/calibrate_count_predictions.py \
  --run-name mlp_multitask_v1_enriched_h256_128_64_do03 \
  --method loglinear_probaware
```

### 21.2 If the goal is to continue structural count research

Use the conditional-positive base run first:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_cp_base \
  --hidden-dims 256 128 64 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --count-modeling-mode conditional_positive
```

Then evaluate:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_cp_base
```

Use the tower version only as a secondary comparison:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_cp_t64 \
  --hidden-dims 256 128 64 \
  --count-head-hidden-dims 64 \
  --dropout 0.3 \
  --learning-rate 3e-4 \
  --count-loss-weight 1.0 \
  --count-modeling-mode conditional_positive
```

Then:

```bash
python3 scripts/evaluate_multitask_predictions.py \
  --run-name mlp_cp_t64
```

Only apply probability-aware calibration to conditional-positive runs if the goal is explicit comparison, because the old calibrator is no longer guaranteed to help.

---

## 22. Updated Report-Ready Summary Paragraph

A multi-task PyTorch MLP was implemented for trigger-based aftershock forecasting, jointly predicting aftershock occurrence probability, aftershock count, and conditional maximum aftershock magnitude for 24h and 72h horizons. The best shared-trunk configuration used hidden dimensions `256 -> 128 -> 64` with dropout `0.3`, achieving test ROC-AUC values of about 0.766 for 24h and 0.749 for 72h, while conditional maximum-magnitude prediction remained stable with MAE around 0.48. Count prediction was the main difficulty because the target is zero-heavy and strongly right-skewed. Larger shared MLPs produced only modest gains, and diagnostics showed that the raw count head learned useful severity-ranking signal but severe scale compression. Post-hoc log-linear calibration improved RMSE-based and positive-only count metrics but worsened MAE by over-lifting zero and low-count rows, while a probability-aware calibrator gave a better tradeoff and became the best practical calibrated count pipeline tried on the non-conditional shared-MLP model. A simple count-tower architecture did not solve the problem, suggesting that head depth alone was not the main bottleneck. A subsequent conditional-positive experiment changed the count objective so the count head was trained only on positive-count rows and the final count prediction became probability-weighted positive-case severity. This materially improved positive-only count metrics and reduced raw count-scale compression, confirming that the original count problem was partly caused by zero-vs-positive conflict in the training objective. However, it also increased overprediction on zero and low-count rows, so all-row MAE worsened even while RMSE improved relative to the earlier raw baseline. Applying the earlier probability-aware calibrator to the new conditional-positive models produced a mixed result: it still modestly helped the base conditional-positive model, but harmed the tower conditional-positive model, indicating that calibration behavior depends strongly on the semantics of the raw count model. Overall, the best practical end-to-end count pipeline tried so far remains the earlier non-conditional shared-MLP plus probability-aware calibration, but the conditional-positive results provide stronger evidence that count underprediction is driven not only by limited capacity but also by the original count-learning formulation.

---

## 23. Updated Overall Conclusion

The neural pipeline is now a usable and extensible framework with a much clearer understanding of the count task.

Current strengths:

* stable modular pipeline;
* fair train / validation / test setup;
* strong probability prediction;
* stable conditional magnitude prediction;
* detailed count diagnostics and calibration workflow;
* a completed structural count experiment that clarifies the role of the count objective.

Current main weakness remains:

```text
aftershock count prediction is still difficult because improving positive-count severity tends to increase overprediction on zero / low-count rows
```

The most important updated findings are:

```text
The count model has useful ranking signal but poor raw-count calibration.
A small count tower alone did not solve the problem.
Conditional-positive training improved positive-count severity learning and reduced scale compression.
The old probability-aware calibrator is not universally transferable once the raw count model semantics change.
The earlier non-conditional shared-MLP plus probability-aware calibration still remains the best practical end-to-end count pipeline tried so far.
```

Therefore, the next recommended direction is:

```text
continue from the conditional-positive idea,
but design the next experiment specifically to control zero / low-count over-lift
rather than simply adding more head capacity.
```


---

## 24. Probability and Maximum-Magnitude Focused Experiments

After the count-focused experiments above, a separate experimental branch was started to focus specifically on the two stronger tasks in the current multi-task neural model:

```text
1. aftershock occurrence probability
2. conditional maximum aftershock magnitude
```

This branch was intentionally kept incremental. The goal was not to redesign the model or change the overall neural pipeline. Instead, the goal was to test whether small, targeted changes could improve the probability and magnitude tasks while keeping the existing shared-trunk MLP setup intact.

Important scope decision:

```text
This branch does not focus on count modeling itself.
Count experiments continued separately, but they were not the objective here.
```

### 24.1 Motivation

Earlier experiments showed that:

* probability prediction was already one of the strongest parts of the model;
* maximum-magnitude prediction was also reasonably stable;
* count prediction was much harder and often dominated the interpretation of the multi-task model.

This raised a useful question:

```text
Are probability and magnitude already near their current ceiling,
or are they still being held back by the shared multi-task setup?
```

A particularly important hypothesis was that the count task might be exerting too much influence on the shared trunk, even though probability and magnitude were currently the tasks behaving best.

So the next experiments focused on:

* reducing count-task influence on the shared loss;
* testing whether a different probability loss improves probability quality;
* checking whether these changes also affect maximum-magnitude prediction.

### 24.2 Probability / magnitude baseline used for this branch

This branch started from the current best shared-trunk architecture found earlier:

```text
feature_set = nn_enriched_v1
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
```

The goal was therefore:

```text
keep the current best architecture fixed,
and change only one training choice at a time
```

This was important so that later differences could be interpreted cleanly.

### 24.3 Experiment: reducing count-loss weight for the probability / magnitude branch

The first experiment in this branch tested whether the count task was pulling the shared representation too strongly away from the needs of the probability and magnitude heads.

The total multi-task loss is:

```text
total_loss
=
probability_loss
+ count_loss_weight * count_loss
+ magnitude_loss_weight * magnitude_loss
```

Previously, the standard setting in the main shared-MLP runs was:

```text
count_loss_weight = 1.0
```

The hypothesis here was:

```text
If the count task is harder and noisier, reducing its contribution may allow
the shared trunk to learn representations that are slightly better for
probability discrimination and conditional magnitude prediction.
```

Three runs were compared:

```text
count_loss_weight = 1.0
count_loss_weight = 0.5
count_loss_weight = 0.25
```

Commands used:

Baseline:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_base_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 1.0
```

Reduced count influence:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5
```

Further reduced count influence:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_countw025_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.25
```

What this experiment was trying to find out:

```text
Can probability and maximum-magnitude improve if the shared trunk is allowed
to care slightly less about the count task?
```

This was not meant as a count experiment. It was a controlled test of shared-task interference.

Findings:

* reducing the count loss weight from `1.0` to `0.5` was beneficial for this branch;
* reducing it further to `0.25` did not continue improving the target tasks and appeared too aggressive.

Most accurate interpretation:

```text
The count task still provides useful shared signal,
but its default influence was slightly too strong for the probability and
maximum-magnitude branch.
```

Main conclusion from this experiment:

```text
count_loss_weight = 0.5
```

became the best setting for the probability / magnitude branch.

This is an important result because it shows that a small multi-task weighting change can improve the stronger tasks without changing architecture or labels.

### 24.4 Experiment: focal loss for the probability head

After reducing count influence, the next question was whether the probability head itself could be improved using a different classification loss.

The standard probability loss was:

```text
BCEWithLogitsLoss
```

A focal-loss option was then added so the probability head could be trained with either:

```text
bce
focal
```

using a configurable focal parameter:

```text
gamma
```

Two focal-loss runs were tested on top of the improved shared-loss setting:

```text
count_loss_weight = 0.5
probability_loss = focal, gamma = 1.5
probability_loss = focal, gamma = 1.0
```

A BCE control run with the same count-loss setting was also run for direct comparison.

Commands used:

BCE control:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_bce_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5 \
  --probability-loss bce
```

Focal gamma 1.5:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_focalg15_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5 \
  --probability-loss focal \
  --focal-gamma 1.5
```

Focal gamma 1.0:

```bash
python3 scripts/train_multitask_mlp.py \
  --dataset-name earthquake_aftershock_v2_gcmt \
  --feature-set nn_enriched_v1 \
  --run-name mlp_pm_focalg10_countw05_enriched_256_128_64_d03_lr3e4 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --hidden-dims 256 128 64 \
  --count-loss-weight 0.5 \
  --probability-loss focal \
  --focal-gamma 1.0
```

What this experiment was trying to find out:

```text
After reducing count-task interference,
can probability discrimination improve further if the probability head focuses
more strongly on harder examples?
```

The motivation for trying focal loss was that it can sometimes improve classification ranking by downweighting easier cases.

Findings:

* focal loss did not improve the probability task in a convincing way;
* focal loss produced at most very small ROC-AUC changes;
* Brier score and log loss became consistently worse;
* the larger focal setting (`gamma = 1.5`) was clearly worse;
* focal loss also did not improve maximum-magnitude performance as a side effect.

Main conclusion from this experiment:

```text
BCE remained the better probability loss for this setup.
```

This is an informative negative result.

It suggests that, in this project:

* the probability task is not primarily limited by the need for a harder-example-focused classification loss;
* the more useful improvement came from reducing shared-task interference, not from replacing BCE.

### 24.5 Current interpretation of the probability / magnitude branch

Taken together, the probability / magnitude branch produced a clearer picture of the multi-task neural model.

Main findings:

1. **Probability and magnitude are not fully saturated.**  
   They can improve slightly through targeted multi-task adjustments.

2. **Shared-task interference is real.**  
   Reducing count-loss weight from `1.0` to `0.5` improved the branch focused on probability and maximum magnitude.

3. **Focal loss is not the right next lever here.**  
   It did not produce a better overall probability model.

4. **Maximum-magnitude prediction remains stable but still somewhat conservative.**  
   It did not collapse during these experiments, which is encouraging, but it also did not improve dramatically from a probability-loss change.

Most accurate interpretation:

```text
The strongest recent improvement for the probability / magnitude branch came
from improving the multi-task balance, not from changing the classification loss.
```

### 24.6 Current best configuration for the probability / magnitude branch

The best current configuration for this branch is:

```text
feature_set = nn_enriched_v1
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 0.5
probability_loss = BCE
```

Most accurate statement:

```text
This is currently the most promising configuration for continuing work on
probability and conditional maximum-magnitude prediction.
```

It is still a small, incremental modification of the earlier best shared-trunk MLP, not a new architecture.


### 24.7 Next experiments and what happened next

After the focal-loss experiment, the next hypothesis was that the probability and maximum-magnitude tasks might benefit from a small amount of **task-specific nonlinear capacity**.

The shared-trunk model already uses:

```text
shared trunk -> task head
````

At this point, the probability head and magnitude head were still simple linear projections from the shared representation. So the next idea was:

```text
keep the shared trunk unchanged,
but give probability and/or magnitude a small task-specific tower
```

The goal was not to redesign the model. It was a controlled test of whether the bottleneck for the stronger tasks was simply that their heads were too shallow.

The implementation was kept minimal and reversible:

* the shared trunk stayed the same;
* the output dictionary keys stayed the same;
* loss functions stayed the same;
* prediction CSV schemas stayed the same;
* only the probability head and/or magnitude head were allowed to become small MLP towers.

This made it possible to test head-capacity changes without changing the rest of the pipeline.

---

## 25. Probability and Magnitude Tower Experiments

After the probability / magnitude branch showed that reducing `count_loss_weight` from `1.0` to `0.5` was useful, and that focal loss did not help, the next question was whether the stronger tasks were being limited by **head capacity**.

The hypothesis was:

```text
Perhaps the shared trunk is already good enough,
but the probability and/or magnitude heads need a small task-specific tower
to make better use of the shared representation.
```

This was tested in a controlled way by adding optional hidden layers to the probability head and magnitude head while keeping the rest of the neural pipeline unchanged.

### 25.1 Motivation

This experiment was motivated by the following observations:

* the probability task was already reasonably strong;
* the magnitude task was stable but still slightly conservative;
* focal loss did not improve the probability task;
* the earlier count-focused experiments suggested that not every bottleneck is solved by making the shared trunk larger.

So the next targeted question became:

```text
Can the probability and/or magnitude tasks improve if they receive
a small amount of task-specific nonlinear capacity after the shared trunk?
```

### 25.2 Architecture change tested

The shared-trunk architecture remained:

```text
shared trunk: 256 -> 128 -> 64
dropout: 0.3
```

The baseline heads were simple linear projections from the shared representation.

The tower experiments added optional head-specific hidden layers:

* probability tower:

  ```text
  64 -> 64 -> 2
  ```
* magnitude tower:

  ```text
  64 -> 64 -> 2
  ```

No changes were made to:

* dataset or label definitions;
* shared trunk structure;
* loss functions;
* prediction output schemas;
* evaluation logic.

This made the experiment a controlled test of head capacity only.

### 25.3 Experimental setup

All tower experiments used the current best probability / magnitude branch settings:

```text
feature_set = nn_enriched_v1
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 0.5
probability_loss = BCE
```

Three runs were compared against the BCE + `count_loss_weight = 0.5` baseline:

1. magnitude tower only
2. probability tower only
3. both towers together

The intended logic was:

```text
test each tower separately first,
then test both together only after the individual effects are understood
```

### 25.4 Magnitude tower results

The magnitude-tower experiment was the most directly motivated, because maximum-magnitude prediction was stable but still slightly conservative.

However, the magnitude tower did not improve performance.

Compared with the BCE + `count_loss_weight = 0.5` baseline:

| Horizon | Split | Baseline MAE | Magnitude-tower MAE | Interpretation |
| ------: | ----- | -----------: | ------------------: | -------------- |
|     24h | Val   |       0.5589 |              0.5643 | Worse          |
|     24h | Test  |       0.4700 |              0.4771 | Worse          |
|     72h | Val   |       0.5629 |              0.5723 | Worse          |
|     72h | Test  |       0.4720 |              0.4810 | Worse          |

Interpretation:

* the degradation was consistent across validation and test;
* this was not just noise in one split;
* the magnitude tower did not reduce the conservative tendency in a useful way.

Most accurate interpretation:

```text
The magnitude task does not appear to be limited by lack of head depth.
A small magnitude-specific tower increased flexibility,
but worsened generalization.
```

A likely reason is that the magnitude task is only supervised on rows where aftershocks exist, so its effective sample size is smaller than the full dataset. A simple linear head may therefore be acting as a useful regularizer.

### 25.5 Probability tower results

A probability-specific tower was also tested.

The result was weaker than hoped:

* ROC-AUC remained roughly similar;
* Brier score was similar or slightly worse;
* log loss was similar or slightly worse.

So the probability tower did not produce a convincing improvement either.

Interpretation:

* the probability task is already being learned reasonably well by the shared trunk plus a linear head;
* adding extra nonlinear head capacity did not clearly improve discrimination or probability quality;
* if anything, it risked slightly worse calibration/generalization.

Most accurate interpretation:

```text
The probability task is not currently bottlenecked by head capacity.
```

### 25.6 Tower experiments overall conclusion

The tower experiments are an informative negative result.

They show that:

* the stronger tasks are **not** obviously limited by shallow heads;
* increasing task-specific head capacity did not improve probability prediction;
* increasing task-specific head capacity clearly worsened magnitude prediction;
* the main useful recent improvement for this branch came from **multi-task loss reweighting**, not from deeper heads.

This is useful because it narrows the search space.

Most accurate summary:

```text
The bottleneck for the probability / magnitude branch is not simply that the task heads are too shallow.
Small task-specific towers did not improve the branch and should not be treated as the new preferred architecture.
```

### 25.7 Updated interpretation of the probability / magnitude branch

After the loss-weight experiment, focal-loss experiment, and tower experiments, the updated interpretation is:

1. **Probability and maximum magnitude are reasonably strong tasks already.**
2. **Reducing count-task interference helped.**
3. **Changing the probability loss to focal loss did not help.**
4. **Adding task-specific towers did not help.**

So the current limitation for this branch is more likely to come from:

* feature information;
* task interference / optimization tradeoffs;
* model-selection strategy;
* calibration and generalization,

rather than from missing head depth.

This is an important conclusion because it means future work should focus more on:

* selecting checkpoints based on the target task of interest;
* improving features;
* exploring calibration or evaluation choices for probability;
* making only targeted architectural changes if they are very well motivated,

rather than continuing to scale up the heads.

---

### 26.2 Best current configuration for the probability / magnitude branch

For the probability and conditional maximum-magnitude branch, the current best configuration remains:

```text
feature_set = nn_enriched_v1
hidden_dims = 256 128 64
dropout = 0.3
learning_rate = 3e-4
count_loss_weight = 0.5
probability_loss = BCE
```

Important update:

* focal loss did not improve this branch;
* probability and magnitude towers did not improve this branch;
* therefore the preferred architecture remains the shared-trunk MLP with simple heads.

Most accurate interpretation:

```text
The best recent improvement for the probability / magnitude branch came from
reducing count-task influence, not from changing the probability loss or
increasing task-specific head depth.
```

---

### 27.2 If continuing the probability / magnitude branch

The next recommended step is **not** another focal-loss variant and **not** a larger task-specific tower.

Those ideas have now been tested and did not produce a convincing improvement.

The more appropriate next direction is:

```text
better checkpoint selection / task-specific model selection
```

Reasoning:

* the current training loop still selects the best checkpoint using total validation loss;
* total loss includes probability, count, and magnitude together;
* but this branch is specifically focused on:

  * probability prediction
  * maximum-magnitude prediction

So a useful next question is:

```text
Does the training process already visit checkpoints that are better for
probability or magnitude individually, even if they are not the best by total loss?
```

So the updated recommended next experiment for this branch is:

```text
keep the same architecture and BCE setup,
but compare checkpoint selection strategies more carefully
```
