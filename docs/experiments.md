# Experiments and Key Findings

This document summarises the main experimental findings across all tasks and modeling approaches.
Rather than listing every trial, it focuses on **what worked, what did not, and why**.

---

## 1. Overview

The project explores three forecasting tasks:

* **Task 1**: Aftershock occurrence (classification)
* **Task 2**: Aftershock count (regression, zero-inflated)
* **Task 3**: Aftershock severity (large-event classification + magnitude regression)

Two main modeling strategies were used:

* **Task-specific models** (e.g. XGBoost, Two-Head NN)
* **Joint multi-task neural network (MLP)**

---

## 2. Final Model Comparison

The table below compares the **best task-specific models** against the **multi-task neural network**.

| Task        | Metric  | Horizon | Best Task Model       | Multi-Head NN |
| ----------- | ------- | ------- | --------------------- | ------------- |
| Probability | ROC-AUC | 24h     | 0.8276 (XGBoost)      | 0.783         |
|             |         | 72h     | 0.7966 (XGBoost)      | 0.763         |
| Count       | MAE     | 24h     | 8.756 (Two-Head NN)   | 8.89          |
|             |         | 72h     | 13.2819 (Two-Head NN) | 13.344        |
| Magnitude   | MAE     | 24h     | 0.438 (XGBoost)       | 0.482         |
|             |         | 72h     | 0.4385 (XGBoost)      | 0.486         |

### Key takeaway

* The **multi-task neural network is competitive**, but
* **task-specific models consistently perform better**

---

## 3. Task-Specific Insights

### 3.1 Task 1 — Probability

* XGBoost achieves the best performance across both horizons
* Neural models (MLP, FT-Transformer) are competitive but slightly worse
* Performance degrades for 72h → longer horizon = weaker signal

**Key insight:**

> Tree-based models outperform neural networks on medium-sized tabular data.

---

### 3.2 Task 2 — Count (Most Challenging Task)

This is the **hardest problem in the project**.

#### Challenges:

* Highly **zero-inflated** (many zeros)
* Strong **right skew / heavy tail**
* Extreme outliers (especially in 2023 validation set)

#### What worked:

* Log transform (`log1p(count)`)
* Two-stage modeling:

  ```text
  E[count] = P(count > 0) × E[count | count > 0]
  ```
* Two-Head Neural Network (best MAE)

#### What did NOT work well:

* Single regression model (struggles with zero inflation)
* Hyperparameter tuning alone (limited gains)
* ZINB model (unstable training)

**Key insight:**

> The main difficulty is **data distribution**, not model capacity.

---

### 3.3 Task 3 — Magnitude & Large Aftershocks

#### Task 3a (Large aftershock classification)

* Much harder than Task 1 due to **rare events**
* XGBoost performs best
* Improvements over baseline are smaller

#### Task 3b (Max magnitude regression)

* More stable than classification
* XGBoost achieves best results
* Neural models do not outperform tree-based models

**Key insight:**

> Rare-event prediction remains fundamentally difficult, even with ML.

---

## 4. Multi-Task Neural Network (MLP)

The multi-head MLP was designed to jointly predict:

* probability
* count
* magnitude

using a shared representation.

---

### 4.1 What Worked

* Strong performance on **probability prediction**
* Stable **magnitude predictions**
* Learns useful shared structure across tasks

---

### 4.2 What Did Not Work

#### ❗ Count prediction remains weak

Observed issues:

* Underestimation of large counts
* Compressed prediction range
* Difficulty handling zero-inflation

---

### 4.3 Task Interference

Reducing count loss weight improved:

* probability performance
* magnitude performance

This indicates:

> The count task negatively interferes with other tasks during joint training.

---

## 5. Additional Neural Experiments

### 5.1 Conditional Count Modeling

Idea:

* Train count model only on positive cases

Result:

* Better predictions for non-zero counts
* Worse overall MAE due to overprediction on zeros

**Conclusion:**

> Trade-off between accuracy on positive cases vs overall performance

---

### 5.2 Probability Checkpoint Selection

Instead of selecting best model using total loss:

* select using probability loss only

Result:

* Significant improvement in ROC-AUC, Brier, log-loss

**Conclusion:**

> Model selection strategy matters as much as model design

---

### 5.3 Focal Loss

Tried replacing BCE with focal loss.

Result:

* Worse calibration
* Slightly worse overall performance

**Conclusion:**

> BCE is better suited for this task than focal loss

---

## 6. Key Lessons Learned

### 6.1 Data matters more than model

* Feature engineering and dataset consistency had larger impact than architecture changes

---

### 6.2 Tree-based models are strong baselines

* XGBoost consistently outperformed neural networks
* Especially effective for:

  * tabular data
  * missing values
  * medium dataset size (~23k samples)

---

### 6.3 Count modeling is fundamentally difficult

Challenges:

* zero inflation
* heavy tails
* rare extreme events

No single model fully solves this.

---

### 6.4 Multi-task learning has trade-offs

* Pros:

  * shared representation
  * unified framework
* Cons:

  * task interference
  * harder optimisation

---

## 7. Final Conclusion

* **Best overall approach**: task-specific models (XGBoost / Two-Head NN)
* **Multi-task NN**: useful as a unified baseline, but not the top performer
* **Biggest challenge**: modeling aftershock count
* **Most reliable signal**: probability of occurrence

---

## 8. Future Directions

* better handling of zero-inflated distributions
* hybrid models (tree + neural)
* improved probability calibration
* larger datasets for deep learning models
