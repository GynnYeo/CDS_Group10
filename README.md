# CDS_Group10

**Earthquake Aftershock Forecasting using Machine Learning**

This repository contains a unified pipeline for **short-term earthquake aftershock forecasting**, combining traditional statistical baselines with modern machine learning and neural network models.

The project is built around a **shared trigger-level dataset**, ensuring all models are trained and evaluated on the same data for fair comparison.

---

## 📌 Project Overview

After a major earthquake (mainshock), predicting aftershock behaviour is critical for hazard assessment and response planning.

This project reframes aftershock forecasting as a **multi-task learning problem**, predicting:

* **Probability** of aftershock occurrence
* **Number** of aftershocks
* **Maximum magnitude** of aftershocks

across two forecast horizons:

* **24 hours**
* **72 hours**

The core research question:

> *Can machine learning models improve short-term aftershock forecasting compared to traditional statistical approaches?*

---

## 🧠 Problem Setup

Each row in the dataset represents a **trigger earthquake**:

```text
Trigger = earthquake with magnitude ≥ 5.0
```

An aftershock is defined as an event:

* occurring after the trigger
* within a forecast horizon (24h / 72h)
* magnitude ≥ 2.5
* within 50 km of the trigger

### 🎯 Targets

| Task   | Description                                             |
| ------ | ------------------------------------------------------- |
| Task 1 | Probability of at least one aftershock                  |
| Task 2 | Number of aftershocks                                   |
| Task 3 | Aftershock severity (larger aftershock + max magnitude) |

---

## ⚙️ Official Workflow

All experiments follow a consistent pipeline:

```text
1. Download raw data
2. Build processed dataset
3. Run baselines
4. Run machine learning / neural models
5. Evaluate and compare results
```

This ensures:

* reproducibility
* no data leakage
* fair model comparison

---

## 🚀 Quick Start

### 1. Download raw data

```bash
python scripts/download_raw_data.py
```

### 2. Build processed dataset

```bash
python -m src.dataset.build_dataset \
    --comcat-input data/raw/comcat \
    --moment-tensor-input data/raw/moment_tensor \
    --dataset-name earthquake_aftershock_v2_gcmt
```

### 3. Run baselines

```bash
python scripts/run_and_save_baselines.py
```

### 4. Run neural model

```bash
python scripts/train_multitask_mlp.py
```

---

## 📂 Repository Structure

```text
CDS_Group10/
├── README.md
├── docs/
├── scripts/
├── src/
├── data/
├── reports/
└── tests/
```

### Key Directories

| Folder            | Purpose                              |
| ----------------- | ------------------------------------ |
| `scripts/`        | Entry points for running pipelines   |
| `src/dataset/`    | Shared preprocessing pipeline        |
| `src/models/`     | Baselines and neural models          |
| `src/evaluation/` | Metrics and evaluation               |
| `data/`           | Raw, interim, and processed datasets |
| `reports/`        | Metrics, predictions, checkpoints    |
| `docs/`           | Detailed documentation               |

---

## 📚 Documentation

Detailed explanations are kept in `docs/` to keep this README concise.

### Core Pipeline

* `docs/data_pipeline.md` — dataset construction and preprocessing
* `docs/system_design.md` — pipeline architecture and design

### Modeling

* `docs/modeling/task1_probability.md` — Task 1 (aftershock occurrence) *(to be added)*
* `docs/modeling/task2_count.md` — Task 2 (aftershock count) *(to be added)*
* `docs/modeling/task3_magnitude.md` — Task 3 (aftershock severity)
* `docs/modeling/multihead_nn.md` — multi-task neural network

### Experiments

* `docs/experiments.md` — experiment findings, trade-offs, and insights

### Additional

* `docs/gui.md` — interactive prediction interface *(optional)*

---

## 🤖 Models

### Baselines

* Climatology model
* Simplified Reasenberg–Jones (RJ) model
* Mean and domain-based baselines

### Machine Learning

* XGBoost (primary strong baseline)
* TabNet
* Linear models

### Neural Networks

* Multi-Layer Perceptron (MLP)
* Multi-task shared-trunk neural network
* Task-specific neural architectures (e.g. Two-Head NN)

---

## 📊 Current Status

### ✅ Implemented

* Shared preprocessing pipeline
* Trigger-level dataset construction
* Baseline models
* Machine learning models (XGBoost, TabNet)
* Multi-task neural network
* Evaluation framework

### 🚧 In Progress

* Task 1 documentation
* Task 2 documentation
* Additional neural experiments
* Model calibration improvements

---

## 📈 Key Insights (So Far)

* **XGBoost performs strongest** across most tasks on tabular data
* **Count prediction is the hardest task** due to zero inflation and heavy tails
* **Multi-task neural networks provide useful shared structure**, but do not outperform specialised models
* **Data consistency and preprocessing are critical** for fair comparison

---

## 🖥️ GUI (Optional)

A simple Gradio-based interface allows:

* uploading earthquake data
* generating predictions
* visualising results on a map

See: `docs/gui.md`

---

## 👥 Team

* Yeo Yee Gynn
* Tonie Enriquez Caponpon
* Hoon Kiah Yen
* Natasha Chan

---

## 📄 Report

Full project report:

* See: `CDS_Project_Report.pdf`

---

## 📌 Main Rule for Contributors

> All models must use the same processed dataset and splits.

Do **not** create separate preprocessing pipelines.

---

## 📬 Repository

GitHub:
https://github.com/GynnYeo/CDS_Group10.git

---
