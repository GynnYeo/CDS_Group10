# CDS_Group10 — Earthquake Aftershock Forecasting

## Overview

This repository implements a **data-driven aftershock forecasting pipeline**.

Given a trigger earthquake (magnitude ≥ 5.0), the goal is to forecast:

- the **probability of at least one aftershock** within:
  - 24 hours
  - 72 hours
- the **expected number of aftershocks** within:
  - 24 hours
  - 72 hours

This project focuses on building a **clean, reusable dataset pipeline** and comparing:

- simple baselines
- classical seismological models (Reasenberg–Jones)
- machine learning / deep learning models (later stages)

---

## Problem Definition

### Trigger event

- any earthquake with **magnitude ≥ 5.0**

### Aftershock definition

A qualifying aftershock must satisfy:

- occurs after the trigger event
- within **24h / 72h**
- within **50 km**
- magnitude ≥ **2.5**

---

## Prediction Targets

### Primary (probability)

- `y_24h`: probability of ≥1 aftershock in 24h  
- `y_72h`: probability of ≥1 aftershock in 72h  

### Secondary (counts)

- `n_aftershocks_24h`
- `n_aftershocks_72h`

---

## Data Split

Time-based split (no leakage):

- **Train:** 2015–2023  
- **Validation:** 2024  
- **Test:** 2025  

---

# Dataset Design

## Core principle

> Build once → reuse everywhere

A single **canonical event dataset** is constructed, then used to derive:

- trigger dataset
- labels
- features
- final modeling table

---

## Data Sources

### Dataset 1 (Primary)

**USGS ComCat Earthquake Catalog**

Used for:

- all earthquake events
- trigger selection
- aftershock labeling
- historical features

This is the **master dataset**.

---

### Dataset 2 (Optional enrichment)

**Global CMT (moment tensor)**

Provides:

- strike
- dip
- rake
- centroid depth
- GCMT magnitude

Important:

- coverage is **partial**
- matching is **not exact**
- used as **optional enrichment only**

---

## Key Design Decision

- ComCat remains the **source of truth**
- GCMT is **left-joined enrichment**
- missing GCMT data is **allowed**
- no ComCat rows are ever dropped

---

# GCMT Merge Strategy (Important)

## Why exact merge is not possible

ComCat and Global CMT are **independent catalogs**:

- no shared event ID
- slight differences in:
  - time
  - location
  - magnitude

So a standard join is impossible.

---

## Solution: nearest-neighbor matching under physical constraints

Instead of exact matching, events are matched using:

- time proximity
- spatial proximity
- magnitude similarity

---

## Matching rules

### Strict match

- time difference ≤ **30 seconds**
- distance ≤ **30 km**
- magnitude difference ≤ **0.3**

### Relaxed fallback

- time difference ≤ **120 seconds**
- distance ≤ **100 km**
- magnitude difference ≤ **0.5**

---

## Selection logic

If multiple candidates exist:

1. smallest time difference  
2. smallest distance  
3. smallest magnitude difference  

If still ambiguous:

- **no match is assigned**

---

## Why this is valid

- earthquake catalogs are not perfectly aligned
- constraints enforce **physical plausibility**
- matching is **conservative (precision > recall)**
- GCMT is **optional enrichment**, not core data

> Missing matches are acceptable. Incorrect matches are avoided.

---

# Dataset Pipeline

The canonical pipeline is implemented in:

```

src/dataset/build_dataset.py

````

### Pipeline steps

1. Load and clean ComCat data  
2. Load and clean GCMT data  
3. Match GCMT onto ComCat (optional enrichment)  
4. Build trigger events (M ≥ 5.0)  
5. Generate labels (24h / 72h)  
6. Build features (no leakage)  
7. Assemble final dataset  
8. Apply train/val/test split  

---

## Output

Final dataset:

- one row per trigger event
- includes:
  - trigger metadata
  - features
  - labels
  - split assignment

---

# Repository Structure

```text
CDS_Group10
├─ data/                         # Stored datasets and pipeline outputs
│  ├─ raw/                       # Raw ComCat and moment tensor source files
│  ├─ interim/                   # Cleaned, enriched, and intermediate pipeline tables
│  └─ processed/                 # Final modeling datasets and split-specific outputs
├─ src/                          # Active source code for the v2 pipeline
│  ├─ app/                       # Reserved for app or interface code; currently empty
│  ├─ baselines/                 # Baseline forecasting models and prediction interfaces
│  ├─ data/                      # Data ingestion, cleaning, downloading, and catalog enrichment
│  │  ├─ comcat/                 # USGS ComCat retrieval, raw loading, and cleaning utilities
│  │  ├─ merge/                  # Fuzzy matching and left-join enrichment with GCMT data
│  │  └─ moment_tensor/          # GCMT download, NDK parsing, and cleaning helpers
│  ├─ dataset/                   # Trigger construction, labels, features, assembly, and splits
│  ├─ features/                  # Reserved for standalone feature modules; currently empty
│  ├─ models/                    # Reserved for training code; currently only cached bytecode exists
│  └─ utils/                     # Shared helpers for paths, file I/O, and geospatial calculations
├─ archive/
│  └─ v1/                        # Archived first-version pipeline, models, notebooks, and reports
└─ README.md
````

---

# Baselines

Planned baseline ladder:

1. **Climatology**

   * constant probability baseline

2. **Reasenberg–Jones (RJ)**

   * classical aftershock model

3. **ETAS (optional)**

   * advanced self-exciting model

---

# Current Status

✅ Completed:

* ComCat ingestion and cleaning
* GCMT parsing and enrichment
* trigger construction
* label generation (24h / 72h + counts)
* minimal feature generation
* dataset assembly
* train/val/test split

➡️ Next:

* implement climatology baseline
* implement RJ baseline
* evaluate models


---

# Versioning

## v2 (current)

* probabilistic + count forecasting
* canonical dataset pipeline
* optional GCMT enrichment

## v1 (archived)

* time-to-next-aftershock formulation
* moved to `archive/v1/`

---

# Key Design Principles

* ComCat-centered pipeline
* optional enrichment (never required)
* no data leakage
* modular and reusable design
* conservative matching (avoid false matches)
* simple v1, extensible later

---

# Summary

This project builds a **clean, reusable earthquake dataset pipeline** and evaluates whether:

> machine learning models can improve short-term aftershock forecasting compared to traditional approaches.

The focus is on:

* correctness of dataset construction
* fair evaluation
* practical and defensible modeling decisions

