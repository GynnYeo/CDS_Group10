# Feature Engineering Documentation

**Task 1 — Aftershock Occurrence Probability (`y_24h`, `y_72h`)**  
**Models:** XGBoost · MLP · TabNet · FT-Transformer

---

## 1. Overview

All four Task 1 models — XGBoost, MLP, TabNet, and FT-Transformer — share an identical feature engineering pipeline implemented in the `engineer_features()` function. The function accepts a raw split dataframe and returns an enriched copy with 35 additional columns on top of the base feature sets (`BASE_TABULAR_FEATURES + QUALITY_FEATURES + GCMT_FEATURES`), bringing the total feature count to 91 (MLP, TabNet, XGBoost) or 83 (FT-Transformer, which uses a slightly smaller GCMT set).

**Why a shared pipeline?**  
Keeping one `engineer_features()` function across all models ensures that differences in test-set performance reflect the model architecture and training strategy only — not differences in the input representation. This makes results directly comparable.

**GCMT-only features**  
Features derived from the Global CMT moment tensor (scalar moment, eigenvalues, rake angle, etc.) are only populated for events with a matched GCMT solution (~80% of the dataset). For the remaining ~20%, these columns are set to `NaN`. Each model handles this differently:

- **XGBoost** and **TabNet** use `missing_strategy="none"` (`NaN` passed directly)
- **MLP** and **FT-Transformer** use `missing_strategy="median"` (imputed before training)

---

## 2. Feature Groups

### 2.1 Depth Regime (3 features)

Fault depth is one of the strongest physical predictors of aftershock productivity. However, the relationship between depth and aftershock occurrence is not linear — it follows three physically distinct regimes governed by the mechanical properties of the rock at each depth range.

#### Physical motivation

1. **Shallow earthquakes (< 70 km)** rupture brittle upper-crustal rock where stress transfer is most efficient, fluid content is highest, and aftershock sequences are longest and most numerous. This depth range includes most destructive earthquakes.
2. **Intermediate earthquakes (70–300 km)** occur within subducting oceanic slabs where mineralogical phase transitions alter rock strength. Their aftershock productivity is significantly lower than shallow events.
3. **Deep earthquakes (> 300 km)** occur in the mantle transition zone where surrounding rock deforms plastically rather than fracturing. Deep-focus earthquakes produce very few aftershocks, and the Omori law decay is much less pronounced.

#### Why binary flags rather than raw depth?

A single continuous depth value forces the model to discover the non-linear, step-like relationship between depth and aftershock productivity during training. Three binary flags encode this seismological knowledge explicitly, giving tree-based models (XGBoost) a direct splitting criterion and giving neural networks a pre-processed signal.

| Feature Name         | Formula / Code                  | Notes                               |
| -------------------- | ------------------------------- | ----------------------------------- |
| `depth_shallow`      | `trigger_depth_km < 70`         | 1 if shallow crustal, 0 otherwise   |
| `depth_intermediate` | `70 <= trigger_depth_km <= 300` | 1 if intermediate/slab, 0 otherwise |
| `depth_deep`         | `trigger_depth_km > 300`        | 1 if deep mantle, 0 otherwise       |

---

### 2.2 Magnitude and Seismicity Activity (7 features)

Earthquake magnitudes and seismicity counts both follow strongly skewed, heavy-tailed distributions. A magnitude-7 earthquake releases ~32× more energy than a magnitude-6 event, but the raw count difference is just 1. Logarithmic transformations are therefore essential for numerical stability and to make relationships approximately linear for the models.

#### 2.2.1 Log magnitude

The Gutenberg-Richter relation states that the number of earthquakes of magnitude ≥ M follows:

`log₁₀(N) = a − bM`

meaning magnitude is already a logarithmic quantity. We derive a second log-transform to linearise the relationship between magnitude and aftershock probability:

**Formula:**  
`log_magnitude = log₁₀(trigger_magnitude)`

#### 2.2.2 Magnitude-depth ratio

This interaction term captures the joint effect of large magnitude at shallow depth — the scenario most strongly associated with productive aftershock sequences. A shallow Mw 6.5 and a deep Mw 6.5 should be treated very differently; this ratio encodes that difference compactly.

**Formula:**  
`mag_depth_ratio = trigger_magnitude / (trigger_depth_km + 1)`

The `+1` prevents division by zero for surface events (`depth = 0 km`).

#### 2.2.3 Log-transformed prior event counts

The number of earthquakes globally in the 24 hours and 7 days before a trigger event is a proxy for the regional tectonic stress state. Because these counts are highly skewed (most quiet periods have ~50–80 events, but active sequences can generate thousands), we apply a `log1p` transform:

- `log_prior_24h = log(1 + prior_global_event_count_24h)`
- `log_prior_7d  = log(1 + prior_global_event_count_7d)`

`log1p` is used rather than `log10` to handle the edge case of zero prior events.

#### 2.2.4 Seismicity acceleration

This feature is directly inspired by the Omori–Utsu law (Omori, 1894; Utsu, 1961), which states that aftershock rate decays as:

`n(t) ∝ 1 / (t + c)^p`

An elevated 24-hour event count relative to the 7-day background rate signals an ongoing aftershock sequence — the single strongest contextual predictor that another aftershock will occur soon.

**Formula:**  
`seismicity_acceleration = prior_global_event_count_24h / (prior_global_event_count_7d / 7)`

The denominator is protected against division by zero by replacing 0 with `NaN`.

#### 2.2.5 Interaction terms

Two multiplicative interaction terms encode composite physical scenarios that individual features cannot capture. A gradient-boosted tree can discover interactions via sequential splits, but neural networks process features linearly at each layer and benefit from pre-computed products.

| Feature Name      | Formula / Code                                 | Notes                                           |
| ----------------- | ---------------------------------------------- | ----------------------------------------------- |
| `mag_x_log_prior` | `trigger_magnitude × log(1 + prior_count_24h)` | Large quake during elevated background activity |
| `mag_x_shallow`   | `trigger_magnitude × depth_shallow`            | Large magnitude AND shallow depth               |

---

### 2.3 Temporal Cyclical Encodings (6 features)

Month, hour, and day-of-year are periodic variables: month 12 is immediately followed by month 1, making the raw integer representation fundamentally circular. Feeding raw integers breaks continuity — the model sees December (12) and January (1) as maximally different when they are in fact adjacent.

We project each cyclic variable onto the unit circle using sine-cosine pairs:

**General formula:**  
`sin_x = sin(2π × x / period)`  
`cos_x = cos(2π × x / period)`

The two-component encoding preserves the circular distance metric so that neighbouring time steps are always close in the feature space, regardless of year-end or day-end boundaries.

| Feature Name                     | Formula / Code                         | Notes                       |
| -------------------------------- | -------------------------------------- | --------------------------- |
| `sin_month`, `cos_month`         | `sin(2π·month/12)`, `cos(2π·month/12)` | Annual seismicity cycle     |
| `sin_hour`, `cos_hour`           | `sin(2π·hour/24)`, `cos(2π·hour/24)`   | Diurnal tidal loading cycle |
| `sin_dayofyear`, `cos_dayofyear` | `sin(2π·doy/365)`, `cos(2π·doy/365)`   | Seasonal stress variation   |

These features capture phenomena such as tidal stress loading on faults, seasonal variations in hydroseismicity, and minor reporting completeness differences across seasons.

---

### 2.4 Spatial Context (1 feature)

A binary flag marks whether an earthquake falls within the circum-Pacific seismic belt — the **Ring of Fire**. The belt encompasses the world's most active subduction zones and accounts for roughly 90% of the world's largest earthquakes.

| Feature Name   | Formula / Code | Notes             |
| -------------- | -------------- | ----------------- | ---------- | ---------------- | ------- | -------------------------------- |
| `ring_of_fire` | `              | trigger_longitude | > 130° AND | trigger_latitude | <= 60°` | 1 if circum-Pacific belt, else 0 |

Events in this region occur in tectonic settings with higher background seismicity rates, denser aftershock sequences, and more reliable GCMT moment tensor coverage.

---

### 2.5 GCMT Moment Tensor Features (20 features)

This is the richest feature group and the one most specific to earthquake physics. Features are computed from the Global Centroid Moment Tensor (GCMT) solution — the most physically complete description of fault rupture available in the dataset. GCMT features are only populated for the ~80% of events with a matched GCMT solution.

**Handling GCMT missingness**  
The ~20% of events without a GCMT match have `NaN` in all GCMT columns. This missingness is structurally related to earthquake magnitude and depth. XGBoost handles `NaN` natively via learned default split directions. MLP and FT-Transformer use median imputation before training.

#### 2.5.1 Seismic moment and magnitude

The scalar seismic moment `M₀` is the physically correct measure of earthquake size. Because it spans many orders of magnitude, it must be log-transformed.

| Feature Name               | Formula / Code                | Notes                    |
| -------------------------- | ----------------------------- | ------------------------ |
| `log_scalar_moment`        | `log₁₀(gcmt_scalar_moment)`   | Log seismic moment       |
| `moment_exponent_centered` | `gcmt_moment_exponent − 24`   | Order-of-magnitude class |
| `gcmt_mw`                  | `(2/3) × log₁₀(M₀) − 10.7`    | Moment magnitude         |
| `mw_trigger_diff`          | `gcmt_mw − trigger_magnitude` | Catalogue discrepancy    |

**Moment magnitude formula (Hanks & Kanamori, 1979):**  
`Mw = (2/3) × log₁₀(M₀) − 10.7`

#### 2.5.2 Fault geometry — dip angle

| Feature Name | Formula / Code              | Notes                   |
| ------------ | --------------------------- | ----------------------- |
| `sin_dip`    | `sin(dip_angle_in_radians)` | Fault plane inclination |

#### 2.5.3 Eigenvalue-based moment tensor decomposition

**CLVD fraction**

`clvd_fraction = 2|λ₂| / (|λ₁| + |λ₃|)`

**Eigenvalue ratio**

`eig_ratio = |λ₁| / (|λ₁| + |λ₃|)`

| Feature Name    | Formula / Code | Notes |
| --------------- | -------------- | ----- | --- | --- | --- | --- | --- | ------------------------------------ |
| `clvd_fraction` | `2             | λ₂    | / ( | λ₁  | +   | λ₃  | )`  | Deviation from pure double-couple    |
| `eig_ratio`     | `              | λ₁    | / ( | λ₁  | +   | λ₃  | )`  | Tensional vs compressional asymmetry |

#### 2.5.4 Principal axis plunge angles

| Feature Name                         | Formula / Code         | Notes                      |
| ------------------------------------ | ---------------------- | -------------------------- |
| `sin_eig1_plunge`, `cos_eig1_plunge` | `sin(p₁°)`, `cos(p₁°)` | T-axis plunge              |
| `sin_eig3_plunge`, `cos_eig3_plunge` | `sin(p₃°)`, `cos(p₃°)` | P-axis plunge              |
| `tp_plunge_diff`                     | `p₁ − p₃`              | T-axis minus P-axis plunge |

#### 2.5.5 Centroid–hypocenter depth difference

| Feature Name          | Formula / Code                     | Notes                     |
| --------------------- | ---------------------------------- | ------------------------- |
| `centroid_depth_diff` | `gcmt_depth_km − trigger_depth_km` | Centroid below hypocenter |

#### 2.5.6 Rupture half-duration

| Feature Name        | Formula / Code                    | Notes                |
| ------------------- | --------------------------------- | -------------------- |
| `log_half_duration` | `log(1 + gcmt_half_duration_sec)` | Proxy for fault area |

#### 2.5.7 GCMT magnitude discrepancy

| Feature Name   | Formula / Code | Notes         |
| -------------- | -------------- | ------------- | --- | --------------------------- |
| `mag_diff_abs` | `              | gcmt_mag_diff | `   | Catalogue quality indicator |

---

### 2.6 Tectonic Regime from Rake Angle (3 features)

The GCMT rake angle `λ` encodes the direction of slip along the fault plane. Rather than feeding the raw angle, we classify each earthquake into three primary tectonic regimes.

**Why three regimes?**  
Aftershock productivity differs systematically by regime. Reverse/thrust faults typically produce the most numerous aftershock sequences, strike-slip is intermediate, and normal faults produce the fewest.

| Feature Name     | Formula / Code      | Notes                        |
| ---------------- | ------------------- | ---------------------------- | --- | ------ | --- | ------ | -------- | --------------- |
| `is_strike_slip` | `min(               | r                            | ,   | r−180° | ,   | r−360° | ) < 45°` | Horizontal slip |
| `is_reverse`     | `45° <= r <= 135°`  | Upward hanging-wall motion   |
| `is_normal`      | `225° <= r <= 315°` | Downward hanging-wall motion |

All three flags may be 0 simultaneously for oblique-slip mechanisms. All three are set to `NaN` for events without a GCMT rake solution.

---

## 3. Model-Specific Differences

The feature engineering function is identical across all four models. However, models differ in how they handle missing values and feature scaling.

| Model          | Feature Set | Missing GCMT         | Scaling          |
| -------------- | ----------- | -------------------- | ---------------- |
| XGBoost        | 91 features | `NaN` passed through | None             |
| MLP            | 91 features | Median imputation    | `StandardScaler` |
| TabNet         | 91 features | `NaN` passed through | None             |
| FT-Transformer | 83 features | Median imputation    | `StandardScaler` |

The FT-Transformer uses 83 features because its implementation was based on an earlier version of the feature set that did not yet include `eig_ratio`, `gcmt_mw`, `mw_trigger_diff`, and the four plunge angle features (`sin/cos_eig1_plunge`, `sin/cos_eig3_plunge`, `tp_plunge_diff`).

---

## 4. Complete Engineered Feature Reference

| #   | Feature                    | Group           | Formula                                      |
| --- | -------------------------- | --------------- | -------------------------------------------- | ------------- | ---------- | --- | ------- | --- | --- |
| 1   | `depth_shallow`            | Depth Regime    | `trigger_depth_km < 70`                      |
| 2   | `depth_intermediate`       | Depth Regime    | `70 <= trigger_depth_km <= 300`              |
| 3   | `depth_deep`               | Depth Regime    | `trigger_depth_km > 300`                     |
| 4   | `log_magnitude`            | Magnitude       | `log₁₀(trigger_magnitude)`                   |
| 5   | `mag_depth_ratio`          | Magnitude       | `trigger_magnitude / (trigger_depth_km + 1)` |
| 6   | `log_prior_24h`            | Seismicity      | `log(1 + prior_count_24h)`                   |
| 7   | `log_prior_7d`             | Seismicity      | `log(1 + prior_count_7d)`                    |
| 8   | `seismicity_acceleration`  | Seismicity      | `prior_24h / (prior_7d / 7)`                 |
| 9   | `mag_x_log_prior`          | Interaction     | `trigger_magnitude × log_prior_24h`          |
| 10  | `mag_x_shallow`            | Interaction     | `trigger_magnitude × depth_shallow`          |
| 11  | `sin_month`                | Temporal        | `sin(2π·month/12)`                           |
| 12  | `cos_month`                | Temporal        | `cos(2π·month/12)`                           |
| 13  | `sin_hour`                 | Temporal        | `sin(2π·hour/24)`                            |
| 14  | `cos_hour`                 | Temporal        | `cos(2π·hour/24)`                            |
| 15  | `sin_dayofyear`            | Temporal        | `sin(2π·doy/365)`                            |
| 16  | `cos_dayofyear`            | Temporal        | `cos(2π·doy/365)`                            |
| 17  | `ring_of_fire`             | Spatial         | `                                            | lon           | > 130° AND | lat | <= 60°` |
| 18  | `log_scalar_moment`        | GCMT Moment     | `log₁₀(gcmt_scalar_moment)`                  |
| 19  | `moment_exponent_centered` | GCMT Moment     | `gcmt_moment_exponent − 24`                  |
| 20  | `gcmt_mw`                  | GCMT Moment     | `(2/3)·log₁₀(M₀) − 10.7`                     |
| 21  | `mw_trigger_diff`          | GCMT Moment     | `gcmt_mw − trigger_magnitude`                |
| 22  | `sin_dip`                  | Fault Geometry  | `sin(dip_angle_radians)`                     |
| 23  | `clvd_fraction`            | Eigenvalue      | `2                                           | λ₂            | / (        | λ₁  | +       | λ₃  | )`  |
| 24  | `eig_ratio`                | Eigenvalue      | `                                            | λ₁            | / (        | λ₁  | +       | λ₃  | )`  |
| 25  | `sin_eig1_plunge`          | Principal Axis  | `sin(T-axis_plunge_radians)`                 |
| 26  | `cos_eig1_plunge`          | Principal Axis  | `cos(T-axis_plunge_radians)`                 |
| 27  | `sin_eig3_plunge`          | Principal Axis  | `sin(P-axis_plunge_radians)`                 |
| 28  | `cos_eig3_plunge`          | Principal Axis  | `cos(P-axis_plunge_radians)`                 |
| 29  | `tp_plunge_diff`           | Principal Axis  | `T-axis_plunge − P-axis_plunge`              |
| 30  | `centroid_depth_diff`      | Rupture         | `gcmt_depth_km − trigger_depth_km`           |
| 31  | `log_half_duration`        | Rupture         | `log(1 + gcmt_half_duration_sec)`            |
| 32  | `mag_diff_abs`             | Catalogue       | `                                            | gcmt_mag_diff | `          |
| 33  | `is_strike_slip`           | Tectonic Regime | `rake: ss_dist < 45°`                        |
| 34  | `is_reverse`               | Tectonic Regime | `rake: 45° <= r <= 135°`                     |
| 35  | `is_normal`                | Tectonic Regime | `rake: 225° <= r <= 315°`                    |

---

## 5. References

- Gutenberg, B. & Richter, C. F. (1944). _Frequency of earthquakes in California._ Bulletin of the Seismological Society of America, 34(4), 185–188.
- Hanks, T. C. & Kanamori, H. (1979). _A moment magnitude scale._ Journal of Geophysical Research, 84(B5), 2348–2350.
- Helmstetter, A. & Shaw, B. E. (2006). _Relation between stress heterogeneity and aftershock rate in the rate-and-state model._ Journal of Geophysical Research: Solid Earth, 111(B7).
- Omori, F. (1894). _On the aftershocks of earthquakes._ Journal of the College of Science, Imperial University of Tokyo, 7, 111–200.
- Utsu, T. (1961). _A statistical study on the occurrence of aftershocks._ Geophysical Magazine, 30, 521–605.
- Dziewonski, A. M., Chou, T.-A. & Woodhouse, J. H. (1981). _Determination of earthquake source parameters from waveform data for studies of global and regional seismicity._ Journal of Geophysical Research, 86(B4), 2825–2852.
