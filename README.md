# BCI Motor Imagery Classification

## Overview

This project implements a complete, multi-subject EEG signal processing and classification pipeline for Brain-Computer Interface (BCI) applications using the **BCI Competition IV Dataset 2a**. The pipeline covers all **9 subjects (A01–A09)** across three sequential stages, each with a baseline and an improved iteration:

| Stage | Notebook | Module |
|-------|----------|--------|
| 1. Preprocessing | `preprocessing.ipynb` | `src/preprocessing.py` |
| 2. Feature Extraction | `feature_extraction.ipynb` | `src/features.py` |
| 3a. Baseline Classification | `training_baseline.ipynb` | `src/models.py` |
| 3b. Improved Classification | `training_improved.ipynb` | `src/models.py` |

---

## Dataset

- **Source:** BCI Competition IV Dataset 2a — https://www.bbci.de/competition/iv/
- **Format:** `.gdf` (General Data Format for biosignals)
- **Subjects:** A01–A09, Training (`T`) and Evaluation (`E`) sessions

### Recording Specifications

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 250 Hz |
| EEG Channels | 22 |
| EOG Channels | 3 (EOG-left, EOG-central, EOG-right) |
| Total Channels | 25 |
| Hardware Bandpass | 0.5 Hz – 100 Hz |

### Motor Imagery Classes

| Class | Event Code | Trials / Subject |
|-------|-----------|-----------------|
| Left Hand | 769 | 72 |
| Right Hand | 770 | 72 |
| Both Feet | 771 | 72 |
| Tongue | 772 | 72 |

288 balanced trials per subject across 9 experimental runs. **Chance level = 25%.**

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                              # 18 .gdf files — T + E sessions for A01–A09
│   ├── processed/                        # A01T–A09T_clean_epo.fif (preprocessed epochs)
│   └── features/                         # A01T–A09T_features.npz (CSP feature arrays)
│
├── notebooks/
│   ├── bci.ipynb                         # Original single-subject EDA (A01T)
│   ├── preprocessing.ipynb               # Multi-subject preprocessing
│   ├── feature_extraction.ipynb          # Multi-subject CSP feature extraction
│   ├── training_baseline.ipynb           # Baseline classifiers (SVM, LDA, RF)
│   └── training_improved.ipynb           # Improved classifiers (tuned SVM, LDA, RF + Ensemble)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Preprocessing pipeline module
│   ├── features.py                       # CSP feature extraction module
│   └── models.py                         # Classifier training & evaluation module
│
├── results/
│   ├── figures/
│   │   ├── preprocessing/                # 55 plots — 6 per subject + 1 aggregate
│   │   ├── features/                     # 64 plots — 7 per subject + 1 aggregate
│   │   └── training/
│   │       ├── baseline/                 # 9 aggregate plots
│   │       └── improved/                 # 3 aggregate comparison plots
│   ├── metrics/
│   │   ├── baseline/                     # accuracy_summary.csv, per_class_metrics.csv, per_fold_scores.csv
│   │   └── improved/                     # improved_accuracy_summary.csv, baseline_vs_improved.csv
│   └── models/
│       ├── baseline/                     # 27 .pkl files — SVM, LDA, RF × 9 subjects
│       └── improved/                     # 36 .pkl files — SVM_tuned, LDA_improved, RF_improved, Ensemble × 9 subjects
│
├── bci_script.md
├── parse_nb.py
├── requirements.txt
└── README.md
```

---

## Installation

**Python:** 3.13

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| `mne` | 1.11.0 | EEG I/O, ICA, CSP, PSD |
| `numpy` | 2.4.2 | Numerical computing |
| `scipy` | 1.17.1 | Signal processing |
| `scikit-learn` | 1.8.0 | CSP, classifiers, cross-validation |
| `pandas` | 3.0.1 | Metrics CSVs |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Heatmaps and correlation plots |
| `ipykernel` | 7.2.0 | Jupyter support |

---

## Pipeline

### Stage 1 — Preprocessing (`src/preprocessing.py`)

Converts raw `.gdf` files into clean, epoched `.fif` files ready for feature extraction.

| Step | Detail |
|------|--------|
| **Load** | `mne.io.read_raw_gdf()` with `EOG-left`, `EOG-central`, `EOG-right` declared |
| **Bandpass filter** | 7–30 Hz FIR — isolates Mu (8–13 Hz) and Beta (13–30 Hz) bands |
| **Montage** | GDF channel labels → standard 10-20 names (required for ICA and topographic plots) |
| **ICA** | 20 components; all 3 EOG channels used for artifact detection. Catches blink artifacts missed by single-channel detection |
| **Epoch** | −0.5s to +4.5s per MI cue; baseline corrected to −0.5s–0s |
| **Amplitude rejection** | Manual 100 µV threshold — MNE's built-in `reject` parameter fails with this dataset's GDF amplitude scaling |
| **Save** | `data/processed/{id}_clean_epo.fif` |

**Output figures** → `results/figures/preprocessing/`:
`_amplitude_histogram`, `_trial_distribution`, `_correlation`, `_epoch_waveforms`, `_topomaps`, `_psd`

---

### Stage 2 — Feature Extraction (`src/features.py`)

Extracts Mu + Beta band CSP features from clean epochs.

| Step | Detail |
|------|--------|
| **Crop** | Active MI window: 0.5s → 3.5s (removes cue response artefact and post-imagery period) |
| **Band-split** | Separate FIR filtering into Mu (8–13 Hz) and Beta (13–30 Hz) |
| **NaN/Inf check** | Corrupt trials removed before CSP fitting |
| **CSP — Mu** | `n_components=8`, `reg=0.05` (auto-raised to 0.2 on NaN output) |
| **CSP — Beta** | `n_components=8`, `reg=0.05` (auto-raised to 0.2 on NaN output) |
| **Concatenate** | Mu + Beta features → **16 features per trial** |
| **Normalize** | `StandardScaler` (zero mean, unit variance) |
| **Save** | `data/features/{id}_features.npz` — `X`: (n_trials × 16), `y`: (n_trials,) |

**Output figures** → `results/figures/features/`:
`_csp_patterns`, `_csp_filters`, `_csp_feature_distribution`, `_csp_scatter`, `_csp_boxplot`, `_csp_feature_correlation`, `_csp_mean_per_class`

---

### Stage 3a — Baseline Classification (`training_baseline.ipynb`)

Default hyperparameters, no ensembling.

| Model | Configuration |
|-------|--------------|
| **SVM** | `kernel='rbf'`, `class_weight='balanced'`, `probability=True` |
| **LDA** | `solver='lsqr'`, `shrinkage='auto'` |
| **RF** | `n_estimators=200`, `class_weight='balanced'` |

Evaluation: **5-fold stratified cross-validation**

**Output:**
- Models → `results/models/baseline/{id}_{SVM|LDA|RF}.pkl`
- Metrics → `results/metrics/baseline/`
  - `accuracy_summary.csv` — mean ± std per subject
  - `per_class_metrics.csv` — precision, recall, F1 per class per subject
  - `per_fold_scores.csv` — accuracy at each fold
- Figures → `results/figures/training/baseline/`
  - `accuracy_bars.png`, `confusion_matrices_{SVM|LDA|RF}.png`, `f1_heatmap.png`, `per_fold_scores.png`, `model_comparison_boxplot.png`, `recall_heatmap.png`, `best_subject_detail_A03T.png`

---

### Stage 3b — Improved Classification (`training_improved.ipynb`)

Hyperparameter-tuned models with a soft-voting Ensemble.

| Model | Improvements over Baseline |
|-------|---------------------------|
| **SVM_tuned** | Grid-searched `C` and `gamma` per subject |
| **LDA_improved** | Same architecture; benefits from tuned feature preprocessing |
| **RF_improved** | Grid-searched `n_estimators` and `max_depth` per subject |
| **Ensemble** | Soft-voting over SVM_tuned + LDA_improved + RF_improved |

**Output:**
- Models → `results/models/improved/{id}_{SVM_tuned|LDA_improved|RF_improved|Ensemble}.pkl`
- Metrics → `results/metrics/improved/`
  - `improved_accuracy_summary.csv` — mean ± std per subject per improved model
  - `baseline_vs_improved.csv` — per-subject SVM baseline vs tuned comparison
- Figures → `results/figures/training/improved/`
  - `baseline_vs_improved_comparison.png`
  - `confusion_matrices_ensemble_improved.png`
  - `f1_heatmap_improved.png`

---

## Results

### Baseline — 5-fold CV Accuracy (%)

| Subject | SVM | LDA | RF |
|---------|-----|-----|----|
| A01T | 84.19 ± 4.37 | 83.54 ± 6.22 | 81.12 ± 3.44 |
| A02T | 58.81 ± 3.27 | 60.19 ± 2.61 | 58.82 ± 5.97 |
| A03T | 87.71 ± 3.75 | 86.02 ± 5.03 | 87.68 ± 2.98 |
| A04T | 55.36 ± 2.34 | 57.65 ± 5.88 | 54.22 ± 5.92 |
| A05T | 54.31 ± 4.22 | 55.14 ± 1.06 | 57.48 ± 5.85 |
| A06T | 53.57 ± 3.68 | 55.68 ± 3.11 | 49.67 ± 4.21 |
| A07T | 82.16 ± 4.12 | 81.45 ± 8.95 | 76.60 ± 11.48 |
| A08T | 87.16 ± 4.91 | 87.58 ± 5.18 | 83.83 ± 4.00 |
| A09T | 62.59 ± 5.58 | 64.96 ± 8.48 | 62.03 ± 5.80 |
| **Avg** | **69.5** | **70.2** | **67.9** |

---

### Improved — 5-fold CV Accuracy (%)

| Subject | SVM_tuned | LDA_improved | RF_improved | Ensemble |
|---------|-----------|--------------|-------------|----------|
| A01T | 89.8 ± 3.6 | 83.6 ± 4.9 | 87.3 ± 4.0 | 88.9 ± 3.1 |
| A02T | 63.4 ± 6.2 | 61.6 ± 6.0 | 61.2 ± 6.9 | 65.2 ± 5.2 |
| A03T | **91.7 ± 2.5** | 87.7 ± 4.9 | **91.7 ± 1.2** | 91.2 ± 3.0 |
| A04T | 58.1 ± 6.9 | 57.7 ± 4.3 | 57.0 ± 3.2 | 55.9 ± 3.1 |
| A05T | 70.3 ± 5.2 | 58.9 ± 3.6 | 69.1 ± 9.6 | 69.6 ± 7.6 |
| A06T | 54.5 ± 5.7 | 54.1 ± 6.5 | 52.1 ± 2.4 | 55.2 ± 4.2 |
| A07T | 84.0 ± 8.1 | 84.0 ± 5.4 | 75.0 ± 12.6 | 82.7 ± 5.9 |
| A08T | **90.2 ± 2.9** | 88.8 ± 4.2 | 87.3 ± 3.2 | 89.9 ± 3.7 |
| A09T | 64.6 ± 6.7 | 66.1 ± 5.3 | 66.6 ± 3.7 | 68.2 ± 6.1 |
| **Avg** | **74.1** | **71.4** | **71.9** | **74.1** |

---

### Baseline SVM vs Tuned SVM — Per-Subject Gains

| Subject | Baseline SVM | Tuned SVM | Δ |
|---------|-------------|-----------|---|
| A01T | 84.2% | 89.8% | **+5.6%** |
| A02T | 58.8% | 63.4% | **+4.6%** |
| A03T | 87.7% | 91.7% | **+4.0%** |
| A04T | 55.4% | 58.1% | **+2.7%** |
| A05T | 54.3% | 70.3% | **+16.0%** ⬆ biggest gain |
| A06T | 53.6% | 54.5% | +0.9% |
| A07T | 82.2% | 84.0% | +1.9% |
| A08T | 87.2% | 90.2% | **+3.1%** |
| A09T | 62.6% | 64.6% | +2.0% |
| **Avg** | **69.5%** | **74.1%** | **+4.6%** |

---

## Analysis

### Subject Variability
The dataset shows large cross-subject variance (53%–92%). This is typical for motor imagery BCI — subjects differ substantially in how clearly their Mu/Beta rhythms reflect motor imagery. High performers (A01T, A03T, A07T, A08T) consistently score >80% across all models. Low performers (A04T, A05T, A06T) likely have weaker or noisier ERD patterns, and may benefit from subject-specific preprocessing tuning.

### Model Comparison
- **SVM_tuned** is the strongest individual model overall (avg 74.1%), benefiting most from per-subject `C`/`gamma` search. Particularly large gains on A05T (+16%) suggest the default RBF kernel was poorly calibrated for that subject.
- **Ensemble** matches SVM_tuned on average (74.1%) with generally lower variance — useful for deployment where consistency matters.
- **LDA_improved** (avg 71.4%) is competitive with much lower complexity, making it a good first-choice baseline.
- **RF_improved** shows the highest within-subject fold variance (A07T: ±12.6%), suggesting sensitivity to small training set sizes after trial rejection.

### Low-Performing Subjects
A04T (58%), A05T (54–70%), A06T (54%) see limited improvement from tuning alone. The underlying cause is likely higher artifact rates at preprocessing time leading to fewer and noisier clean trials, not model capacity.

---

## Key EDA Findings

### Scalp Topomaps — Spatial Separability (A01T)
All four classes show distinct 8–30 Hz power distributions, confirming CSP effectiveness:
- **Left Hand:** Right-hemisphere desynchronization (C4 area) — contralateral ERD
- **Right Hand:** Left-hemisphere desynchronization (C3 area) — lateralized ERD
- **Feet:** Central desynchronization at Cz — midline motor cortex
- **Tongue:** Highest overall power, unique lateral distribution

### Channel Correlation (A01T)
Three anatomical clusters in clean epochs:
- **Frontal** (Fz, FC3–FC4): 0.84–0.97 within-cluster correlation
- **Central** (C5–C6 strip): 0.75–0.95; C3–C4 = 0.63 (opposing hemispheric roles)
- **Parietal** (CP3–POz): 0.85–0.95; Front-to-back (Fz→POz) = 0.07

---

## Limitations

- **Trial loss:** ~43% average per subject from amplitude rejection — Feet and Tongue classes most affected due to EMG contamination.
- **Class imbalance post-rejection:** Up to 1:2.3 class ratio (Tongue vs Right Hand for A01T) — all models use `class_weight='balanced'`.
- **Low-performer ceiling:** A04T, A05T, A06T remain below 60% (SVM baseline) even after tuning — subject-specific preprocessing may be required.
- **Training sessions only:** All 9 evaluation sessions (`A0xE.gdf`) are in `data/raw/` but not yet preprocessed or evaluated.
- **`improved/` figures:** Only 3 aggregate comparison plots — per-subject improved plots not yet generated.

---

## Next Steps

1. **Subject-specific preprocessing** — Adaptive rejection thresholds, Common Average Reference (CAR), per-subject ICA tuning for low performers
2. **Deep learning** — EEGNet on raw epochs (bypasses hand-crafted CSP features)
3. **Cross-subject generalization** — Train on 8 subjects, evaluate on held-out 9th
4. **Evaluation sessions** — Process `A0xE.gdf` files for proper held-out test evaluation
5. **Per-subject improved figures** — Generate full diagnostic plots for the improved pipeline per subject

---

## References

- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.