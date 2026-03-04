# BCI Motor Imagery Classification

## Overview

This project implements a complete, multi-subject EEG signal processing and classification pipeline for Brain-Computer Interface (BCI) applications using the **BCI Competition IV Dataset 2a** across all 9 subjects (A01–A09). Two independent feature extraction and classification approaches are compared:

- **CSP Pipeline** — Common Spatial Patterns (Mu + Beta band) → SVM / LDA / RF (baseline + tuned)
- **Riemannian Pipeline** — OAS covariance matrices on tangent space → SVM / LDA / RF

| Stage | Notebook | Module |
|-------|----------|--------|
| Preprocessing | `preprocessing.ipynb` | `src/preprocessing.py` |
| CSP Feature Extraction | `feature_extraction.ipynb` | `src/features.py` |
| Riemannian Feature Extraction | `features_riemannian.ipynb` | `src/riemannian.py` |
| Baseline Classification | `training_baseline.ipynb` | `src/models.py` |
| Improved Classification | `training_improved.ipynb` | `src/models.py` |
| Riemannian Classification | `training_riemann.ipynb` | `src/models.py` + `src/riemannian.py` |

---

## Dataset

- **Source:** BCI Competition IV Dataset 2a — https://www.bbci.de/competition/iv/
- **Format:** `.gdf` | **Subjects:** A01–A09, Training (`T`) + Evaluation (`E`) sessions

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 250 Hz |
| EEG Channels | 22 |
| EOG Channels | 3 (EOG-left, EOG-central, EOG-right) |
| Total Channels | 25 |
| Hardware Bandpass | 0.5 – 100 Hz |
| Classes | Left Hand (769), Right Hand (770), Feet (771), Tongue (772) |
| Trials / subject | 288 (72 per class) — **Chance = 25%** |

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                              # 18 .gdf files (T + E for A01–A09)
│   ├── processed/                        # A01T–A09T_clean_epo.fif
│   ├── features/                         # A01T–A09T_features.npz  (CSP, 16-dim)
│   └── features_riemannian/              # A01T–A09T_riemannian.npz (tangent space)
│
├── notebooks/
│   ├── bci.ipynb                         # Original single-subject EDA (A01T)
│   ├── preprocessing.ipynb               # Multi-subject preprocessing
│   ├── feature_extraction.ipynb          # CSP feature extraction (A01–A09)
│   ├── features_riemannian.ipynb         # Riemannian feature extraction (A01–A09)
│   ├── training_baseline.ipynb           # Baseline CSP classifiers (SVM, LDA, RF)
│   ├── training_improved.ipynb           # Tuned CSP classifiers + Ensemble
│   └── training_riemann.ipynb            # Riemannian classifiers (SVM, LDA, RF)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Preprocessing module
│   ├── features.py                       # CSP feature extraction module
│   ├── riemannian.py                     # Riemannian feature extraction module
│   └── models.py                         # Classifier training & evaluation module
│
├── results/
│   ├── figures/
│   │   ├── preprocessing/                # 55 plots — 6 per subject + 1 aggregate
│   │   ├── features/
│   │   │   ├── csp/                      # 64 plots — 7 per subject + 1 aggregate
│   │   │   └── riemannian/               # 28 plots — 3 per subject + 1 aggregate
│   │   └── training/
│   │       ├── baseline/                 # 9 aggregate CSP-baseline plots
│   │       ├── improved/                 # 3 aggregate CSP-improved plots
│   │       └── riemannian/               # 13 aggregate Riemannian + comparison plots
│   ├── metrics/
│   │   ├── baseline/                     # accuracy_summary.csv, per_class_metrics.csv, per_fold_scores.csv
│   │   ├── improved/                     # improved_accuracy_summary.csv, baseline_vs_improved.csv
│   │   └── riemannian/                   # accuracy_summary.csv, riemannian_accuracy_summary.csv,
│   │                                     # per_class_metrics.csv, per_fold_scores.csv
│   └── models/
│       ├── baseline/                     # 27 .pkl — {SVM, LDA, RF} × 9 subjects
│       ├── improved/                     # 36 .pkl — {SVM_tuned, LDA_improved, RF_improved, Ensemble} × 9
│       └── riemannian/                   # 27 .pkl — {Riemannian_SVM, Riemannian_LDA, Riemannian_RF} × 9
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
| `pyriemann` | — | Covariance estimation, tangent space |
| `pandas` | 3.0.1 | Metrics CSVs |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Heatmaps |
| `ipykernel` | 7.2.0 | Jupyter support |

---

## Pipeline

### Stage 1 — Preprocessing (`src/preprocessing.py`)

| Step | Detail |
|------|--------|
| **Load** | `mne.io.read_raw_gdf()` with all 3 EOG channels declared |
| **Bandpass filter** | 7–30 Hz FIR — isolates Mu (8–13 Hz) and Beta (13–30 Hz) |
| **Montage** | GDF channel labels → standard 10-20 names |
| **ICA** | 20 components; all 3 EOG channels used for artifact detection |
| **Epoch** | −0.5s to +4.5s, baseline corrected to −0.5s–0s |
| **Amplitude rejection** | Manual 100 µV threshold (MNE built-in `reject` param fails with this dataset's GDF scaling) |
| **Save** | `data/processed/{id}_clean_epo.fif` |

**Output figures** → `results/figures/preprocessing/`:
`_amplitude_histogram`, `_trial_distribution`, `_correlation`, `_epoch_waveforms`, `_topomaps`, `_psd`

---

### Stage 2a — CSP Feature Extraction (`src/features.py`)

| Step | Detail |
|------|--------|
| **Crop** | Active MI window: 0.5s → 3.5s |
| **Band-split** | Separate FIR filtering into Mu (8–13 Hz) and Beta (13–30 Hz) |
| **CSP — Mu** | 8 components, `reg=0.05` (auto-raised to 0.2 on NaN) |
| **CSP — Beta** | 8 components, `reg=0.05` (auto-raised to 0.2 on NaN) |
| **Concatenate** | Mu + Beta → **16 features per trial** |
| **Normalize** | `StandardScaler` |
| **Save** | `data/features/{id}_features.npz` — `X`: (n_trials × 16), `y`: (n_trials,) |

**Output figures** → `results/figures/features/csp/`:
`_csp_patterns`, `_csp_filters`, `_csp_feature_distribution`, `_csp_scatter`, `_csp_boxplot`, `_csp_feature_correlation`, `_csp_mean_per_class`

---

### Stage 2b — Riemannian Feature Extraction (`src/riemannian.py`)

| Step | Detail |
|------|--------|
| **Crop** | Active MI window: 0.5s → 3.5s |
| **Covariances** | OAS-regularized covariance matrices per trial (22 EEG channels) — shape: (n_trials, 22, 22) |
| **Tangent space** | Project covariances onto Riemannian tangent space at the geometric mean — shape: (n_trials, 253) |
| **Normalize** | `StandardScaler` |
| **Save** | `data/features_riemannian/{id}_riemannian.npz` — `X`: (n_trials × 253), `y`: (n_trials,) |

**Output figures** → `results/figures/features/riemannian/`:
`_riemannian_scatter` (PCA 2D projection), `_mean_covariance` (per-class mean covariance heatmap), `_feature_distribution`

---

### Stage 3 — Classification (`src/models.py`)

All pipelines use **5-fold stratified cross-validation**.

#### Baseline CSP (`training_baseline.ipynb`)
| Model | Config |
|-------|--------|
| SVM | `kernel='rbf'`, `class_weight='balanced'`, `probability=True` |
| LDA | `solver='lsqr'`, `shrinkage='auto'` |
| RF | `n_estimators=200`, `class_weight='balanced'` |
- Models → `results/models/baseline/` | Metrics → `results/metrics/baseline/`

#### Improved CSP (`training_improved.ipynb`)
| Model | Config |
|-------|--------|
| SVM_tuned | Grid-searched `C` and `gamma` per subject |
| LDA_improved | Same architecture, tuned feature preprocessing |
| RF_improved | Grid-searched `n_estimators` and `max_depth` |
| Ensemble | Soft-voting over SVM_tuned + LDA_improved + RF_improved |
- Models → `results/models/improved/` | Metrics → `results/metrics/improved/`

#### Riemannian (`training_riemann.ipynb`)
| Model | Config |
|-------|--------|
| Riemannian_SVM | `kernel='rbf'`, `class_weight='balanced'` on tangent space features |
| Riemannian_LDA | `solver='lsqr'`, `shrinkage='auto'` on tangent space features |
| Riemannian_RF | `n_estimators=200`, `class_weight='balanced'` on tangent space features |
- Models → `results/models/riemannian/` | Metrics → `results/metrics/riemannian/`

---

## Results

### CSP Baseline — 5-fold CV Accuracy (%)

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

### CSP Improved — 5-fold CV Accuracy (%) + Baseline SVM vs Tuned SVM

| Subject | SVM_tuned | LDA_improved | RF_improved | Ensemble | Δ (vs Baseline SVM) |
|---------|-----------|--------------|-------------|----------|----------------------|
| A01T | 89.8 ± 3.6 | 83.6 ± 4.9 | 87.3 ± 4.0 | 88.9 ± 3.1 | +5.6% |
| A02T | 63.4 ± 6.2 | 61.6 ± 6.0 | 61.2 ± 6.9 | 65.2 ± 5.2 | +4.6% |
| A03T | **91.7 ± 2.5** | 87.7 ± 4.9 | **91.7 ± 1.2** | 91.2 ± 3.0 | +4.0% |
| A04T | 58.1 ± 6.9 | 57.7 ± 4.3 | 57.0 ± 3.2 | 55.9 ± 3.1 | +2.7% |
| A05T | 70.3 ± 5.2 | 58.9 ± 3.6 | 69.1 ± 9.6 | 69.6 ± 7.6 | **+16.0%** ⬆ |
| A06T | 54.5 ± 5.7 | 54.1 ± 6.5 | 52.1 ± 2.4 | 55.2 ± 4.2 | +0.9% |
| A07T | 84.0 ± 8.1 | 84.0 ± 5.4 | 75.0 ± 12.6 | 82.7 ± 5.9 | +1.9% |
| A08T | **90.2 ± 2.9** | 88.8 ± 4.2 | 87.3 ± 3.2 | 89.9 ± 3.7 | +3.1% |
| A09T | 64.6 ± 6.7 | 66.1 ± 5.3 | 66.6 ± 3.7 | 68.2 ± 6.1 | +2.0% |
| **Avg** | **74.1** | **71.4** | **71.9** | **74.1** | **+4.6%** |

---

### Riemannian — 5-fold CV Accuracy (%)

| Subject | SVM | LDA | RF |
|---------|-----|-----|----|
| A01T | 75.64 ± 3.11 | 78.73 ± 6.81 | 78.05 ± 3.52 |
| A02T | 55.07 ± 8.61 | 59.71 ± 3.23 | 56.49 ± 4.86 |
| A03T | 83.21 ± 6.45 | **88.25 ± 5.42** | 81.00 ± 4.46 |
| A04T | 58.42 ± 6.55 | 62.62 ± 6.78 | 53.48 ± 9.54 |
| A05T | 40.95 ± 9.62 | 41.05 ± 12.79 | 38.58 ± 4.57 |
| A06T | 51.03 ± 5.96 | 53.51 ± 5.01 | 47.84 ± 4.07 |
| A07T | 76.69 ± 6.00 | 72.57 ± 10.96 | 74.69 ± 8.77 |
| A08T | 88.38 ± 3.40 | **90.45 ± 4.09** | 81.72 ± 3.67 |
| A09T | 68.45 ± 9.60 | 69.61 ± 5.91 | 69.60 ± 6.82 |
| **Avg** | **66.4** | **68.5** | **64.6** |

---

### Three-Way Comparison — Best Model per Pipeline

| Subject | CSP Baseline (SVM) | CSP Tuned (SVM) | Riemannian (LDA) | Best |
|---------|-------------------|-----------------|------------------|------|
| A01T | 84.2% | 89.8% | 78.7% | CSP Tuned |
| A02T | 58.8% | 63.4% | 59.7% | CSP Tuned |
| A03T | 87.7% | 91.7% | 88.3% | CSP Tuned |
| A04T | 55.4% | 58.1% | 62.6% | **Riemannian** |
| A05T | 54.3% | 70.3% | 41.1% | CSP Tuned |
| A06T | 53.6% | 54.5% | 53.5% | CSP Tuned |
| A07T | 82.2% | 84.0% | 72.6% | CSP Tuned |
| A08T | 87.2% | 90.2% | 90.5% | **Riemannian** |
| A09T | 62.6% | 64.6% | 69.6% | **Riemannian** |
| **Avg** | **69.5%** | **74.1%** | **68.5%** | CSP Tuned overall |

**Cross-subject comparison figures** → `results/figures/training/riemannian/`:
`csp_vs_riemannian.png`, `three_way_comparison.png`, `confusion_matrices_riemannian_svm.png`, `f1_heatmap_riemannian.png`, and more.

---

## Analysis

### CSP Pipeline
The CSP approach explicitly targets motor imagery-relevant frequency bands (Mu + Beta) and learns spatial filters that maximise variance differences between classes. **Tuning `C`/`gamma` per subject gave an average +4.6% gain**, with A05T seeing the largest improvement (+16%) — indicating that the default RBF kernel was poorly scaled for that subject's feature distribution.

### Riemannian Pipeline
Riemannian geometry operates directly on full-band covariance matrices, capturing richer channel-interaction structure without manual band selection. **LDA is the strongest Riemannian model** (avg 68.5%) — tangent space projections produce near-Gaussian distributions well-suited to linear classifiers. Riemannian geometry wins outright for A04T, A08T, and A09T, suggesting these subjects' discriminative information is better captured in the covariance structure than in band-specific spatial patterns.

### A05T — Notable Outlier
Riemannian accuracy (40–41%) falls **below chance** for A05T, despite CSP_tuned reaching 70.3%. This subject likely has very few clean trials after rejection (~30 trials in the smallest class), making the 22×22 full covariance estimate highly unreliable. OAS regularization helps but cannot fully overcome extremely small sample sizes.

### Subject Variability
Accuracy ranges from 40% to 92% across subjects and approaches. High performers (A01T, A03T, A07T, A08T) are consistent across pipelines. Low performers (A04T, A05T, A06T) reflect noisier ERD patterns and higher trial dropout — subject-specific preprocessing tuning is likely necessary before classification improvements plateau.

---

## Limitations

- **Trial loss:** ~43% average dropout from amplitude rejection — Feet and Tongue most affected.
- **A05T Riemannian failure:** Near- or below-chance Riemannian accuracy likely due to insufficient clean trials for reliable full covariance estimation.
- **Evaluation sessions:** All `A0xE.gdf` files are in `data/raw/` but not yet preprocessed or evaluated.
- **No cross-subject model:** All models are trained and evaluated per-subject.
- **CSP data path:** CSP `.npz` files remain in `data/features/` (flat); the `csp/` subfolder restructuring applies only to figures (`results/figures/features/csp/`).

---

## Next Steps

1. **Cross-subject generalization** — Train on 8 subjects, evaluate on held-out 9th (subject-independent BCI)
2. **Evaluation sessions** — Process `A0xE.gdf` for held-out test evaluation
3. **Deep learning** — EEGNet / ShallowConvNet on raw epochs, bypassing hand-crafted features
4. **Riemannian + CSP fusion** — Concatenate tangent space and CSP features for a combined representation
5. **Subject-adaptive preprocessing** — Per-subject rejection thresholds and additional ICA tuning for low performers

---

## References

- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
- Barachant, A., et al. (2012). Multiclass Brain–Computer Interface Classification by Riemannian Geometry. *IEEE Transactions on Biomedical Engineering*, 59(4), 920–928.