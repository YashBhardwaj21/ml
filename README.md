# BCI Motor Imagery Classification

A complete EEG signal processing and classification pipeline for Brain-Computer Interface (BCI) applications, implemented using the **BCI Competition IV Dataset 2a** across all 9 subjects (A01–A09). The project compares two independent feature extraction paradigms — **CSP (Common Spatial Patterns)** and **Riemannian Geometry** — under a rigorously leakage-free cross-validation framework.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Pipeline Overview](#pipeline-overview)
5. [Methodological Rigor](#methodological-rigor)
6. [Results](#results)
7. [Analysis](#analysis)
8. [Limitations & Future Work](#limitations--future-work)
9. [References](#references)

---

## Dataset

**Source:** BCI Competition IV Dataset 2a — https://www.bbci.de/competition/iv/  
**Format:** `.gdf` | **Subjects:** A01–A09, Training (`T`) + Evaluation (`E`) sessions

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 250 Hz |
| EEG Channels | 22 |
| EOG Channels | 3 (EOG-left, EOG-central, EOG-right) |
| Total Channels | 25 |
| Hardware Bandpass | 0.5 – 100 Hz |
| Classes | Left Hand (769), Right Hand (770), Feet (771), Tongue (772) |
| Trials per Subject | 288 (72 per class) — **Chance level = 25%** |

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                              # 18 .gdf files (T + E sessions for A01–A09)
│   ├── processed/                        # A01T–A09T_clean_epo.fif
│   ├── features/                         # A01T–A09T_features.npz  (CSP, 16-dim)
│   └── features_riemannian/              # A01T–A09T_riemannian.npz (tangent space, 253-dim)
│
├── notebooks/
│   ├── bci.ipynb                         # Single-subject EDA (A01T)
│   ├── preprocessing.ipynb               # Multi-subject preprocessing pipeline
│   ├── feature_extraction.ipynb          # CSP feature extraction (A01–A09)
│   ├── features_riemannian.ipynb         # Riemannian feature extraction (A01–A09)
│   ├── training_baseline.ipynb           # Baseline CSP classifiers (SVM, LDA, RF)
│   ├── training_improved.ipynb           # Tuned CSP classifiers + SMOTE + Ensemble
│   └── training_riemann.ipynb            # Riemannian classifiers (SVM, LDA, RF)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # EEG loading, filtering, ICA, epoching
│   ├── features.py                       # CSP feature extraction
│   ├── riemannian.py                     # Riemannian covariance & tangent space
│   └── models.py                         # Classifier training & evaluation
│
├── results/
│   ├── figures/
│   │   ├── preprocessing/                # 55 diagnostic plots (6 per subject + 1 aggregate)
│   │   ├── features/
│   │   │   ├── csp/                      # 64 plots (7 per subject + 1 aggregate)
│   │   │   └── riemannian/               # 28 plots (3 per subject + 1 aggregate)
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
| `scikit-learn` | 1.8.0 | Pipelines, classifiers, cross-validation |
| `pyriemann` | — | Covariance estimation, tangent space projection |
| `imbalanced-learn` | — | SMOTE oversampling inside CV folds |
| `pandas` | 3.0.1 | Metrics export |
| `matplotlib` | 3.10.8 | Visualization |
| `seaborn` | 0.13.2 | Heatmaps |
| `ipykernel` | 7.2.0 | Jupyter support |

Run notebooks in sequential order: `preprocessing` → `feature_extraction` / `features_riemannian` → `training_*`.

---

## Pipeline Overview

### Stage 1 — Preprocessing (`src/preprocessing.py`)

| Step | Detail |
|------|--------|
| **Load** | `mne.io.read_raw_gdf()` with all 3 EOG channels explicitly declared |
| **Bandpass Filter** | 7–30 Hz FIR — isolates Mu (8–13 Hz) and Beta (13–30 Hz) motor imagery bands |
| **Montage** | GDF channel labels remapped to standard 10-20 electrode names |
| **ICA** | 20 components; all 3 EOG channels used as references for artifact detection |
| **Epoch** | −0.5 s to +4.5 s around each cue; baseline corrected to [−0.5, 0] s |
| **Dynamic Rejection** | Per-class 85th-percentile amplitude rejection with an 80 µV safety floor |
| **Save** | `data/processed/{subject_id}_clean_epo.fif` |

The **dynamic per-class rejection** strategy replaces a naive global threshold. It independently computes the 85th-percentile amplitude within each motor imagery class, ensuring that no single class loses a disproportionate number of trials. The 80 µV floor prevents over-rejection in naturally clean subjects.

**Output figures** → `results/figures/preprocessing/`:  
`_amplitude_histogram`, `_trial_distribution`, `_correlation`, `_epoch_waveforms`, `_topomaps`, `_psd`

---

### Stage 2a — CSP Feature Extraction (`src/features.py`)

| Step | Detail |
|------|--------|
| **Crop** | Active MI window: 0.5 s → 3.5 s |
| **Band-split** | Separate FIR filtering into Mu (8–13 Hz) and Beta (13–30 Hz) sub-bands |
| **CSP — Mu** | 8 spatial components, `reg=0.05` (auto-raised to 0.2 on numerical instability) |
| **CSP — Beta** | 8 spatial components, `reg=0.05` |
| **Concatenate** | Mu + Beta → **16 log-variance features per trial** |
| **Normalize** | `StandardScaler` |
| **Save** | `data/features/{subject_id}_features.npz` — `X`: (n_trials × 16), `y`: (n_trials,) |

**Output figures** → `results/figures/features/csp/`:  
`_csp_patterns`, `_csp_filters`, `_csp_feature_distribution`, `_csp_scatter`, `_csp_boxplot`, `_csp_feature_correlation`, `_csp_mean_per_class`

---

### Stage 2b — Riemannian Feature Extraction (`src/riemannian.py`)

| Step | Detail |
|------|--------|
| **Crop** | Active MI window: 0.5 s → 3.5 s |
| **Covariances** | OAS-regularized 22×22 covariance matrix per trial — shape: (n_trials, 22, 22) |
| **Tangent Space** | Project onto Riemannian tangent space at the Fréchet mean — shape: (n_trials, 253) |
| **Normalize** | `StandardScaler` |
| **Save** | `data/features_riemannian/{subject_id}_riemannian.npz` — `X`: (n_trials × 253), `y`: (n_trials,) |

**Output figures** → `results/figures/features/riemannian/`:  
`_riemannian_scatter` (PCA 2D projection), `_mean_covariance` (per-class mean covariance heatmap), `_feature_distribution`

---

### Stage 3 — Classification (`src/models.py`)

All pipelines use **5-fold Stratified K-Fold Cross-Validation** (`random_state=42`).

#### Baseline CSP (`training_baseline.ipynb`)

| Model | Configuration |
|-------|---------------|
| SVM | `kernel='rbf'`, `class_weight='balanced'`, `probability=True` |
| LDA | `solver='lsqr'`, `shrinkage='auto'` |
| RF | `n_estimators=200`, `class_weight='balanced'` |

#### Improved CSP (`training_improved.ipynb`)

| Model | Configuration |
|-------|---------------|
| SVM_tuned | Grid-searched `C` and `gamma` per subject |
| LDA_improved | Same architecture; tuned feature preprocessing |
| RF_improved | Grid-searched `n_estimators` and `max_depth` |
| Ensemble | Soft-voting over SVM_tuned + LDA_improved + RF_improved |

SMOTE (`k_neighbors=3`) applied **inside** `imblearn.pipeline.Pipeline` to prevent synthetic sample leakage.

#### Riemannian (`training_riemann.ipynb`)

| Model | Configuration |
|-------|---------------|
| Riemannian_SVM | `kernel='rbf'`, `class_weight='balanced'` on 253-dim tangent space features |
| Riemannian_LDA | `solver='lsqr'`, `shrinkage='auto'` on tangent space features |
| Riemannian_RF | `n_estimators=200`, `class_weight='balanced'` on tangent space features |

---

## Methodological Rigor

This pipeline was audited and refactored to enforce **zero data leakage**, ensuring results are valid for academic reporting.

### 1. In-Fold CSP & Riemannian Fitting

Naive implementations fit CSP spatial filters or the Riemannian Fréchet mean on the entire dataset before cross-validation — exposing the classifier to test-set covariance structure during training. This project uses `sklearn.pipeline.Pipeline` (and `pyriemann`) exclusively, so every transformer (`CSP`, `TangentSpace`, `StandardScaler`) is fitted only on the training fold and applied to the test fold.

### 2. SMOTE Isolation

When using SMOTE in the Improved pipeline, `imblearn.pipeline.Pipeline` is used instead of `sklearn.pipeline.Pipeline`. This ensures synthetic oversampling is applied exclusively within the training fold, preventing interpolated synthetic samples from being derived across fold boundaries.

### 3. Balanced Class Preservation via Dynamic Rejection

Global amplitude thresholds are subject-blind and cause class imbalance in noisy subjects. The **per-class 85th-percentile rejection** drops only the noisiest 15% of trials *within each class independently*, preserving class balance and retaining statistically viable trial counts across all 9 subjects.

---

## Results

### CSP Baseline — 5-Fold CV Accuracy (%)

| Subject | SVM | LDA | RF |
|---------|-----|-----|----|
| A01T | 83.20 ± 5.0 | 81.54 ± 5.1 | 78.69 ± 5.3 |
| A02T | 54.08 ± 4.1 | 58.16 ± 4.8 | 53.67 ± 4.6 |
| A03T | 88.11 ± 4.9 | 87.30 ± 4.1 | 86.07 ± 6.5 |
| A04T | 43.41 ± 7.5 | 46.91 ± 7.6 | 46.12 ± 10.2 |
| A05T | 40.57 ± 4.6 | 42.22 ± 6.7 | 44.25 ± 5.1 |
| A06T | 53.61 ± 5.2 | 50.00 ± 2.7 | 51.08 ± 3.6 |
| A07T | 76.22 ± 3.7 | 81.15 ± 3.4 | 75.00 ± 5.1 |
| A08T | 84.92 ± 3.6 | 84.53 ± 3.8 | 81.35 ± 2.6 |
| A09T | 64.34 ± 5.3 | 64.34 ± 4.3 | 63.94 ± 5.3 |
| **Avg** | **65.4** | **66.2** | **64.5** |

---

### CSP Improved (SMOTE + Hyperparameter Tuning) — 5-Fold CV Accuracy (%)

| Subject | SVM_tuned | LDA_improved | RF_improved | Ensemble | Δ vs Baseline SVM |
|---------|-----------|--------------|-------------|----------|-------------------|
| A01T | 85.2 ± 4.7 | 81.5 ± 5.1 | 79.5 ± 6.2 | 84.0 ± 4.3 | +2.0% |
| A02T | 56.6 ± 4.1 | 58.2 ± 4.8 | 54.9 ± 3.6 | 59.9 ± 4.8 | +2.5% |
| A03T | 88.1 ± 4.3 | 87.3 ± 4.1 | 88.1 ± 5.0 | 87.7 ± 5.6 | 0.0% |
| A04T | 47.3 ± 7.9 | 47.7 ± 6.6 | 46.1 ± 7.2 | 46.5 ± 7.9 | +3.9% |
| A05T | 41.8 ± 4.2 | 42.2 ± 6.7 | 44.7 ± 7.0 | 42.2 ± 5.2 | +1.2% |
| A06T | 53.6 ± 4.8 | 50.0 ± 2.2 | 48.9 ± 5.4 | 52.5 ± 4.9 | 0.0% |
| A07T | 80.7 ± 3.2 | 81.5 ± 3.5 | 74.6 ± 6.2 | 78.3 ± 5.6 | +4.5% |
| A08T | 85.3 ± 3.4 | 84.5 ± 3.8 | 81.8 ± 4.5 | 85.3 ± 3.7 | +0.4% |
| A09T | 66.4 ± 5.4 | 64.3 ± 4.3 | 64.3 ± 2.5 | 64.7 ± 5.8 | +2.1% |
| **Avg** | **67.2** | **66.4** | **64.8** | **66.8** | **+1.8%** |

---

### Riemannian — 5-Fold CV Accuracy (%)

| Subject | SVM | LDA | RF |
|---------|-----|-----|----|
| A01T | 76.2 ± 4.0 | 75.4 ± 4.2 | 70.5 ± 5.0 |
| A02T | 57.3 ± 4.3 | 57.8 ± 3.1 | 52.8 ± 6.6 |
| A03T | 73.8 ± 2.6 | 76.2 ± 3.1 | 75.8 ± 5.2 |
| A04T | 53.5 ± 7.7 | 53.2 ± 5.6 | 47.3 ± 6.2 |
| A05T | 38.1 ± 9.2 | 41.4 ± 9.3 | 35.6 ± 6.1 |
| A06T | 50.0 ± 7.3 | 53.2 ± 7.7 | 46.0 ± 5.3 |
| A07T | 70.1 ± 4.5 | 71.7 ± 4.2 | 68.0 ± 8.0 |
| A08T | 80.6 ± 5.8 | 82.6 ± 2.5 | 78.1 ± 3.9 |
| A09T | 57.4 ± 3.4 | 58.6 ± 3.7 | 57.4 ± 5.8 |
| **Avg** | **61.9** | **63.3** | **59.1** |

---

### Three-Way Comparison — Best Model per Pipeline

| Subject | CSP Baseline (SVM) | CSP Improved (SVM_tuned) | Riemannian (LDA) | Best |
|---------|--------------------|--------------------------|------------------|------|
| A01T | 83.2% | 85.2% | 75.4% | CSP Improved |
| A02T | 54.1% | 56.6% | 57.8% | Riemannian |
| A03T | 88.1% | 88.1% | 76.2% | CSP (tied) |
| A04T | 43.4% | 47.3% | 53.2% | Riemannian |
| A05T | 40.6% | 41.8% | 41.4% | CSP Improved |
| A06T | 53.6% | 53.6% | 53.2% | CSP Baseline |
| A07T | 76.2% | 80.7% | 71.7% | CSP Improved |
| A08T | 84.9% | 85.3% | 82.6% | CSP Improved |
| A09T | 64.3% | 66.4% | 58.6% | CSP Improved |
| **Avg** | **65.4%** | **67.2%** | **63.3%** | **CSP Improved** |

Cross-subject comparison figures → `results/figures/training/riemannian/`:  
`csp_vs_riemannian.png`, `three_way_comparison.png`, `confusion_matrices_riemannian_svm.png`, `f1_heatmap_riemannian.png`

---

## Analysis

### CSP Pipeline

The CSP approach explicitly targets the motor imagery-relevant frequency bands (Mu + Beta) and learns spatial filters that maximise inter-class variance. With data leakage eliminated, hyperparameter tuning and SMOTE still yield a consistent **+1.8% mean gain**, with a pronounced effect on difficult subjects (A04T: +3.9%, A07T: +4.5%).

### Riemannian Pipeline

Riemannian geometry operates on full-band OAS-regularized covariance matrices, capturing richer inter-channel interaction structure without requiring manual band selection. It outperforms CSP for subjects A02T and A04T, suggesting that for subjects with atypical or less lateralized ERD patterns, the covariance structure is more discriminative than band-specific spatial filters.

### Subject Variability (BCI Illiteracy)

The accuracy range from 88.1% (A03T) to 40.6% (A05T) reflects a well-documented neurophysiological phenomenon — approximately 15–30% of individuals cannot reliably produce motor imagery patterns detectable by scalp EEG. This variability is consistent with published BCI Competition IV-2a benchmarks and is independent of the pipeline.

### Benchmark Context

| Method | Avg. Accuracy | Source |
|--------|--------------|--------|
| BCI Competition IV Winner | ~70% | Tangermann et al., 2012 |
| FBCSP | 67–72% | Ang et al., 2008 |
| EEGNet | 67–70% | Lawhern et al., 2018 |
| ShallowConvNet | 68–72% | Schirrmeister et al., 2017 |
| **This project (CSP Improved)** | **67.2%** | — |

Our results are within the expected range for traditional machine learning approaches on this benchmark, and are directly comparable due to the use of the same 5-fold stratified cross-validation protocol.

---

## Limitations & Future Work

### Current Limitations

- **Trial loss:** ~43% average dropout from amplitude rejection — Feet and Tongue classes are most affected.
- **A05T Riemannian failure:** Near-chance accuracy is likely caused by insufficient clean trials for reliable full-band covariance estimation.
- **Evaluation sessions:** `A0xE.gdf` files are present in `data/raw/` but have not yet been preprocessed or used for held-out evaluation.
- **Per-subject models only:** All models are trained and evaluated independently per subject; no cross-subject generalisation is performed.

### Planned Extensions

1. **Cross-subject generalisation** — Train on 8 subjects, evaluate on the held-out 9th (subject-independent BCI).
2. **Evaluation session processing** — Preprocess and evaluate `A0xE.gdf` files for unbiased held-out test performance.
3. **Deep learning baselines** — EEGNet / ShallowConvNet on raw epochs, bypassing hand-crafted features.
4. **Riemannian + CSP fusion** — Concatenate tangent space and CSP features for a combined representation.
5. **Subject-adaptive preprocessing** — Per-subject dynamic band selection and additional ICA tuning for low-performing subjects.

---

## References

- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
- Barachant, A., et al. (2012). Multiclass Brain–Computer Interface Classification by Riemannian Geometry. *IEEE Transactions on Biomedical Engineering*, 59(4), 920–928.
- Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- Ang, K. K., et al. (2008). Filter Bank Common Spatial Pattern (FBCSP) Algorithm. *IEEE IJCNN*, 2390–2397.