# BCI Motor Imagery Classification

## Overview

This project implements a complete, **multi-subject** EEG signal processing and classification pipeline for Brain-Computer Interface (BCI) applications using the **BCI Competition IV Dataset 2a**. The pipeline covers all 9 subjects (A01–A09) across three stages:

1. **Preprocessing** — ICA artifact removal, bandpass filtering, epoching, amplitude rejection
2. **Feature Extraction** — Common Spatial Patterns (CSP) on Mu + Beta bands
3. **Classification** — SVM, LDA, and Random Forest with 5-fold stratified cross-validation

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

Each training session contains 288 balanced trials across 9 experimental runs.

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                              # Raw .gdf files — 18 total (T + E for A01–A09)
│   ├── processed/                        # Clean epoched data — A01T–A09T_clean_epo.fif
│   └── features/                         # CSP feature arrays — A01T–A09T_features.npz
│
├── notebooks/
│   ├── bci.ipynb                         # Original single-subject EDA (A01T)
│   ├── preprocessing.ipynb               # Multi-subject preprocessing (A01–A09)
│   ├── feature_extraction.ipynb          # Multi-subject CSP feature extraction (A01–A09)
│   └── training.ipynb                    # Multi-subject classifier training (A01–A09)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Preprocessing module
│   ├── features.py                       # Feature extraction module
│   └── models.py                         # Classifier training & evaluation module
│
├── results/
│   ├── figures/
│   │   ├── preprocessing/                # 55 plots — 6 per subject + 1 aggregate
│   │   ├── features/                     # 64 plots — 7 per subject + 1 aggregate
│   │   └── training/
│   │       └── baseline/                 # 9 aggregate training plots
│   ├── metrics/
│   │   └── baseline/                     # 3 CSV files with accuracy, F1, and fold scores
│   └── models/
│       └── baseline/                     # 27 trained model files (3 models × 9 subjects)
│
├── bci_script.md                         # Notebook walkthrough / documentation
├── parse_nb.py                           # Utility: extracts code cells from notebooks
├── requirements.txt                      # Pinned dependencies (Python 3.13)
└── README.md
```

---

## Installation

**Python version:** 3.13

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
| `seaborn` | 0.13.2 | Heatmaps and statistical plots |
| `ipykernel` | 7.2.0 | Jupyter support |

---

## Source Modules

### `src/preprocessing.py`

End-to-end preprocessing of raw `.gdf` files into clean, epoched `.fif` files.

| Function | Description |
|----------|-------------|
| `load_subject(subject_id, data_path)` | Loads `.gdf` with EOG channels declared |
| `apply_filter(raw)` | 7–30 Hz FIR bandpass |
| `set_montage(raw_filtered)` | Maps GDF channel names → standard 10-20 labels |
| `run_ica(raw_filtered)` | 20-component ICA; removes artifacts using all 3 EOG channels |
| `create_epochs(raw_clean)` | Epochs (−0.5s to +4.5s), baseline-corrected, rejects trials > 100 µV |
| `visualize_subject(...)` | Generates all 6 preprocessing plots |
| `preprocess_subject(subject_id, ...)` | End-to-end wrapper — saves `.fif` to `data/processed/` |

**Output plots** → `results/figures/preprocessing/`:
`_amplitude_histogram`, `_trial_distribution`, `_correlation`, `_epoch_waveforms`, `_topomaps`, `_psd`

---

### `src/features.py`

Extracts CSP features from preprocessed epochs.

| Function | Description |
|----------|-------------|
| `extract_csp_features(epochs_final, n_components, tmin, tmax)` | Extracts 16-dim normalized CSP feature vector per trial |
| `save_features(subject_id, X_csp, y, save_path)` | Saves `X` and `y` as `.npz` to `data/features/` |
| `load_features(subject_id, load_path)` | Loads previously saved feature file |
| `visualize_features(X_csp, y, csp, epochs_info, subject_id, ...)` | Generates all 7 CSP diagnostic plots |

**Output plots** → `results/figures/features/`:
`_csp_patterns`, `_csp_filters`, `_csp_feature_distribution`, `_csp_scatter`, `_csp_boxplot`, `_csp_feature_correlation`, `_csp_mean_per_class`

---

### `src/models.py`

Classifier training, evaluation, persistence, and visualization.

| Function | Description |
|----------|-------------|
| `build_svm()` | SVM — RBF kernel, `class_weight='balanced'`, `probability=True` |
| `build_lda()` | LDA — `solver='lsqr'`, `shrinkage='auto'` |
| `build_rf()` | Random Forest — 200 estimators, `class_weight='balanced'` |
| `evaluate_model(model, X, y, n_splits=5)` | 5-fold stratified CV; returns scores, confusion matrix, classification report |
| `save_model(model, subject_id, model_name, save_path)` | Serializes trained model to `.pkl` |
| `load_model(subject_id, model_name, load_path)` | Loads a saved `.pkl` model |
| `save_metrics(all_results, subjects, save_path)` | Writes 3 CSVs: accuracy summary, per-class metrics, per-fold scores |
| `run_all_visualizations(all_results, subjects, figures_path)` | Generates all 9 training plots |

**Output plots** → `results/figures/training/baseline/`:

| File | Description |
|------|-------------|
| `accuracy_bars.png` | Bar chart — mean ± std accuracy across all subjects and models |
| `confusion_matrices_SVM.png` | 3×3 grid of normalized confusion matrices (SVM) |
| `confusion_matrices_LDA.png` | 3×3 grid of normalized confusion matrices (LDA) |
| `confusion_matrices_RF.png` | 3×3 grid of normalized confusion matrices (RF) |
| `f1_heatmap.png` | F1 score heatmap — subjects × classes, all 3 models |
| `per_fold_scores.png` | Per-fold accuracy line plot per subject |
| `model_comparison_boxplot.png` | Boxplot — accuracy distribution across subjects per model |
| `recall_heatmap.png` | Recall heatmap — subjects × classes, all 3 models |
| `best_subject_detail_A03T.png` | Confusion matrices for best subject across all 3 models |

---

## Pipeline

### Stage 1 — Preprocessing

| Step | Detail |
|------|--------|
| Load | `.gdf` with EOG channels declared |
| Filter | 7–30 Hz FIR bandpass — isolates Mu (8–13 Hz) and Beta (13–30 Hz) |
| Montage | GDF labels → standard 10-20 names |
| ICA | 20 components; all 3 EOG channels used for artifact detection |
| Epoch | −0.5s to +4.5s, baseline corrected to −0.5s–0s |
| Reject | Manual 100 µV threshold (MNE's built-in `reject` param doesn't work with this dataset's GDF scaling) |
| Save | `data/processed/{id}_clean_epo.fif` |

### Stage 2 — Feature Extraction

| Step | Detail |
|------|--------|
| Crop | Active MI window: 0.5s to 3.5s |
| Band-split | Separate Mu (8–13 Hz) and Beta (13–30 Hz) filtering |
| CSP (Mu) | 8 components, `reg=0.05`; auto-raises to `0.2` on NaN |
| CSP (Beta) | 8 components, `reg=0.05`; auto-raises to `0.2` on NaN |
| Concatenate | Mu + Beta → **16 features per trial** |
| Normalize | `StandardScaler` (zero mean, unit variance) |
| Save | `data/features/{id}_features.npz` — keys: `X` (n_trials × 16), `y` |

### Stage 3 — Classification

| Step | Detail |
|------|--------|
| Load | `data/features/{id}_features.npz` |
| Models | SVM (RBF), LDA (shrinkage), Random Forest (200 trees) |
| Evaluation | 5-fold stratified cross-validation (accuracy, CM, classification report) |
| Save models | `results/models/baseline/{id}_{model}.pkl` |
| Save metrics | `results/metrics/baseline/` — 3 CSV files |

---

## Results

### Baseline Classification Accuracy (5-fold CV %)

| Subject | SVM | LDA | RF |
|---------|-----|-----|----|
| A01T | 84.19 ± 4.37 | 83.54 ± 6.22 | 81.12 ± 3.44 |
| A02T | 58.81 ± 3.27 | 60.19 ± 2.61 | 58.82 ± 5.97 |
| **A03T** | **87.71 ± 3.75** | **86.02 ± 5.03** | **87.68 ± 2.98** |
| A04T | 55.36 ± 2.34 | 57.65 ± 5.88 | 54.22 ± 5.92 |
| A05T | 54.31 ± 4.22 | 55.14 ± 1.06 | 57.48 ± 5.85 |
| A06T | 53.57 ± 3.68 | 55.68 ± 3.11 | 49.67 ± 4.21 |
| A07T | 82.16 ± 4.12 | 81.45 ± 8.95 | 76.60 ± 11.48 |
| **A08T** | **87.16 ± 4.91** | **87.58 ± 5.18** | **83.83 ± 4.00** |
| A09T | 62.59 ± 5.58 | 64.96 ± 8.48 | 62.03 ± 5.80 |

**Chance level: 25%** (4-class classification)

- Best subject: **A03T** (SVM: 87.71%)
- High performers (>80%): A01T, A03T, A07T, A08T
- Low performers (<60%): A04T, A05T, A06T — likely high per-subject artifact rates
- SVM and LDA are generally competitive; RF shows higher variance (especially A07T)

---

## Limitations

- **Trial loss:** ~43% average dropout per subject from amplitude rejection — Feet and Tongue classes most affected
- **Class imbalance post-rejection:** Requires `class_weight='balanced'` in all classifiers
- **Subject variability:** Accuracy ranges from 53% to 88% — pipeline improvements (re-referencing, adaptive thresholding) may help low-performing subjects
- **Evaluation sessions:** `A0xE.gdf` files are in `data/raw/` but not yet processed
- **`improved/` directories:** `results/models/improved/`, `results/metrics/improved/`, and `results/figures/training/improved/` are reserved for the next iteration

---

## Next Steps

1. **Hyperparameter tuning** — Grid search over SVM `C`/`gamma`, RF `n_estimators`, CSP regularization
2. **Pipeline improvements** — Common Average Reference (CAR), adaptive rejection thresholds per subject
3. **Deep learning** — EEGNet end-to-end classification on raw epochs
4. **Cross-subject generalization** — Train on N-1 subjects, evaluate on held-out subject
5. **Evaluation sessions** — Process `A0xE.gdf` for proper held-out test set evaluation

---

## References

- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.