# BCI Motor Imagery Classification

## Overview

This project implements a complete, **multi-subject** EEG signal processing pipeline for Brain-Computer Interface (BCI) applications using the **BCI Competition IV Dataset 2a**. The pipeline covers all 9 subjects (A01–A09):

1. **Preprocessing** — artifact removal, ICA, epoching, amplitude rejection
2. **Feature Extraction** — Common Spatial Patterns (CSP) on Mu + Beta bands

The output is a clean, ML-ready feature matrix for each subject, ready for classifier training.

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

| Class | Event Code | Trials / Subject | Description |
|-------|-----------|-----------------|-------------|
| Left Hand | 769 | 72 | Imagined left hand movement |
| Right Hand | 770 | 72 | Imagined right hand movement |
| Both Feet | 771 | 72 | Imagined feet movement |
| Tongue | 772 | 72 | Imagined tongue movement |

Each training session contains 288 balanced trials across 9 experimental runs.

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                          # Raw .gdf files — 18 total (T + E for A01–A09)
│   │   ├── A01T.gdf  A01E.gdf
│   │   └── ...       (through A09)
│   ├── processed/                    # Preprocessed epochs — one .fif per subject (A01–A09)
│   │   ├── A01T_clean_epo.fif
│   │   └── ...       (through A09)
│   └── features/                     # Extracted CSP feature arrays — one .npz per subject
│       ├── A01T_features.npz         # keys: X (n_trials × 16), y (n_trials,)
│       └── ...       (through A09)
│
├── notebooks/
│   ├── bci.ipynb                     # Original single-subject EDA notebook (A01T)
│   ├── preprocessing.ipynb           # Multi-subject preprocessing notebook (A01–A09)
│   └── feature_extraction.ipynb      # Multi-subject CSP feature extraction notebook (A01–A09)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Preprocessing module (loading → ICA → epoching)
│   └── features.py                   # Feature extraction module (CSP + visualization)
│
├── results/
│   └── figures/                      # 120 diagnostic plots (13 per subject + 2 aggregate)
│       ├── Preprocessing plots (per subject):
│       │   ├── A0xT_amplitude_histogram.png
│       │   ├── A0xT_trial_distribution.png
│       │   ├── A0xT_correlation.png
│       │   ├── A0xT_epoch_waveforms.png
│       │   ├── A0xT_topomaps.png
│       │   └── A0xT_psd.png
│       ├── Feature extraction plots (per subject):
│       │   ├── A0xT_csp_patterns.png
│       │   ├── A0xT_csp_filters.png
│       │   ├── A0xT_csp_feature_distribution.png
│       │   ├── A0xT_csp_scatter.png
│       │   ├── A0xT_csp_boxplot.png
│       │   ├── A0xT_csp_feature_correlation.png
│       │   └── A0xT_csp_mean_per_class.png
│       └── Aggregate plots:
│           ├── all_subjects_trial_summary.png
│           └── all_subjects_csp_patterns.png
│
├── bci_script.md                     # Step-by-step notebook walkthrough /                       
├── requirements.txt                  # Pinned dependencies (Python 3.13)
└── README.md
```

---

## Installation & Prerequisites

**Python version:** 3.13 (pinned in `requirements.txt`)

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| `mne` | 1.11.0 | EEG I/O, ICA, CSP, PSD |
| `numpy` | 2.4.2 | Numerical computing |
| `scipy` | 1.17.1 | Signal processing |
| `scikit-learn` | 1.8.0 | CSP, StandardScaler, classifiers |
| `pandas` | 3.0.1 | Tabular data |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualization |
| `ipykernel` | 7.2.0 | Jupyter support |

---

## Source Modules

### `src/preprocessing.py` — EEG Preprocessing

End-to-end preprocessing of raw `.gdf` files into clean, epoched `.fif` files.

| Function | Description |
|----------|-------------|
| `load_subject(subject_id, data_path)` | Loads `.gdf` with EOG channels declared |
| `apply_filter(raw)` | 7–30 Hz FIR bandpass filter |
| `set_montage(raw_filtered)` | Maps GDF channel names to standard 10-20 labels |
| `run_ica(raw_filtered)` | 20-component ICA; removes EOG artifacts via all 3 EOG channels |
| `create_epochs(raw_clean)` | Epochs (−0.5s to +4.5s), baseline correction, rejects trials > 100 µV |
| `visualize_subject(...)` | Generates all 6 preprocessing diagnostic plots |
| `preprocess_subject(subject_id, ...)` | **End-to-end wrapper** — runs all steps, saves `.fif` |

**Usage:**
```python
from src.preprocessing import preprocess_subject, visualize_subject

raw, raw_clean, epochs, epochs_final = preprocess_subject(
    'A01T', data_path='data/raw/', save_path='data/processed/'
)
visualize_subject(raw, raw_clean, epochs, epochs_final,
                  subject_id='A01T', figures_path='results/figures/')
```

---

### `src/features.py` — CSP Feature Extraction

Extracts Common Spatial Pattern (CSP) features from preprocessed epochs for use in classification.

| Function | Description |
|----------|-------------|
| `extract_csp_features(epochs_final, n_components, tmin, tmax)` | **Core function** — extracts 16-dim CSP feature vector per trial |
| `save_features(subject_id, X_csp, y, save_path)` | Saves `X` and `y` as `.npz` |
| `load_features(subject_id, load_path)` | Loads previously saved `.npz` feature file |
| `visualize_features(X_csp, y, csp, epochs_info, subject_id, ...)` | Generates all 7 CSP diagnostic plots |

**Visualization functions:**

| Function | Output File |
|----------|-------------|
| `plot_csp_patterns()` | `{id}_csp_patterns.png` |
| `plot_csp_filters()` | `{id}_csp_filters.png` |
| `plot_csp_feature_distribution()` | `{id}_csp_feature_distribution.png` |
| `plot_csp_scatter()` | `{id}_csp_scatter.png` |
| `plot_csp_feature_boxplot()` | `{id}_csp_boxplot.png` |
| `plot_csp_feature_correlation()` | `{id}_csp_feature_correlation.png` |
| `plot_csp_variance_explained()` | `{id}_csp_mean_per_class.png` |

**Usage:**
```python
from src.features import extract_csp_features, save_features, visualize_features

X_csp, y, csp_mu, csp_beta, scaler, le = extract_csp_features(
    epochs_final, n_components=8, tmin=0.5, tmax=3.5
)
save_features('A01T', X_csp, y, save_path='data/features/')
visualize_features(X_csp, y, csp_mu, epochs_final.info,
                   subject_id='A01T', figures_path='results/figures/')
```

---

## Pipeline

### Stage 1 — Preprocessing (`src/preprocessing.py`)

| Step | Description |
|------|-------------|
| **Load** | Read `.gdf` with EOG channels declared (`EOG-left`, `EOG-central`, `EOG-right`) |
| **Filter** | 7–30 Hz FIR bandpass — isolates Mu (8–13 Hz) and Beta (13–30 Hz) bands |
| **Montage** | Rename GDF channel labels to standard 10-20 names for spatial analysis |
| **ICA** | 20-component ICA; all 3 EOG channels used for artifact detection and removal |
| **Epoch** | Segment at MI cues: −0.5s to +4.5s, baseline corrected to pre-cue window |
| **Reject** | Drop trials with max amplitude > 100 µV (bimodal threshold from histogram) |
| **Save** | Write clean epochs to `data/processed/{id}_clean_epo.fif` |

> **Note on rejection threshold:** MNE's built-in `reject` parameter does not work correctly with this dataset's GDF amplitude scaling. Manual rejection based on the amplitude histogram is used instead.

### Stage 2 — Feature Extraction (`src/features.py`)

| Step | Description |
|------|-------------|
| **Crop** | Trim epochs to active MI window: 0.5s to 3.5s (removes cue response + post-imagery) |
| **Band-split** | Filter separately into Mu (8–13 Hz) and Beta (13–30 Hz) bands |
| **Validate** | Remove any trials containing NaN or Inf values |
| **CSP (Mu)** | Fit `CSP(n_components=8, reg=0.05)` on Mu band → 8 log-variance features |
| **CSP (Beta)** | Fit `CSP(n_components=8, reg=0.05)` on Beta band → 8 log-variance features |
| **Concatenate** | Stack Mu + Beta features → **16 features per trial** |
| **Normalize** | `StandardScaler` to zero mean and unit variance |
| **Save** | Write `X` (n_trials × 16) and `y` to `data/features/{id}_features.npz` |

**Why separate Mu and Beta bands?**
- Mu ERD and Beta ERD/ERS carry complementary discriminative information
- Fitting CSP independently on each band avoids cross-band interference
- Regularization (`reg=0.05`) handles ill-conditioned covariance with small trial counts; auto-raised to `0.2` if NaN is produced

**Output feature shape per subject:** `n_clean_trials × 16`

---

## Results

All 9 subjects have been fully processed through both stages.

| Stage | Output | Location |
|-------|--------|----------|
| Preprocessing | `A0xT_clean_epo.fif` (9 files) | `data/processed/` |
| Feature extraction | `A0xT_features.npz` (9 files) | `data/features/` |
| Figures | 120 diagnostic plots | `results/figures/` |

### Subject A01T — Reference Preprocessing Results

| Class | Clean Trials | Dropped | Retention |
|-------|------------|---------|-----------|
| Left Hand | 44 | 28 | 61% |
| Right Hand | 61 | 11 | 85% |
| Feet | 33 | 39 | 46% |
| Tongue | 26 | 46 | 36% |
| **Total** | **164** | **124** | **57%** |

Final epoch shape for A01T: **164 × 25 × 1,251** → CSP feature matrix: **164 × 16**

Cross-subject trial counts and aggregate CSP patterns are in `results/figures/all_subjects_trial_summary.png` and `all_subjects_csp_patterns.png`.

---

## Key EDA Findings (A01T)

### Scalp Topomaps — Spatial Separability
All four classes show distinct spatial power distributions in the 8–30 Hz band, confirming CSP will be effective:
- **Left Hand:** Desynchronization over right hemisphere (C4) — contralateral ERD
- **Right Hand:** Desynchronization over left hemisphere (C3) — lateralized ERD
- **Feet:** Desynchronization at Cz — midline motor cortex
- **Tongue:** Highest overall power, unique lateral distribution

### Channel Correlation Structure
Three anatomical clusters emerge in clean epochs:
- **Frontal** (Fz, FC3–FC4): 0.84–0.97 within-cluster correlation
- **Central** (C5–C6 strip): 0.75–0.95; C3–C4 = 0.63 (opposing hemispheric roles)
- **Parietal** (CP3–POz): 0.85–0.95

---

## Limitations

- **Trial loss:** ~43% average per subject — Feet and Tongue classes lose most due to EMG contamination.
- **Class imbalance post-rejection:** Up to 1:2.3 ratio (e.g., Tongue vs Right Hand for A01T) — will require `class_weight='balanced'` in classifiers.
- **ICA completeness:** Saturation artifacts (−1600 µV) can interfere with ICA decomposition.
- **No re-referencing:** Common Average Reference (CAR) not applied — could improve SNR further.
- **Evaluation sessions:** `A0xE.gdf` files are in `data/raw/` but not yet preprocessed or evaluated.

---

## Next Steps

1. **Classification** — Train SVM (RBF kernel, `class_weight='balanced'`) on `data/features/*.npz`
2. **Baseline models** — LDA and Random Forest for comparison
3. **Deep learning** — EEGNet for end-to-end classification
4. **Cross-subject generalization** — Evaluate whether one model works across all 9 subjects
5. **Evaluation sessions** — Process `A0xE.gdf` files for held-out evaluation

---

## References

- Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.