# BCI Motor Imagery Classification — EDA & Preprocessing

## Overview

This project implements a complete, **multi-subject** EEG signal processing and Exploratory Data Analysis (EDA) pipeline for Brain-Computer Interface (BCI) applications using the **BCI Competition IV Dataset 2a**. The pipeline processes raw EEG recordings for all **9 subjects** (A01–A09), performs rigorous artifact removal, and produces clean, ML-ready epoched data for a 4-class motor imagery classification task.

The core challenge in EEG-based BCI is that brain signals are extremely weak (microvolts), buried under noise from eye movements, muscle activity, and hardware artifacts. This project documents every step taken to isolate genuine motor imagery brain signals from that noise.

---

## Dataset

- **Source:** BCI Competition IV Dataset 2a
- **Link:** https://www.bbci.de/competition/iv/#dataset2a
- **Format:** `.gdf` (General Data Format for biosignals)
- **Subjects:** A01–A09 (Training `T` and Evaluation `E` sessions for each)

### Recording Specifications

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 250 Hz |
| EEG Channels | 22 |
| EOG Channels | 3 (EOG-left, EOG-central, EOG-right) |
| Total Channels | 25 |
| Hardware Bandpass | 0.5 Hz – 100 Hz |

### Motor Imagery Classes

| Class | Event Code | Trials (per subject) | Description |
|-------|-----------|----------------------|-------------|
| Left Hand | 769 | 72 | Imagined left hand movement |
| Right Hand | 770 | 72 | Imagined right hand movement |
| Both Feet | 771 | 72 | Imagined feet movement |
| Tongue | 772 | 72 | Imagined tongue movement |

Each subject's training session contains 288 trials (perfectly balanced at 72 per class) across 9 experimental runs.

### Additional Event Markers

| Event Code | Count | Meaning |
|-----------|-------|---------|
| 768 | 288 | Trial start / fixation cross |
| 32766 | 9 | New run begins |
| 1023 | varies | Trial rejected by original experimenters |
| 1072 | varies | Eye movement artifact marker |
| 276, 277 | 1 each | Calibration/setup markers |

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                          # Raw input files (18 files – T and E for A01–A09)
│   │   ├── A01T.gdf  A01E.gdf
│   │   ├── A02T.gdf  A02E.gdf
│   │   └── ...       (through A09)
│   ├── processed/                    # Clean epoched data (one .fif per training subject)
│   │   ├── A01T_clean_epo.fif
│   │   ├── A02T_clean_epo.fif
│   │   └── ...       (A01–A09)
│   └── features/                     # (Reserved for extracted CSP/feature files)
├── notebooks/
│   ├── bci.ipynb                     # Original single-subject EDA notebook (A01T)
│   └── preprocessing.ipynb           # Multi-subject preprocessing notebook (A01–A09)
├── src/
│   ├── __init__.py
│   └── preprocessing.py              # Reusable preprocessing module
├── results/
│   └── figures/                      # Per-subject diagnostic plots (55 files)
│       ├── A0xT_amplitude_histogram.png
│       ├── A0xT_correlation.png
│       ├── A0xT_epoch_waveforms.png
│       ├── A0xT_psd.png
│       ├── A0xT_topomaps.png
│       ├── A0xT_trial_distribution.png
│       └── all_subjects_trial_summary.png
├── bci_script.md                     # Step-by-step notebook walkthrough / documentation
├── parse_nb.py                       # Utility: extracts code cells from notebooks
├── requirements.txt                  # Pinned dependencies (Python 3.13)
└── README.md
```

---

## Installation & Prerequisites

**Python version:** 3.13 (as pinned in `requirements.txt`)

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `mne` | 1.11.0 | Core EEG I/O and processing |
| `numpy` | 2.4.2 | Numerical computing |
| `scipy` | 1.17.1 | Signal processing utilities |
| `scikit-learn` | 1.8.0 | ML pipeline (CSP, classifiers) |
| `pandas` | 3.0.1 | Tabular data handling |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualization |
| `ipykernel` | 7.2.0 | Jupyter notebook support |

---

## Source Module — `src/preprocessing.py`

The entire preprocessing pipeline is encapsulated in `src/preprocessing.py` as a set of composable functions. This allows any subject to be processed with a single call to `preprocess_subject()`.

### Public API

| Function | Description |
|----------|-------------|
| `load_subject(subject_id, data_path)` | Loads a `.gdf` file with EOG channels declared |
| `apply_filter(raw)` | Applies 7–30 Hz FIR bandpass filter |
| `set_montage(raw_filtered)` | Renames GDF channels to standard 10-20 labels and sets montage |
| `run_ica(raw_filtered)` | Fits 20-component ICA, detects and removes EOG artifacts using all 3 EOG channels |
| `create_epochs(raw_clean)` | Epochs at MI cues (−0.5s to +4.5s), baselines, and rejects trials > 100 µV |
| `visualize_subject(...)` | Generates all 6 diagnostic plots for a subject |
| `preprocess_subject(subject_id, ...)` | **End-to-end wrapper** — runs all steps and saves `.fif` output |

### Visualization Functions

| Function | Output File |
|----------|-------------|
| `plot_amplitude_histogram()` | `{id}_amplitude_histogram.png` |
| `plot_trial_distribution()` | `{id}_trial_distribution.png` |
| `plot_channel_correlation()` | `{id}_correlation.png` |
| `plot_epoch_waveforms()` | `{id}_epoch_waveforms.png` |
| `plot_topomaps()` | `{id}_topomaps.png` |
| `plot_psd()` | `{id}_psd.png` |

### Usage Example

```python
from src.preprocessing import preprocess_subject, visualize_subject

# Process a subject end-to-end
raw, raw_clean, epochs, epochs_final = preprocess_subject(
    'A01T',
    data_path='data/raw/',
    save_path='data/processed/'
)

# Generate all diagnostic plots
visualize_subject(raw, raw_clean, epochs, epochs_final,
                  subject_id='A01T',
                  figures_path='results/figures/')
```

---

## Preprocessing Pipeline

### Step 1 — Data Loading

Raw `.gdf` files are loaded using `mne.io.read_raw_gdf()` with EOG channels explicitly declared (`EOG-left`, `EOG-central`, `EOG-right`). Each file contains ~672,000 timepoints across 25 channels.

**Saturation artifacts** at −1600 µV (hardware amplifier clipping) are present in some trials, already flagged by the original researchers with event code 1023. These are addressed at the amplitude rejection step.

### Step 2 — Electrode Montage Assignment

GDF files do not store electrode positions. Channel names are mapped from the GDF convention to standard 10-20 labels:

| GDF Name | 10-20 Name | Brain Region |
|----------|-----------|--------------|
| EEG-Fz | Fz | Frontal midline |
| EEG-C3 | C3 | Left motor cortex |
| EEG-Cz | Cz | Central midline |
| EEG-C4 | C4 | Right motor cortex |
| EEG-Pz | Pz | Parietal midline |
| EEG-0 to EEG-16 | FC3–POz | Motor/parietal areas |

### Step 3 — Bandpass Filtering (7–30 Hz)

A FIR bandpass filter isolates the motor imagery frequency bands:

- **Mu rhythm (8–13 Hz):** Primary motor cortex oscillation — suppresses during movement imagery (ERD).
- **Beta rhythm (13–30 Hz):** Secondary motor rhythm — also shows ERD during imagery, rebounds after (ERS).

Filtering removes slow DC drift (< 7 Hz) and high-frequency muscle noise (> 30 Hz).

### Step 4 — ICA Artifact Removal

Independent Component Analysis (ICA) with **20 components** identifies and removes ocular artifacts. All 3 EOG channels are used for detection:

- `EOG-left` / `EOG-right` — horizontal eye movements
- `EOG-central` — vertical blinks (strongest artifact source)

Using all 3 channels is critical; using only one misses components detectable only by the others (e.g., blink components visible only in `EOG-central`). Bad components are identified per subject and excluded before applying ICA to the clean signal.

### Step 5 — Epoching

Continuous clean signal is segmented into discrete trials time-locked to each motor imagery cue:

| Parameter | Value |
|-----------|-------|
| Time window | −0.5s to +4.5s relative to cue |
| Window length | 5 s (1,251 timepoints at 250 Hz) |
| Baseline correction | −0.5s to 0s (pre-cue subtracted) |
| Initial epochs | 288 per subject |

### Step 6 — Amplitude-Based Artifact Rejection

Trial maximum amplitude is computed across all channels and timepoints. Trials exceeding **100 µV** are rejected. This threshold was chosen by inspecting the bimodal amplitude distribution for Subject A01T, where a clear gap at ~100–120 µV separates clean trials from artifact-contaminated ones.

> **Note:** MNE's built-in `reject` parameter in `mne.Epochs` does not function correctly with this dataset's internal GDF scaling. Manual rejection based on the amplitude histogram is used instead.

---

## Results

All 9 subjects have been fully processed. Clean epochs are saved to `data/processed/` and all diagnostic figures to `results/figures/`.

### Subject A01T (Reference Subject — Detailed Analysis)

**Final clean dataset after rejection:**

| Class | Clean Trials | Dropped | Retention Rate |
|-------|------------|---------|----------------|
| Left Hand | 44 | 28 | 61% |
| Right Hand | 61 | 11 | 85% |
| Feet | 33 | 39 | 46% |
| Tongue | 26 | 46 | 36% |
| **Total** | **164** | **124** | **57%** |

**Final data shape: 164 trials × 25 channels × 1,251 timepoints**

Right Hand was the cleanest class. Feet and Tongue showed the most artifacts — consistent with literature where foot imagery causes subtle muscle tension and tongue imagery causes jaw/EMG contamination.

### Pipeline Summary

| Stage | Trials | Notes |
|-------|--------|-------|
| Raw recording | 288 MI trials | Balanced: 72 per class |
| After bandpass filter | 288 | No trials removed |
| After ICA | 288 | 2+ artifact components removed |
| After amplitude rejection | subject-dependent | Threshold: 100 µV |

An aggregate cross-subject trial summary is available at `results/figures/all_subjects_trial_summary.png`.

---

## Key EDA Findings (A01T)

### Channel Correlation Structure
Three anatomically meaningful clusters emerge:
- **Frontal (Fz, FC3–FC4):** Within-cluster correlation 0.84–0.97
- **Central (C5–C6 strip):** Within-cluster correlation 0.75–0.95; C3–C4 correlation = 0.63 (reflecting opposing left/right roles)
- **Parietal (CP3–POz):** Within-cluster correlation 0.85–0.95

Front-to-back correlation (Fz → POz) = 0.07 — near-complete independence, neurophysiologically correct.

### Power Spectral Density
A clear peak is observed around 10–12 Hz (Mu rhythm) in the PSD, confirming the data contains the frequency content needed for motor imagery classification.

### Scalp Topomaps (8–30 Hz Power)
All four classes show distinct spatial power distributions:
- **Left Hand:** Desynchronization over right hemisphere (C4 area) — contralateral ERD
- **Right Hand:** Desynchronization over left hemisphere (C3 area) — confirms lateralized ERD
- **Feet:** Central desynchronization at Cz — midline motor cortex
- **Tongue:** Highest overall power, unique lateral distribution

This spatial separability confirms that **Common Spatial Patterns (CSP)** will be an effective feature extractor.

---

## Limitations

- **Trial loss:** Removing ~43% of trials for A01T is substantial. Per-subject dropout rates vary (see `results/figures/`).
- **Class imbalance after rejection:** 26 Tongue vs 61 Right Hand trials for A01T (1:2.3 ratio) — requires balanced class weighting in classifiers.
- **ICA completeness:** Saturation artifacts (−1600 µV) partially interfere with ICA decomposition; some residual artifacts may remain.
- **No re-referencing:** Common Average Reference (CAR) was not applied — this could further improve signal quality.
- **Evaluation sessions:** Only training sessions (`T`) are preprocessed. Evaluation sessions (`E`) are available in `data/raw/` but not yet processed.

---

## Next Steps

1. **Feature Extraction — Common Spatial Patterns (CSP):** Learn spatial filters that maximize class variance differences. Output goes to `data/features/`.
2. **Classification — SVM with RBF kernel:** Train on CSP features with `class_weight='balanced'`. Expected accuracy > 70% for A01T per literature benchmarks.
3. **Alternative models:** LDA (fast baseline), Random Forest, and EEGNet (deep learning).
4. **Cross-subject generalization:** Evaluate whether a single model generalizes across all 9 subjects.
5. **Evaluation sessions:** Process `A0xE.gdf` files for held-out evaluation.

---

## References

- Tangermann, M., Müller, K. R., Aertsen, A., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842-1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.