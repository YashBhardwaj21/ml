# BCI Motor Imagery Classification: EDA & Preprocessing

## Overview

This project implements a complete EEG signal processing and exploratory data analysis (EDA) pipeline for Brain-Computer Interface (BCI) applications using the BCI Competition IV Dataset 2a. The pipeline processes raw EEG recordings from a single subject (A01T), performs rigorous artifact removal, and produces clean, ML-ready epoched data for a 4-class motor imagery classification task.

The core challenge in EEG-based BCI is that brain signals are extremely weak (microvolts), buried under noise from eye movements, muscle activity, and hardware artifacts. This notebook documents every step taken to isolate genuine motor imagery brain signals from that noise.

---

## Dataset

- **Source:** BCI Competition IV Dataset 2a
- **Link:** https://www.bbci.de/competition/iv/
- **Format:** `.gdf` (General Data Format for biosignals)
- **Subject used:** A01T (Training session)

### Recording Specifications

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 250 Hz |
| EEG Channels | 22 |
| EOG Channels | 3 (EOG-left, EOG-central, EOG-right) |
| Total Channels | 25 |
| Recording Duration | 2690.11 seconds (~44.8 minutes) |
| Hardware Bandpass | 0.5 Hz – 100 Hz |

### Motor Imagery Classes

| Class | Event Code | Trials | Description |
|-------|-----------|--------|-------------|
| Left Hand | 769 | 72 | Imagined left hand movement |
| Right Hand | 770 | 72 | Imagined right hand movement |
| Both Feet | 771 | 72 | Imagined feet movement |
| Tongue | 772 | 72 | Imagined tongue movement |

The dataset is perfectly balanced with 72 trials per class (288 total) across 9 experimental runs, making it ideal for machine learning without class imbalance at the raw stage.

### Additional Event Markers Found in Data

| Event Code | Count | Meaning |
|-----------|-------|---------|
| 768 | 288 | Trial start / fixation cross (appears before every MI cue) |
| 32766 | 9 | New run begins (session structure marker) |
| 1023 | 15 | Rejected trial (flagged by original experimenters) |
| 1072 | 1 | Eye movement artifact marker |
| 276, 277 | 1 each | Calibration/setup markers at session start |

---

## Project Structure

```
├── data/
│   ├── A01T.gdf                  # Raw input file
│   └── A01T_clean_epo.fif        # Output: clean epoched data ready for ML
├── bci.ipynb                     # Main Jupyter notebook (full pipeline)
└── README.md
```

---

## Installation & Prerequisites

```bash
pip install mne numpy pandas matplotlib seaborn scikit-learn scipy
```

Python 3.8+ recommended. All processing is done in a single Jupyter notebook using MNE-Python as the core EEG processing library.

---

## Preprocessing Pipeline

### Step 1 — Data Loading

Raw `.gdf` file loaded using `mne.io.read_raw_gdf()` with EOG channels explicitly declared. The file contains 672,528 timepoints across 25 channels.

**Key discovery during loading:** Raw signal statistics showed saturation artifacts at -1600 µV (hardware amplifier clipping) in the continuous recording. These extreme values were present in 15 trials already flagged by the original researchers with event code 1023, and were handled in the artifact rejection step.

### Step 2 — Electrode Montage Assignment

GDF files do not store electrode position information. The standard 10-20 montage was manually assigned by renaming channels from the GDF naming convention (e.g., `EEG-C3`, `EEG-0`) to standard 10-20 labels (e.g., `C3`, `FC3`). This is required for topographic visualizations and ICA.

Full channel mapping used:

| GDF Name | 10-20 Name | Brain Region |
|----------|-----------|--------------|
| EEG-Fz | Fz | Frontal midline |
| EEG-C3 | C3 | Left motor cortex |
| EEG-Cz | Cz | Central midline |
| EEG-C4 | C4 | Right motor cortex |
| EEG-Pz | Pz | Parietal midline |
| EEG-0 to EEG-16 | FC3–POz | Various motor/parietal |

### Step 3 — Bandpass Filtering (7–30 Hz)

A FIR bandpass filter was applied to isolate the two frequency bands relevant to motor imagery:

- **Mu rhythm (8–13 Hz):** Primary motor cortex oscillation. Suppresses during movement or imagination of movement (Event Related Desynchronization, ERD).
- **Beta rhythm (13–30 Hz):** Secondary motor rhythm. Also shows ERD during motor imagery and rebounds after (Event Related Synchronization, ERS).

Filtering removed slow DC drift (below 7 Hz) and high-frequency muscle noise (above 30 Hz). Visual inspection of channel C3 confirmed the filtered signal shows cleaner rhythmic oscillations (±20 µV) compared to the raw signal (±30 µV with high-frequency jitter).

### Step 4 — ICA Artifact Removal

Independent Component Analysis (ICA) with 20 components was fitted on the filtered data to identify and remove ocular artifacts.

**All 3 EOG channels were used for detection** (not just one), which is critical because:
- `EOG-left` and `EOG-right` detect horizontal eye movements
- `EOG-central` detects vertical blinks (most powerful artifact source)

**Components identified and removed:**

| Component | Detected By | Artifact Type |
|-----------|------------|---------------|
| ICA000 | EOG-left, EOG-central, EOG-right | Strong combined eye artifact |
| ICA011 | EOG-central only | Blink artifact |

Using all 3 EOG channels identified component 11 (blink artifact) that was missed when only checking EOG-left. After ICA application:
- Before ICA: max amplitude ~62 µV, min ~-54 µV (on filtered data)
- After ICA: cleaner signal with reduced frontal contamination

### Step 5 — Epoching

The continuous clean signal was segmented into discrete trials time-locked to each motor imagery cue:

| Parameter | Value |
|-----------|-------|
| Time window | -0.5s to 4.5s relative to cue |
| Total window length | 5 seconds (1,251 timepoints at 250 Hz) |
| Baseline correction | -0.5s to 0s (pre-cue period subtracted) |
| Initial epochs created | 288 |

Baseline correction removes any DC offset present before the cue, ensuring amplitude changes are measured relative to the pre-stimulus resting state.

### Step 6 — Amplitude-Based Artifact Rejection

After epoching, amplitude distribution analysis revealed a clear **bimodal distribution** — a natural separation between clean trials and artifact-contaminated trials:

| Threshold | Surviving Trials |
|-----------|----------------|
| 100 µV | 164 trials |
| 150 µV | 226 trials |
| 200 µV | 276 trials |

The bimodal gap in the histogram at ~100–120 µV indicated that 100 µV is the natural rejection threshold for this subject. Trials above 100 µV belong to a second artifact-dominated population, not genuine brain signal variability.

**Note:** MNE's built-in `reject` parameter in `mne.Epochs` did not function correctly with this dataset's internal GDF scaling. Manual rejection based on the amplitude histogram was used instead.

**Final clean dataset after rejection:**

| Class | Clean Trials | Dropped | Retention Rate |
|-------|------------|---------|----------------|
| Left Hand | 44 | 28 | 61% |
| Right Hand | 61 | 11 | 85% |
| Feet | 33 | 39 | 46% |
| Tongue | 26 | 46 | 36% |
| **Total** | **164** | **124** | **57%** |

Right Hand was the cleanest class. Feet and Tongue showed the most artifacts — consistent with EEG literature where foot imagery causes subtle muscle tension and tongue imagery causes jaw/EMG contamination.

**Final data shape: (164 trials × 25 channels × 1,251 timepoints)**

---

## Key Findings from EDA

### 1. Channel Correlation Structure (Clean Epochs)

The correlation matrix computed on clean epochs revealed meaningful neuroanatomical structure — three distinct clusters corresponding to brain regions:

- **Frontal cluster (Fz, FC3, FC1, FCz, FC2, FC4):** Within-cluster correlation 0.84–0.97. These channels sit close together over frontal motor planning areas.
- **Central cluster (C5, C3, C1, Cz, C2, C4, C6):** Within-cluster correlation 0.75–0.95. Primary motor cortex strip. C3–C4 correlation = 0.63, notably lower than their neighbors, reflecting their opposing roles in left vs right hand imagery.
- **Parietal cluster (CP3–POz):** Within-cluster correlation 0.85–0.95. Sensory-motor integration region.

The front-to-back correlation (Fz → POz = 0.07) confirms near-complete independence between frontal and parietal regions — neurophysiologically correct.

### 2. Power Spectral Density

A clear peak was observed around 10–12 Hz (Mu rhythm) in the PSD across all channels. This is the signature of the idle motor cortex and confirms the data contains the frequency content needed for motor imagery classification.

### 3. Scalp Topomaps (8–30 Hz Power)

The most important finding: all four classes show **distinct spatial power distributions**:

- **Left Hand:** Low-power region in the right hemisphere (C4 area) — contralateral motor cortex desynchronization
- **Right Hand:** Low-power region shifts to left hemisphere (C3 area) — confirms lateralized ERD
- **Feet:** Central low-power region at Cz — midline motor cortex, consistent with foot representation
- **Tongue:** Highest overall power (max 293 µV²/Hz vs 225 for Left Hand), unique lateral distribution

This spatial separability directly confirms that **Common Spatial Patterns (CSP) will be an effective feature extractor** for classification.

### 4. Epoch Time-Domain Analysis

All four classes showed mean amplitude near 0 µV across the entire trial window — this is correct. Motor imagery produces changes in oscillatory power (ERD/ERS), not raw amplitude deflections. A brief ERP response at t=0 (cue onset) was visible across all classes, consistent with the visual stimulus processing.

---

## Limitations

- **43% trial loss:** Removing 124 of 288 trials is substantial. Subject A01T shows higher artifact rates in Feet and Tongue classes, which may reflect genuine subject-specific EMG contamination during those imagery tasks.
- **Class imbalance after rejection:** 26 Tongue vs 61 Right Hand trials (1:2.3 ratio) requires balanced class weighting in the ML classifier.
- **Single subject:** This analysis covers only A01T. The 9-subject dataset will show different rejection rates and spatial patterns per subject.
- **ICA limitations:** The saturation artifacts (-1600 µV) in the raw data partially interfered with ICA decomposition, explaining why only 2 components were cleanly identified.
- **No re-referencing:** Common Average Reference (CAR) was not applied, which could further improve signal quality before classification.

---

## Results Summary

| Stage | Trials | Notes |
|-------|--------|-------|
| Raw recording | 288 MI trials | Perfectly balanced, 72 per class |
| After bandpass filter | 288 | No trials removed, signal cleaned |
| After ICA | 288 | 2 artifact components removed |
| After amplitude rejection | 164 | 124 trials dropped at 100 µV threshold |
| **Final clean data** | **164** | **Shape: 164 × 25 × 1251** |

---

## Next Steps

1. **Feature Extraction — Common Spatial Patterns (CSP):** Learn optimal spatial filters that maximize variance differences between classes. CSP is the standard approach for motor imagery given the spatial separability confirmed in topomaps.

2. **Classification — SVM with RBF kernel:** Train on CSP features with `class_weight='balanced'` to handle the trial imbalance. Expected accuracy >70% for Subject A01T based on literature benchmarks.

3. **Alternative models:** LDA (fast baseline), Random Forest, and EEGNet (deep learning) for comparison.

4. **Cross-subject generalization:** Repeat pipeline for all 9 subjects and evaluate whether a single model can generalize across subjects.

---

## References

- Tangermann, M., Müller, K. R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., ... & Blankertz, B. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842-1857.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.