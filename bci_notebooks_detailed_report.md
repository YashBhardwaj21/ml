# Machine Learning Project: BCI Notebook Analysis Report

This document provides a comprehensive technical breakdown of the 7 Jupyter Notebooks present in the project. It details the **What**, **How**, and **Why** for each step of the pipeline, from raw EEG data exploration to advanced Riemannian geometry classification.

---

## 1. `bci.ipynb` (Deep Exploratory Data Analysis)

**What is it?**
A meticulous and deep exploratory data analysis (EDA) conducted on a single subject (`A01T`) from the BCI Competition IV Dataset 2a.

**How does it work?**
- **Data Loading:** Loads the raw `.gdf` files containing 22 EEG channels and 3 EOG channels using `mne`.
- **Filtering:** Applies a FIR bandpass filter (7–30 Hz) targeting the Mu (8–13 Hz) and Beta (13–30 Hz) bands relevant to motor imagery.
- **Artifact Removal:** Fits Independent Component Analysis (ICA) strictly prioritizing the 3 EOG channels to find and nullify eye movement artifacts.
- **Epoching & Rejection:** Slices the continuous data into discrete 5-second windows (-0.5s to 4.5s relative to the cue). Rather than using MNE's auto-reject, the notebook derives a manual threshold (100–150µV max amplitude) by plotting the histogram of peak-to-peak amplitudes, identifying a natural cutoff to drop noisy trials.
- **Initial Visualization:** Computes Power Spectral Densities (PSD) to visualize brain wave bands and plots scalp topomaps. It extracts a small set of Common Spatial Pattern (CSP) features just to trace the data dimensions visually.

**Why was it done?**
Before blindly passing 9 subjects through a pipeline, it is critical to uncover dataset quirks for a single subject. This notebook scientifically validates the artifact rejection strategy (manual vs. auto) and validates the bandpass frequency ranges to build the foundation for the automated pipeline.

---

## 2. `preprocessing.ipynb` (Automated Pre-Processing)

**What is it?**
The operationalized pipeline that standardizes and cleans the raw EEG data for all 9 subjects simultaneously.

**How does it work?**
- **Batch Processing:** Iterates over subjects `A01T` through `A09T`.
- **Pipeline Execution:** Hooks into the codebase (`src.preprocessing`) to rapidly apply the filtering, ICA-based EOG correlation dropping, and amplitude thresholding (max 100µV) discovered during the EDA phase.
- **Exporting:** Drops corrupted epochs and retains only high-quality motor imagery trials. The cleaned epochs are saved to disk as `[Subject]_clean_epo.fif`.
- **Summarization:** Generates automated reporting charts (`all_subjects_trial_summary.png`) showing exactly how many trials survived the cleaning process across all 4 classes (Left Hand, Right Hand, Feet, Tongue) per subject.

**Why was it done?**
It converts raw, highly disorganized physiological time-series data into a clean, uniform format. It also ensures reproducibility: if a model fails later, the researchers know it's not due to random eye-blink artifacts or inconsistent data structures creeping in.

---

## 3. `feature_extraction.ipynb` (CSP Feature Extraction)

**What is it?**
A feature engineering notebook specifically focused on extracting traditional Common Spatial Pattern (CSP) features from the cleaned epochs.

**How does it work?**
- **Loading:** Loads the `.fif` clean epoch files.
- **CSP Computation:** Fits the CSP algorithm targeting 4 components. 
- **Visualization:** Plots detailed scalp topomaps representing the spatial filters for the Mu and Beta bands across all 9 subjects. It also plots 2D scatter visualizations showing the linear separability of the 4 classes based on the first two CSP features.
- **Saving:** The transformed low-dimensional arrays (`X_csp`, `y`) are saved in `.npz` format for the machine learning models.

**Why was it done?**
Raw EEG dimensionality (Channels x Timepoints) is far too large and sparse for standard ML models. CSP acts as a supervised dimensionality reduction technique; it explicitly finds spatial filters (linear combinations of electrodes) that maximize the variance for one specific imagery task while minimizing it for the others.

---

## 4. `features_riemannian.ipynb` (Riemannian Geometry Feature Extraction)

**What is it?**
An advanced feature extraction notebook utilizing differential geometry, specifically treating EEG signals as covariance matrices.

**How does it work?**
- **Covariance Estimation:** Calculates the spatial covariance matrix for each trial's EEG signal using the `pyriemann` library.
- **Tangent Space Mapping:** Covariance matrices represent Symmetric Positive Definite (SPD) matrices, which sit on a curved Riemannian manifold. The notebook projects these curved structures into a flat, Euclidean Tangent Space.
- **Visualization:** Performs Principal Component Analysis (PCA) on the tangent space vectors to produce 2D scatter plots of class separability, printing explained variance.
- **Saving:** Persists these advanced (`X_ts`, `y`) features to disk.

**Why was it done?**
CSP can sometimes overfit or fail if spatial covariance is non-stationary across trials. By treating the brain state as an entity on a Riemannian manifold, we map the actual geometric structure of the brain's activity directly into feature vectors, often resulting in state-of-the-art robustness for Brain-Computer Interfaces.

---

## 5. `training_baseline.ipynb` (Baseline Modeling)

**What is it?**
The first modeling phase. Establishes the performance floor using simple, out-of-the-box machine learning over the CSP features.

**How does it work?**
- **Models:** Instantiates vanilla Support Vector Machine (SVM), Linear Discriminant Analysis (LDA), and Random Forest (RF) classifiers.
- **Evaluation:** Evaluates each model via cross-validation for every subject. Accuracy, standard deviation, and F1-scores are logged.
- **Exporting:** Serializes the trained models (`.pkl` format) and summary statistics to CSV for future comparison.

**Why was it done?**
Before introducing complex hyperparameter tuning or advanced architectures, it is vital to know that the pipeline actually works. It establishes baseline metrics (e.g., how far above the 25.0% chance level are the models performing?) and identifies which subjects naturally produce strong "BCI Illiteracy" (bad performance) vs. strong BCI signals.

---

## 6. `training_improved.ipynb` (Improved Optimization Pipeline)

**What is it?**
An enhancement over the baseline, leveraging hyperparameter optimization, class-balancing techniques, and model ensembling to squeeze maximum accuracy from the CSP features.

**How does it work?**
- **Imbalance Handling:** The rigorous artifact rejection in the preprocessing phase causes class imbalances (e.g., dropping 20 "Feet" trials but only 2 "Tongue" trials). This notebook uses **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the classes.
- **Tuning:** Performs a comprehensive `GridSearchCV` to find the mathematically optimal `C` and `gamma` parameters for an RBF-kernel SVM.
- **Ensemble:** Creates a Soft Voting Classifier that averages the prediction probabilities of the tuned SVM, an optimized LDA layer (using least squares and auto-shrinkage), and an RF.
- **Comparison:** Generates robust visual comparisons (Bar charts and Heatmaps) directly comparing Baseline SVM vs Tuned SVM vs Ensemble across all subjects. 

**Why was it done?**
To address weaknesses discovered in the baseline (imbalanced classes heavily biasing classifiers) and to prove that careful feature-engineering and optimization can noticeably elevate the accuracy ceiling of traditional CSP pipelines.

---

## 7. `training_riemann.ipynb` (Riemannian Classification and Final Comparison)

**What is it?**
The final modeling stage. It trains models directly on the advanced Riemannian Tangent Space features and evaluates the "CSP vs. Riemannian" hypothesis.

**How does it work?**
- **Training:** Runs SVM, LDA, and RF on the tangent space data (`X_ts`).
- **Comprehensive Benchmarking:** Ingests the saved metrics from the `baseline` and `improved` pipelines and creates a **Three-Way Comparison Bar Chart** (CSP Baseline vs. CSP Improved vs. Riemannian SVM).
- **Deep Metrics:** Generates confusion matrices and F1-score parameter heatmaps specifically dissecting where Riemannian features excel (or fail).

**Why was it done?**
To establish the absolute state-of-the-art pipeline for this repository. It scientifically answers the question: *Is it better to deeply optimize a traditional CSP architecture (Notebook 6) or simply apply Riemannian Geometry out-of-the-box (Notebook 7)?* It wraps up the ML project by providing clear, empirical evidence natively visualizing the overall architecture efficacy. 
