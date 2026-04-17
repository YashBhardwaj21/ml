### Cell 0 (code)
```python
import subprocess
subprocess.run(['pip', 'install', 'imbalanced-learn', '--break-system-packages'], 
               capture_output=True)
print("imbalanced-learn installed")
```

### Cell 1 (code)
```python
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # fix: SMOTE inside CV folds only
import pickle
import os

from src.features import load_epoched_bands
from src.models import build_csp_pipeline, evaluate_model, save_model, save_metrics, run_all_visualizations

```

### Cell 2 (code)
```python
subjects      = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
FEATURES_PATH = '../data/features/'
MODELS_PATH   = '../results/models/improved/'
METRICS_PATH  = '../results/metrics/improved/'
FIGURES_PATH  = '../results/figures/training/improved/'
CLASS_NAMES   = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

os.makedirs(MODELS_PATH,  exist_ok=True)
os.makedirs(METRICS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)
```

### Cell 3 (code)
```python
# Load baseline results for comparison later
baseline_results = {}

for subject_id in subjects:
    baseline_results[subject_id] = {}
    for model_name in ['SVM', 'LDA', 'RF']:
        path = f'../results/models/baseline/{subject_id}_{model_name}.pkl'
        with open(path, 'rb') as f:
            model = pickle.load(f)
        baseline_results[subject_id][model_name] = model

print("Baseline models loaded successfully")
```

### Cell 4 (code)
```python
from sklearn.pipeline import Pipeline, FeatureUnion
from mne.decoding import CSP
from src.models import BandExtractor
from sklearn.preprocessing import StandardScaler

def make_smote_csp_pipeline(clf):
    """Chain CSP FeatureUnion and SMOTE correctly."""
    # ImbPipeline raises TypeError if intermediate step is an sklearn.pipeline.Pipeline.
    # We flatten the pipeline structure so FeatureUnion and StandardScaler are native.
    return ImbPipeline([
        ('features', FeatureUnion([
            ('mu', Pipeline([
                ('extract', BandExtractor(0)),
                ('csp', CSP(n_components=4, reg=0.05, log=True, norm_trace=False))
            ])),
            ('beta', Pipeline([
                ('extract', BandExtractor(1)),
                ('csp', CSP(n_components=4, reg=0.05, log=True, norm_trace=False))
            ]))
        ])),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf',   clf)
    ])


def tune_svm(X, y):
    param_grid = {
        'clf__C':     [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    base_svm = SVC(kernel='rbf', class_weight='balanced',
                   probability=True, random_state=42)
    pipe = make_smote_csp_pipeline(base_svm)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv,
                        scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X, y)
    print(f"  Best SVM params: {grid.best_params_}")
    print(f"  Best CV score:   {grid.best_score_*100:.1f}%")
    return grid.best_estimator_


def build_ensemble():
    return VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', class_weight='balanced',
                        probability=True, random_state=42)),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr',
                                               shrinkage='auto')),
            ('rf',  RandomForestClassifier(n_estimators=200,
                                           class_weight='balanced',
                                           random_state=42))
        ],
        voting='soft'
    )


def apply_smote(X, y):
    """Info only."""
    print(f"  Before SMOTE: {np.bincount(y)}")
    sm = SMOTE(random_state=42, k_neighbors=3)
    try:
       X_bal, y_bal = sm.fit_resample(X, y)
       print(f"  After SMOTE:  {np.bincount(y_bal)}")
    except:
       print("  Skipping detailed SMOTE display due to dimensions.")
    return X, y

print("Improvement functions defined")

```

### Cell 5 (code)
```python
# Train improved models on all subjects

improved_results = {}

for subject_id in subjects:
    print(f"\n{'='*50}")
    print(f"  {subject_id}")
    print(f"{'='*50}")

    X, y = load_epoched_bands(subject_id, load_path='../data/processed/')
    print(f"  Class distribution (raw): {np.bincount(y)}")

    improved_results[subject_id] = {}

    print("\n  Tuning SVM...")
    svm_pipe   = tune_svm(X, y)
    result_svm = evaluate_model(svm_pipe, X, y)
    print(f"  Tuned SVM accuracy: {result_svm['mean']*100:.1f}% (+/- {result_svm['std']*100:.1f}%)")
    svm_pipe.fit(X, y)
    save_model(svm_pipe, subject_id, 'SVM_tuned', save_path=MODELS_PATH)
    improved_results[subject_id]['SVM_tuned'] = result_svm

    print("\n  Training LDA...")
    lda_pipe   = make_smote_csp_pipeline(
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    result_lda = evaluate_model(lda_pipe, X, y)
    print(f"  LDA accuracy: {result_lda['mean']*100:.1f}% (+/- {result_lda['std']*100:.1f}%)")
    lda_pipe.fit(X, y)
    save_model(lda_pipe, subject_id, 'LDA_improved', save_path=MODELS_PATH)
    improved_results[subject_id]['LDA_improved'] = result_lda

    print("\n  Training RF...")
    rf_pipe   = make_smote_csp_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    )
    result_rf = evaluate_model(rf_pipe, X, y)
    print(f"  RF accuracy: {result_rf['mean']*100:.1f}% (+/- {result_rf['std']*100:.1f}%)")
    rf_pipe.fit(X, y)
    save_model(rf_pipe, subject_id, 'RF_improved', save_path=MODELS_PATH)
    improved_results[subject_id]['RF_improved'] = result_rf

    print("\n  Training Ensemble...")
    ens_pipe   = make_smote_csp_pipeline(build_ensemble())
    result_ens = evaluate_model(ens_pipe, X, y)
    print(f"  Ensemble accuracy: {result_ens['mean']*100:.1f}% (+/- {result_ens['std']*100:.1f}%)")
    ens_pipe.fit(X, y)
    save_model(ens_pipe, subject_id, 'Ensemble', save_path=MODELS_PATH)
    improved_results[subject_id]['Ensemble'] = result_ens

print("\nAll subjects done.")

```

### Cell 6 (code)
```python
# Accuracy summary — improved models

print("IMPROVED MODEL ACCURACY SUMMARY")
print("="*80)

rows = []
for subject_id in subjects:
    row = {'Subject': subject_id}
    for model_name in ['SVM_tuned', 'LDA_improved', 'RF_improved', 'Ensemble']:
        mean = improved_results[subject_id][model_name]['mean']
        std  = improved_results[subject_id][model_name]['std']
        row[model_name] = f"{mean*100:.1f}% (+/-{std*100:.1f}%)"
    rows.append(row)

# display table (human-readable strings — for screen only, not saved)
display_df = pd.DataFrame(rows)
print(display_df.to_string(index=False))

print()
print("AVERAGE ACROSS ALL SUBJECTS")
print("-"*40)
for model_name in ['SVM_tuned', 'LDA_improved', 'RF_improved', 'Ensemble']:
    avg = np.mean([improved_results[s][model_name]['mean'] for s in subjects])
    std = np.std([improved_results[s][model_name]['mean'] for s in subjects])
    print(f"  {model_name}: {avg*100:.1f}% (+/- {std*100:.1f}%)")

# save clean float CSVs through the shared utility — no string pollution
acc_df, f1_df, fold_df = save_metrics(
    improved_results, subjects, save_path=METRICS_PATH
)

```

### Cell 7 (code)
```python
# Side by side comparison — baseline vs improved (SVM vs SVM_tuned)

print("BASELINE vs IMPROVED COMPARISON (SVM)")
print("="*60)

comparison_rows = []
for subject_id in subjects:
    # Load baseline accuracy from saved CSV
    baseline_csv = pd.read_csv('../results/metrics/baseline/accuracy_summary.csv')
    baseline_row = baseline_csv[baseline_csv['Subject'] == subject_id].iloc[0]
    baseline_acc = baseline_row['SVM_mean']

    improved_acc = improved_results[subject_id]['SVM_tuned']['mean'] * 100
    diff         = improved_acc - baseline_acc

    comparison_rows.append({
        'Subject':      subject_id,
        'Baseline SVM': f"{baseline_acc:.1f}%",
        'Tuned SVM':    f"{improved_acc:.1f}%",
        'Improvement':  f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
    })
    print(f"  {subject_id}: {baseline_acc:.1f}% -> {improved_acc:.1f}% ({'+' if diff>=0 else ''}{diff:.1f}%)")

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(f'{METRICS_PATH}baseline_vs_improved.csv', index=False)
```

### Cell 8 (code)
```python
# Visual comparison — baseline vs improved

baseline_csv  = pd.read_csv('../results/metrics/baseline/accuracy_summary.csv')
baseline_svm  = baseline_csv['SVM_mean'].values
improved_svm  = [improved_results[s]['SVM_tuned']['mean'] * 100 for s in subjects]
improved_ens  = [improved_results[s]['Ensemble']['mean']  * 100 for s in subjects]

x     = np.arange(len(subjects))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, baseline_svm, width, label='Baseline SVM',  color='steelblue', alpha=0.7)
ax.bar(x,         improved_svm, width, label='Tuned SVM',     color='darkorange', alpha=0.7)
ax.bar(x + width, improved_ens, width, label='Ensemble',      color='seagreen',   alpha=0.7)

ax.axhline(y=25,  color='black', linestyle='--', linewidth=1, label='Chance (25%)')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Baseline vs Improved — All 9 Subjects')
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}baseline_vs_improved_comparison.png', dpi=100)
plt.show()
```

### Cell 9 (code)
```python
# Confusion matrices for best improved model (Ensemble) — all subjects

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

for ax, subject_id in zip(axes.flat, subjects):
    cm      = improved_results[subject_id]['Ensemble']['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        vmin=0, vmax=1
    )
    acc = improved_results[subject_id]['Ensemble']['mean'] * 100
    ax.set_title(f"{subject_id} — Ensemble ({acc:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Confusion Matrices — Ensemble — All 9 Subjects (Improved)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}confusion_matrices_ensemble_improved.png', dpi=100)
plt.show()
```

### Cell 10 (code)
```python
# F1 heatmap for improved models

fig, axes = plt.subplots(1, 4, figsize=(24, 7))

for ax, model_name in zip(axes, ['SVM_tuned', 'LDA_improved', 'RF_improved', 'Ensemble']):
    f1_matrix = []
    for subject_id in subjects:
        report = improved_results[subject_id][model_name]['report']
        f1_row = [report[cls]['f1-score'] for cls in CLASS_NAMES]
        f1_matrix.append(f1_row)

    sns.heatmap(
        np.array(f1_matrix),
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=CLASS_NAMES,
        yticklabels=subjects,
        ax=ax,
        vmin=0, vmax=1
    )
    avg = np.mean([improved_results[s][model_name]['mean'] for s in subjects])
    ax.set_title(f'{model_name}\nAvg: {avg*100:.1f}%')
    ax.set_xlabel('Class')
    ax.set_ylabel('Subject')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('F1 Score Heatmap — Improved Models — All Subjects',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}f1_heatmap_improved.png', dpi=100)
plt.show()
```

### Cell 11 (code)
```python
# verify no string pollution in saved CSV
verify = pd.read_csv(f'{METRICS_PATH}accuracy_summary.csv')
print("Column dtypes (all should be float64 or object for Subject only):")
print(verify.dtypes)
print()
print("First row values:")
print(verify.iloc[0])

# assert no column except Subject contains strings
for col in verify.columns:
    if col == 'Subject':
        continue
    assert verify[col].dtype in ['float64', 'int64'], \
        f"Column {col} is {verify[col].dtype} — string pollution detected!"

print("\nAll numeric columns confirmed clean.")

```

