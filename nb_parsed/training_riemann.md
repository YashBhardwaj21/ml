### Cell 0 (code)
```python
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.riemannian import load_riemannian_features
from src.models import (
    build_svm, build_lda, build_rf,
    evaluate_model, save_model,
    save_metrics, run_all_visualizations
)
```

### Cell 1 (code)
```python
subjects       = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
PROCESSED_PATH = '../data/processed/'          # ← raw clean epochs for pipeline CV
MODELS_PATH    = '../results/models/riemannian/'
METRICS_PATH   = '../results/metrics/riemannian/'
FIGURES_PATH   = '../results/figures/training/riemannian/'
CLASS_NAMES    = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

```

### Cell 2 (code)
```python
from src.riemannian import (
    build_riemannian_svm,
    build_riemannian_lda,
    build_riemannian_rf,
    load_riemannian_epochs
)

all_results = {}

for subject_id in subjects:
    print(f"\n{'='*50}")
    print(f"  {subject_id}")
    print(f"{'='*50}")

    # raw epochs array — no fitting has happened yet
    X, y, info = load_riemannian_epochs(subject_id, load_path=PROCESSED_PATH)
    print(f"  X shape: {X.shape}  →  (trials, channels, timepoints)")

    models = {
        'SVM': build_riemannian_svm(),
        'LDA': build_riemannian_lda(),
        'RF':  build_riemannian_rf()
    }

    all_results[subject_id] = {}

    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        result = evaluate_model(model, X, y)
        print(f"  Accuracy: {result['mean']*100:.1f}% (+/- {result['std']*100:.1f}%)")
        print(f"  Per fold: {np.round(result['scores']*100, 1)}")

        model.fit(X, y)   # final fit on all data for saving
        save_model(model, subject_id, f'Riemannian_{model_name}', save_path=MODELS_PATH)
        all_results[subject_id][model_name] = result

print("\nAll subjects done.")

```

### Cell 3 (code)
```python
print("RIEMANNIAN ACCURACY SUMMARY")
print("=" * 70)

# display table — human-readable strings for screen only
rows = []
for subject_id in subjects:
    row = {'Subject': subject_id}
    for model_name in ['SVM', 'LDA', 'RF']:
        mean = all_results[subject_id][model_name]['mean']
        std  = all_results[subject_id][model_name]['std']
        row[model_name] = f"{mean*100:.1f}% (+/-{std*100:.1f}%)"
    rows.append(row)

display_df = pd.DataFrame(rows)
print(display_df.to_string(index=False))

print()
print("AVERAGE ACROSS ALL SUBJECTS")
print("-" * 40)
for model_name in ['SVM', 'LDA', 'RF']:
    avg = np.mean([all_results[s][model_name]['mean'] for s in subjects])
    std = np.std([all_results[s][model_name]['mean'] for s in subjects])
    print(f"  {model_name}: {avg*100:.1f}% (+/- {std*100:.1f}%)")
print(f"  Chance level: 25.0%")

# save clean float CSVs through the shared utility — same as baseline and improved
acc_df, f1_df, fold_df = save_metrics(
    all_results, subjects, save_path=METRICS_PATH
)

```

### Cell 4 (code)
```python
# Compare CSP baseline vs Riemannian

baseline_csv   = pd.read_csv('../results/metrics/baseline/accuracy_summary.csv')
baseline_svm   = baseline_csv['SVM_mean'].values
riemannian_svm = [all_results[s]['SVM']['mean'] * 100 for s in subjects]

x     = np.arange(len(subjects))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, baseline_svm,   width, label='CSP + SVM (Baseline)',
       color='steelblue', alpha=0.8)
ax.bar(x + width/2, riemannian_svm, width, label='Riemannian + SVM',
       color='darkorange', alpha=0.8)

ax.axhline(y=25, color='black', linestyle='--', linewidth=1.5, label='Chance (25%)')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylabel('Accuracy (%)')
ax.set_title('CSP Baseline vs Riemannian Geometry — All 9 Subjects')
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}csp_vs_riemannian.png', dpi=100)
plt.show()
```

### Cell 5 (code)
```python
# Confusion matrices for Riemannian SVM

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

for ax, subject_id in zip(axes.flat, subjects):
    cm      = all_results[subject_id]['SVM']['cm']
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
    acc = all_results[subject_id]['SVM']['mean'] * 100
    ax.set_title(f"{subject_id} — Riemannian SVM ({acc:.1f}%)",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Confusion Matrices — Riemannian SVM — All 9 Subjects',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}confusion_matrices_riemannian_svm.png', dpi=100)
plt.show()
```

### Cell 6 (code)
```python
# Load CSVs — all pure floats now, no string parsing needed
baseline_csv = pd.read_csv('../results/metrics/baseline/accuracy_summary.csv')
improved_csv = pd.read_csv('../results/metrics/improved/accuracy_summary.csv')

# direct float column access — no extract_acc() needed
baseline_svm   = baseline_csv['SVM_mean'].values
improved_svm   = improved_csv['SVM_tuned_mean'].values        # ← matches new column name
riemannian_svm = [all_results[s]['SVM']['mean'] * 100 for s in subjects]

x     = np.arange(len(subjects))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width, baseline_svm,   width, label='CSP Baseline SVM',
       color='steelblue',  alpha=0.8)
ax.bar(x,         improved_svm,   width, label='CSP Improved SVM',
       color='darkorange', alpha=0.8)
ax.bar(x + width, riemannian_svm, width, label='Riemannian SVM',
       color='seagreen',   alpha=0.8)

ax.axhline(y=25, color='black', linestyle='--', linewidth=1.5, label='Chance (25%)')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Three-Way Comparison — CSP Baseline vs CSP Improved vs Riemannian')
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}three_way_comparison.png', dpi=100)
plt.show()

```

### Cell 7 (code)
```python
# F1 heatmap for Riemannian models

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, model_name in zip(axes, ['SVM', 'LDA', 'RF']):
    f1_matrix = []
    for subject_id in subjects:
        report = all_results[subject_id][model_name]['report']
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
    avg = np.mean([all_results[s][model_name]['mean'] for s in subjects])
    ax.set_title(f'Riemannian {model_name}\nAvg: {avg*100:.1f}%')
    ax.set_xlabel('Class')
    ax.set_ylabel('Subject')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('F1 Score Heatmap — Riemannian Models — All Subjects',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}f1_heatmap_riemannian.png', dpi=100)
plt.show()
```

### Cell 8 (code)
```python
# Save metrics
save_metrics(all_results, subjects, save_path=METRICS_PATH)
run_all_visualizations(all_results, subjects, figures_path=FIGURES_PATH)
```

