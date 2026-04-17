### Cell 0 (code)
```python
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mne

from src.features import (
    extract_csp_features,
    save_features,
    load_features,
    visualize_features
)
```

### Cell 1 (code)
```python
subjects     = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
LOAD_PATH    = '../data/processed/'
SAVE_PATH    = '../data/features/'
FIGURES_PATH = '../results/figures/features/csp/'
```

### Cell 2 (code)
```python
csp_objects = {}

for subject_id in subjects:
    print(f"\n{'='*50}")
    print(f"  {subject_id}")
    print(f"{'='*50}")

    epochs_final = mne.read_epochs(
        f'{LOAD_PATH}{subject_id}_clean_epo.fif',
        preload=True,
        verbose=False
    )

    X_csp, y, csp_mu, csp_beta, scaler, le = extract_csp_features(epochs_final, n_components=4)
    save_features(subject_id, X_csp, y, save_path=SAVE_PATH)
    visualize_features(X_csp, y, csp_mu, epochs_final.info, subject_id, figures_path=FIGURES_PATH)

    csp_objects[subject_id] = {
        'csp_mu':   csp_mu,
        'csp_beta': csp_beta,
        'scaler':   scaler,
        'le':       le,
        'X':        X_csp,
        'y':        y
    }

    print(f"Done: {subject_id}")
```

### Cell 3 (code)
```python
fig, axes = plt.subplots(9, 8, figsize=(28, 36))

for row, subject_id in enumerate(subjects):
    epochs_info = mne.read_epochs(
        f'{LOAD_PATH}{subject_id}_clean_epo.fif',
        preload=False,
        verbose=False
    ).info

    csp_mu   = csp_objects[subject_id]['csp_mu']
    csp_beta = csp_objects[subject_id]['csp_beta']

    for col in range(4):
        mne.viz.plot_topomap(
            csp_mu.patterns_[col],
            epochs_info,
            axes=axes[row, col],
            show=False,
            contours=4
        )
        mne.viz.plot_topomap(
            csp_beta.patterns_[col],
            epochs_info,
            axes=axes[row, col + 4],
            show=False,
            contours=4
        )
        if row == 0:
            axes[row, col].set_title(f'Mu P{col+1}', fontsize=10)
            axes[row, col+4].set_title(f'Beta P{col+1}', fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel(subject_id, fontsize=9)

plt.suptitle('CSP Patterns — Mu Band (left) vs Beta Band (right) — All 9 Subjects',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}all_subjects_csp_patterns.png', dpi=100)
plt.show()
```

### Cell 4 (code)
```python
# Cross-subject feature separability comparison
# Shows how well CSP features separate classes for each subject

import numpy as np

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

for ax, subject_id in zip(axes.flat, subjects):
    X = csp_objects[subject_id]['X']
    y = csp_objects[subject_id]['y']

    colors = ['blue', 'red', 'green', 'purple']
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = y == cls_idx
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=color, label=cls_name, alpha=0.6, s=20)

    ax.set_title(f'{subject_id}', fontsize=11, fontweight='bold')
    ax.set_xlabel('CSP Feature 1')
    ax.set_ylabel('CSP Feature 2')
    if subject_id == 'A01T':
        ax.legend(fontsize=7)

plt.suptitle('CSP Feature Scatter — Feature 1 vs Feature 2 — All Subjects',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}all_subjects_csp_scatter.png', dpi=100)
plt.show()
```

### Cell 5 (code)
```python
# Verify all feature files saved correctly
import os

print("Feature files in data/features/:")
print()

for subject_id in subjects:
    path = f'../data/features/{subject_id}_features.npz'
    if os.path.exists(path):
        X, y = load_features(subject_id, load_path=SAVE_PATH)
    else:
        print(f"  {subject_id} - MISSING")
```

