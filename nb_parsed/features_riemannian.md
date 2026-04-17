### Cell 0 (code)
```python
import subprocess
result = subprocess.run(
    ['pip', 'install', 'pyriemann', '--quiet'],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
print("pyriemann installed")
```

### Cell 1 (code)
```python
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.decomposition import PCA

from src.riemannian import (
    extract_riemannian_features_offline,   # renamed — visualisation only
    load_riemannian_epochs,                # new — used by training notebook
    build_riemannian_svm,
    build_riemannian_lda,
    build_riemannian_rf,
    visualize_riemannian,
    CLASS_NAMES
)

```

### Cell 2 (code)
```python
import os

subjects     = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
LOAD_PATH    = '../data/processed/'
SAVE_PATH    = '../data/features_riemannian/'
FIGURES_PATH = '../results/figures/features/riemannian/'

os.makedirs(SAVE_PATH,    exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

print("Paths ready")
```

### Cell 3 (code)
```python
riemannian_objects = {}

for subject_id in subjects:
    print(f"\n{'='*50}")
    print(f"  {subject_id}")
    print(f"{'='*50}")

    epochs_final = mne.read_epochs(
        f'{LOAD_PATH}{subject_id}_clean_epo.fif',
        preload=True,
        verbose=False
    )

    # offline extractor — for visualisation only, not for CV accuracy
    X_ts, y, X_cov, cov_est, ts, scaler, le = extract_riemannian_features_offline(
        epochs_final
    )

    # visualise (PCA scatter, covariance heatmaps — fine to use full-fit features)
    visualize_riemannian(X_ts, X_cov, y, subject_id,
                         epochs_final.info, figures_path=FIGURES_PATH)

    # NOTE: we do NOT save X_ts to .npz anymore — training loads raw epochs
    # and fits inside CV. The .npz files from previous runs should be deleted
    # to avoid accidentally loading stale leaky features.

    riemannian_objects[subject_id] = {
        'X_ts':  X_ts,    # offline only — visualisation
        'X_cov': X_cov,
        'y':     y,
        'info':  epochs_final.info
    }

    print(f"  Done: {subject_id}")

print("\nAll subjects done.")

```

### Cell 4 (code)
```python
# Cross subject PCA scatter comparison

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
colors    = ['blue', 'red', 'green', 'purple']

for ax, subject_id in zip(axes.flat, subjects):
    X    = riemannian_objects[subject_id]['X_ts']
    y    = riemannian_objects[subject_id]['y']
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_.sum() * 100

    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = y == cls_idx
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=cls_name, alpha=0.6, s=15)

    ax.set_title(f'{subject_id} ({var:.0f}% var)', fontsize=11, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if subject_id == 'A01T':
        ax.legend(fontsize=7)

plt.suptitle('Riemannian Features PCA — All 9 Subjects',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_PATH}all_subjects_riemannian_scatter.png', dpi=100)
plt.show()
```

### Cell 5 (code)
```python
import os
import glob

print("Cleaning up obsolete leaky Riemannian .npz files...")

stale_files = glob.glob('../data/features_riemannian/*.npz')
if len(stale_files) == 0:
    print("  No stale .npz files found. Clean!")
else:
    for f in stale_files:
        os.remove(f)
        print(f"  Deleted {f}")
    print("\nSuccessfully purged old offline features.")

```

