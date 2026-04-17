### Cell 0 (code)
```python
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import load_epoched_bands
from src.models import (
    build_svm, build_lda, build_rf, build_csp_pipeline,
    evaluate_model, save_model,
    save_metrics, run_all_visualizations
)

```

### Cell 1 (code)
```python
subjects      = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
FEATURES_PATH = '../data/features/'
MODELS_PATH   = '../results/models/baseline/'
METRICS_PATH  = '../results/metrics/baseline/'
FIGURES_PATH  = '../results/figures/training/baseline/'
CLASS_NAMES   = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
```

### Cell 2 (code)
```python
all_results = {}

for subject_id in subjects:
    print(f"\n{'='*50}")
    print(f"  {subject_id}")
    print(f"{'='*50}")

    # Load raw epochs instead of pre-CSPed features
    X, y = load_epoched_bands(subject_id, load_path='../data/processed/')

    models = {
        'SVM': build_csp_pipeline(build_svm()),
        'LDA': build_csp_pipeline(build_lda()),
        'RF':  build_csp_pipeline(build_rf())
    }

    all_results[subject_id] = {}

    for model_name, model in models.items():
        print(f"\n  Training {model_name} (CSP inside CV)...")
        result = evaluate_model(model, X, y)
        print(f"  Accuracy: {result['mean']*100:.1f}% (+/- {result['std']*100:.1f}%)")
        print(f"  Per fold: {np.round(result['scores']*100, 1)}")

        model.fit(X, y)
        save_model(model, subject_id, model_name, save_path=MODELS_PATH)
        all_results[subject_id][model_name] = result

print("\nAll subjects done.")

```

### Cell 3 (code)
```python
# Save all metrics to CSV files
acc_df, f1_df, fold_df = save_metrics(all_results, subjects, save_path=METRICS_PATH)
```

### Cell 4 (code)
```python
# Print accuracy summary table
print("ACCURACY SUMMARY")
print("="*70)
rows = []
for subject_id in subjects:
    row = {'Subject': subject_id}
    for model_name in ['SVM', 'LDA', 'RF']:
        mean = all_results[subject_id][model_name]['mean']
        std  = all_results[subject_id][model_name]['std']
        row[model_name] = f"{mean*100:.1f}% (+/-{std*100:.1f}%)"
    rows.append(row)

results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))

print()
print("AVERAGE ACROSS ALL SUBJECTS")
print("-"*40)
for model_name in ['SVM', 'LDA', 'RF']:
    avg = np.mean([all_results[s][model_name]['mean'] for s in subjects])
    std = np.std([all_results[s][model_name]['mean'] for s in subjects])
    print(f"  {model_name}: {avg*100:.1f}% (+/- {std*100:.1f}%)")
print(f"  Chance level: 25.0%")
```

### Cell 5 (code)
```python
# Best and worst subjects
print("BEST MODEL PER SUBJECT")
print("-"*40)
for subject_id in subjects:
    accs       = {m: all_results[subject_id][m]['mean'] for m in ['SVM', 'LDA', 'RF']}
    best_model = max(accs, key=accs.get)
    worst_model = min(accs, key=accs.get)
    print(f"  {subject_id}: best={best_model} ({accs[best_model]*100:.1f}%)  worst={worst_model} ({accs[worst_model]*100:.1f}%)")

print()
best_overall  = max(subjects, key=lambda s: all_results[s]['SVM']['mean'])
worst_overall = min(subjects, key=lambda s: all_results[s]['SVM']['mean'])
print(f"Best overall subject (SVM):  {best_overall} — {all_results[best_overall]['SVM']['mean']*100:.1f}%")
print(f"Worst overall subject (SVM): {worst_overall} — {all_results[worst_overall]['SVM']['mean']*100:.1f}%")
```

### Cell 6 (code)
```python
# Generate all visualizations
run_all_visualizations(all_results, subjects, figures_path=FIGURES_PATH)
```

