### Cell 0 (code)
```python
import sys
sys.path.append('..')
from src.preprocessing import preprocess_subject, visualize_subject
```

### Cell 1 (code)
```python
subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']

DATA_PATH    = '../data/raw/'
SAVE_PATH    = '../data/processed/'
FIGURES_PATH = '../results/figures/preprocessing/'

results = {}
```

### Cell 2 (code)
```python
FIGURES_PATH = '../results/figures/preprocessing/'

for subject_id in subjects:
    try:
        raw, raw_clean, epochs, epochs_final = preprocess_subject(
            subject_id,
            data_path=DATA_PATH,
            save_path=SAVE_PATH
        )

        visualize_subject(
            raw, raw_clean, epochs, epochs_final,
            subject_id,
            figures_path=FIGURES_PATH
        )

        results[subject_id] = {
            'status':       'success',
            'total_trials': len(epochs_final),
            'Left Hand':    len(epochs_final['Left Hand']),
            'Right Hand':   len(epochs_final['Right Hand']),
            'Feet':         len(epochs_final['Feet']),
            'Tongue':       len(epochs_final['Tongue'])
        }

    except Exception as e:
        print(f"FAILED: {subject_id} - {e}")
        results[subject_id] = {'status': 'failed', 'error': str(e)}
```

### Cell 3 (code)
```python
import pandas as pd

rows = []
for subject_id, result in results.items():
    if result['status'] == 'success':
        rows.append({
            'Subject':    subject_id,
            'Status':     'success',
            'Total':      result['total_trials'],
            'Left Hand':  result['Left Hand'],
            'Right Hand': result['Right Hand'],
            'Feet':       result['Feet'],
            'Tongue':     result['Tongue'],
            'Dropped':    288 - result['total_trials']
        })
    else:
        rows.append({
            'Subject':    subject_id,
            'Status':     'failed',
            'Total':      0,
            'Left Hand':  0,
            'Right Hand': 0,
            'Feet':       0,
            'Tongue':     0,
            'Dropped':    288
        })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
```

### Cell 4 (code)
```python
import matplotlib.pyplot as plt
import numpy as np

successful = summary_df[summary_df['Status'] == 'success']
x     = np.arange(len(successful))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width*1.5, successful['Left Hand'],  width, label='Left Hand',  color='blue')
ax.bar(x - width*0.5, successful['Right Hand'], width, label='Right Hand', color='red')
ax.bar(x + width*0.5, successful['Feet'],       width, label='Feet',       color='green')
ax.bar(x + width*1.5, successful['Tongue'],     width, label='Tongue',     color='purple')
ax.axhline(y=30, color='black', linestyle='--', linewidth=1, label='Minimum (30)')
ax.set_xticks(x)
ax.set_xticklabels(successful['Subject'])
ax.set_ylabel('Clean Trials')
ax.set_title('Clean Trial Count Per Class Per Subject — All 9 Subjects')
ax.legend()
plt.tight_layout()
plt.savefig('../results/figures/preprocessing/all_subjects_trial_summary.png', dpi=100)
plt.show()
```

