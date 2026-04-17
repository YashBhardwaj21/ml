### Cell 0 (markdown)
# BCI Competition IV Dataset 2a - Deep Exploratory Data Analysis (EDA)

### Cell 1 (code)
```python
!pip install mne matplotlib numpy pandas scipy scikit-learn seaborn
```

### Cell 2 (code)
```python
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from mne.decoding import CSP

mne.set_log_level('WARNING')  # reduce verbose output
print(f"MNE Version: {mne.__version__}")
print("All imports successful!")
```

### Cell 3 (code)
```python
DATA_PATH = 'data/'
SUBJECT = 'A01T'
file_path = f'{DATA_PATH}{SUBJECT}.gdf'

raw = mne.io.read_raw_gdf(
    file_path, 
    preload=True,
    eog=['EOG-left', 'EOG-central', 'EOG-right']
)

print(" File loaded successfully!")
```

### Cell 4 (code)
```python
print("=" * 50)
print("        DATASET OVERVIEW")
print("=" * 50)
print(f"Subject          : {SUBJECT}")
print(f"Sampling Rate    : {raw.info['sfreq']} Hz")
print(f"Total Channels   : {len(raw.ch_names)}")
print(f"Duration         : {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.1f} mins)")
print(f"Total Timepoints : {len(raw.times)}")
print()
print("EEG Channels (22):")
print(raw.ch_names[:22])
print()
print("EOG Channels (3):")
print(raw.ch_names[22:])
```

### Cell 5 (code)
```python
# First look at raw annotations before converting to events
print("Raw annotations in file:")
print(raw.annotations)
print()
print(f"Total annotations: {len(raw.annotations)}")
print()

# See unique annotation types
unique_annotations = set(raw.annotations.description)
print("Unique annotation types found:")
for ann in sorted(unique_annotations):
    count = sum(raw.annotations.description == ann)
    print(f"  '{ann}' — appears {count} times")
```

### Cell 6 (code)
```python
events, event_id = mne.events_from_annotations(raw)

print(f"Total events: {len(events)}")
print()
print("event_id dictionary (what MNE extracted):")
for key, val in event_id.items():
    print(f"  '{key}' → internal code {val}")
print()
print("First 20 events (sample, prev_id, event_id):")
print(events[:20])
```

### Cell 7 (code)
```python
print("Event counts:")
for key, val in event_id.items():
    mask = events[:, 2] == val
    count = np.sum(mask)
    times_of_event = events[mask, 0] / raw.info['sfreq']
    print(f"\n  Event '{key}' (code {val}):")
    print(f"    Count: {count}")
    if count > 0:
        print(f"    First occurrence: {times_of_event[0]:.2f}s")
        print(f"    Last occurrence:  {times_of_event[-1]:.2f}s")
        print(f"    Avg gap between: {np.mean(np.diff(times_of_event)):.2f}s" if count > 1 else "")
```

### Cell 8 (code)
```python
fig, ax = plt.subplots(figsize=(15, 4))

mi_events = {k: v for k, v in event_id.items() if k in ['769', '770', '771', '772']}
colors = {'769': 'blue', '770': 'red', '771': 'green', '772': 'purple'}
labels_map = {'769': 'Left Hand', '770': 'Right Hand', '771': 'Feet', '772': 'Tongue'}

for key, val in mi_events.items():
    mask = events[:, 2] == val
    times_ev = events[mask, 0] / raw.info['sfreq']
    ax.scatter(times_ev, [val] * sum(mask), 
               c=colors[key], label=labels_map[key], alpha=0.7, s=30)

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Event Code', fontsize=12)
ax.set_title('Trial Timeline — When Each Motor Imagery Cue Occurred', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
```

### Cell 9 (code)
```python
raw_filtered = raw.copy().filter(7., 30., fir_design='firwin', verbose=False)
print(f"✅ Bandpass filter applied (7-30Hz)")
print(f"raw_filtered max: {raw_filtered.get_data(picks='eeg').max()*1e6:.2f} µV")
print(f"raw_filtered min: {raw_filtered.get_data(picks='eeg').min()*1e6:.2f} µV")
```

### Cell 10 (code)
```python
channel_rename = {
    'EEG-Fz':     'Fz',
    'EEG-0':      'FC3',
    'EEG-1':      'FC1',
    'EEG-2':      'FCz',
    'EEG-3':      'FC2',
    'EEG-4':      'FC4',
    'EEG-5':      'C5',
    'EEG-C3':     'C3',
    'EEG-6':      'C1',
    'EEG-Cz':     'Cz',
    'EEG-7':      'C2',
    'EEG-C4':     'C4',
    'EEG-8':      'C6',
    'EEG-9':      'CP3',
    'EEG-10':     'CP1',
    'EEG-11':     'CPz',
    'EEG-12':     'CP2',
    'EEG-13':     'CP4',
    'EEG-14':     'P1',
    'EEG-Pz':     'Pz',
    'EEG-15':     'P2',
    'EEG-16':     'POz',
}

raw_filtered.rename_channels(channel_rename)

montage = mne.channels.make_standard_montage('standard_1020')
raw_filtered.set_montage(montage, on_missing='ignore')

print("✅ Montage set successfully!")
print("Channels after rename:")
print(raw_filtered.ch_names)
```

### Cell 11 (code)
```python
c3_idx = raw.ch_names.index([ch for ch in raw.ch_names if 'C3' in ch][0])
start, stop = raw.time_as_index([100, 110])

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

axes[0].plot(raw.times[start:stop],
             raw[c3_idx, start:stop][0][0] * 1e6,
             color='gray', linewidth=0.8)
axes[0].set_title('Raw EEG — Channel C3 (unfiltered, original)', fontsize=12)
axes[0].set_ylabel('Amplitude (µV)')

axes[1].plot(raw_filtered.times[start:stop],
             raw_filtered[c3_idx, start:stop][0][0] * 1e6,
             color='steelblue', linewidth=0.8)
axes[1].set_title('Filtered EEG (7-30Hz) — Channel C3', fontsize=12)
axes[1].set_ylabel('Amplitude (µV)')
axes[1].set_xlabel('Time (seconds)')

plt.suptitle('Effect of Bandpass Filtering on EEG Signal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Cell 12 (code)
```python
print("Fitting ICA... (this takes ~1 minute)")

ica = ICA(n_components=20, random_state=42, max_iter='auto')
ica.fit(raw_filtered)

print(f"✅ ICA fitted on 20 components")

# Check all 3 EOG channels
eog_indices_left, eog_scores_left = ica.find_bads_eog(raw_filtered,
                                                        ch_name='EOG-left',
                                                        verbose=False)
eog_indices_central, eog_scores_central = ica.find_bads_eog(raw_filtered,
                                                              ch_name='EOG-central',
                                                              verbose=False)
eog_indices_right, eog_scores_right = ica.find_bads_eog(raw_filtered,
                                                          ch_name='EOG-right',
                                                          verbose=False)

# Combine all unique bad components
all_eog_indices = list(set(eog_indices_left + eog_indices_central + eog_indices_right))
ica.exclude = all_eog_indices

print(f"EOG-left found:    components {eog_indices_left}")
print(f"EOG-central found: components {eog_indices_central}")
print(f"EOG-right found:   components {eog_indices_right}")
print(f"🗑️  Total unique components to remove: {all_eog_indices}")

# Plot scores separately (axes parameter not supported)
ica.plot_scores(eog_scores_left,    title='Correlation with EOG-left',    show=True)
ica.plot_scores(eog_scores_central, title='Correlation with EOG-central', show=True)
ica.plot_scores(eog_scores_right,   title='Correlation with EOG-right',   show=True)

# Plot properties of bad components
if len(all_eog_indices) > 0:
    ica.plot_properties(raw_filtered, picks=all_eog_indices)
    plt.show()
else:
    print("No EOG components found automatically")
```

### Cell 13 (code)
```python
# Apply ICA and create clean signal
raw_clean = raw_filtered.copy()
ica.apply(raw_clean)

# Verify cleaning worked
print(f"✅ ICA applied — removed components {all_eog_indices}")
print()
print(f"Before ICA: max={raw_filtered.get_data(picks='eeg').max()*1e6:.2f} µV  "
      f"min={raw_filtered.get_data(picks='eeg').min()*1e6:.2f} µV")
print(f"After ICA:  max={raw_clean.get_data(picks='eeg').max()*1e6:.2f} µV  "
      f"min={raw_clean.get_data(picks='eeg').min()*1e6:.2f} µV")
```

### Cell 14 (code)
```python
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

# Stage 1 - Raw (dirty)
psd_raw = raw.compute_psd(fmin=1, fmax=50, picks='eeg')
psd_raw.plot(axes=axes[0], show=False)
axes[0].axvspan(8, 13, alpha=0.15, color='green', label='Mu (8-13Hz)')
axes[0].axvspan(13, 30, alpha=0.15, color='orange', label='Beta (13-30Hz)')
axes[0].axvspan(1, 4, alpha=0.1, color='blue', label='Delta (1-4Hz)')
axes[0].axvspan(4, 8, alpha=0.1, color='purple', label='Theta (4-8Hz)')
axes[0].set_title('Stage 1 — Raw EEG (with artifacts, -1600µV spikes)', fontsize=12)
axes[0].legend(loc='upper right', fontsize=8)

# Stage 2 - Filtered
psd_filtered = raw_filtered.compute_psd(fmin=1, fmax=50, picks='eeg')
psd_filtered.plot(axes=axes[1], show=False)
axes[1].axvspan(8, 13, alpha=0.15, color='green')
axes[1].axvspan(13, 30, alpha=0.15, color='orange')
axes[1].axvspan(1, 4, alpha=0.1, color='blue')
axes[1].axvspan(4, 8, alpha=0.1, color='purple')
axes[1].set_title('Stage 2 — After Bandpass Filter (7-30Hz)', fontsize=12)

# Stage 3 - Clean (after ICA)
psd_clean = raw_clean.compute_psd(fmin=1, fmax=50, picks='eeg')
psd_clean.plot(axes=axes[2], show=False)
axes[2].axvspan(8, 13, alpha=0.15, color='green')
axes[2].axvspan(13, 30, alpha=0.15, color='orange')
axes[2].axvspan(1, 4, alpha=0.1, color='blue')
axes[2].axvspan(4, 8, alpha=0.1, color='purple')
axes[2].set_title('Stage 3 — After ICA Artifact Removal', fontsize=12)

plt.suptitle('Power Spectral Density — All Three Preprocessing Stages', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Cell 15 (code)
```python
c3_idx = raw_filtered.ch_names.index('C3')
start, stop = raw_filtered.time_as_index([50, 60])

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

axes[0].plot(raw_filtered.times[start:stop],
             raw_filtered[c3_idx, start:stop][0][0] * 1e6,
             color='tomato', linewidth=0.8)
axes[0].set_title('Before ICA (with eye artifacts)', fontsize=12)
axes[0].set_ylabel('Amplitude (µV)')

axes[1].plot(raw_clean.times[start:stop],
             raw_clean[c3_idx, start:stop][0][0] * 1e6,
             color='seagreen', linewidth=0.8)
axes[1].set_title('After ICA (artifacts removed)', fontsize=12)
axes[1].set_ylabel('Amplitude (µV)')
axes[1].set_xlabel('Time (seconds)')

plt.suptitle('ICA Artifact Removal — Before vs After', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Cell 16 (code)
```python
events, event_id = mne.events_from_annotations(raw)

# Use all 4 classes
mi_event_id = {
    'Left Hand':  event_id['769'],
    'Right Hand': event_id['770'],
    'Feet':       event_id['771'],
    'Tongue':     event_id['772']
}

# -0.5 to 4.5 seconds around each cue (4 second MI window)
epochs = mne.Epochs(
    raw_clean, events,
    event_id=mi_event_id,
    tmin=-0.5, tmax=4.5,
    baseline=(-0.5, 0),
    preload=True
    # NO reject here — we learned MNE's auto reject doesn't work 
    # correctly with this dataset's scaling
)

print(f"✅ Epochs created")
print(f"Total epochs: {len(epochs)}")
print(f"Epoch shape: {epochs.get_data().shape}")
print()

# Check actual amplitudes BEFORE manual rejection
ep_data = epochs.get_data() * 1e6
trial_max = ep_data.max(axis=(1,2))
print(f"Max amplitude in epochs: {trial_max.max():.2f} µV")
print(f"Trials exceeding 100µV:  {(trial_max > 100).sum()}")
print(f"Trials exceeding 150µV:  {(trial_max > 150).sum()}")
print()

# Manual rejection at 100µV (based on histogram bimodal gap we found)
clean_mask = trial_max <= 100
epochs_final = epochs[clean_mask]

print(f"Trials before rejection: {len(epochs)}")
print(f"Trials after rejection:  {len(epochs_final)}")
print(f"Trials dropped:          {len(epochs) - len(epochs_final)}")
print()
print("Final trials per class:")
for label in mi_event_id:
    print(f"  {label}: {len(epochs_final[label])} trials")
```

### Cell 17 (code)
```python
epoch_data = epochs.get_data() * 1e6  # shape: (288, 22, timepoints)

print("Epoch amplitude statistics (post-rejection):")
print(f"Max amplitude across all epochs: {epoch_data.max():.2f} µV")
print(f"Min amplitude across all epochs: {epoch_data.min():.2f} µV")
print(f"Mean std across all trials: {epoch_data.std(axis=2).mean():.2f} µV")
print()

# Check if any trial has suspiciously high amplitude
trial_max = epoch_data.max(axis=(1,2))  # max per trial
trial_min = epoch_data.min(axis=(1,2))  # min per trial

suspicious = np.where(trial_max > 100)[0]
print(f"Trials with amplitude > 100µV: {len(suspicious)}")
if len(suspicious) > 0:
    print(f"Trial indices: {suspicious}")
```

### Cell 18 (code)
```python
epoch_data = epochs.get_data() * 1e6

trial_max = epoch_data.max(axis=(1,2))
trial_min = epoch_data.min(axis=(1,2))
trial_ptp = trial_max - trial_min  # peak to peak per trial

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.hist(trial_max, bins=40, color='tomato', edgecolor='black')
plt.axvline(100, color='black', linestyle='--', label='100µV threshold')
plt.xlabel('Max Amplitude (µV)')
plt.ylabel('Number of Trials')
plt.title('Distribution of Max Amplitude per Trial')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(trial_ptp, bins=40, color='steelblue', edgecolor='black')
plt.axvline(150, color='black', linestyle='--', label='150µV threshold')
plt.xlabel('Peak-to-Peak Amplitude (µV)')
plt.title('Distribution of Peak-to-Peak per Trial')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(sorted(trial_max), color='purple')
plt.axhline(100, color='red', linestyle='--', label='100µV')
plt.axhline(150, color='orange', linestyle='--', label='150µV')
plt.axhline(200, color='green', linestyle='--', label='200µV')
plt.xlabel('Trial (sorted)')
plt.ylabel('Max Amplitude (µV)')
plt.title('Sorted Max Amplitude — Find Natural Cutoff')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Trials > 100µV: {(trial_max > 100).sum()}")
print(f"Trials > 150µV: {(trial_max > 150).sum()}")
print(f"Trials > 200µV: {(trial_max > 200).sum()}")
print(f"Trials > 300µV: {(trial_max > 300).sum()}")
print()
print(f"Mean max amplitude: {trial_max.mean():.2f} µV")
print(f"Median max amplitude: {np.median(trial_max):.2f} µV")
```

### Cell 19 (code)
```python
# First visualize the amplitude distribution to pick a good threshold
epoch_data = epochs.get_data() * 1e6
trial_max = epoch_data.max(axis=(1,2))

plt.figure(figsize=(12, 5))
plt.hist(trial_max, bins=50, color='steelblue', edgecolor='white')
plt.axvline(100, color='orange', linestyle='--', linewidth=2, label='100 µV')
plt.axvline(150, color='red', linestyle='--', linewidth=2, label='150 µV')
plt.axvline(200, color='darkred', linestyle='--', linewidth=2, label='200 µV')
plt.xlabel('Max Amplitude per Trial (µV)', fontsize=12)
plt.ylabel('Number of Trials', fontsize=12)
plt.title('Trial Amplitude Distribution — Where to Draw the Rejection Line?', fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Total trials: {len(trial_max)}")
print(f"Surviving at 100µV threshold: {np.sum(trial_max <= 100)} trials")
print(f"Surviving at 150µV threshold: {np.sum(trial_max <= 150)} trials")
print(f"Surviving at 200µV threshold: {np.sum(trial_max <= 200)} trials")
print()

# Show per-class survival at 150µV
print("Surviving trials per class at 150µV threshold:")
for cls in epochs.event_id:
    cls_data = epochs[cls].get_data() * 1e6
    cls_max = cls_data.max(axis=(1,2))
    surviving = np.sum(cls_max <= 150)
    print(f"  {cls}: {surviving}/72 trials survive")
```

### Cell 20 (code)
```python
# Check if raw_clean actually has the ICA applied
# and if the amplitudes are different from raw
data_clean = raw_clean.get_data(picks='eeg') * 1e6
print(f"raw_clean max amplitude: {data_clean.max():.2f} µV")
print(f"raw_clean min amplitude: {data_clean.min():.2f} µV")
print()

# Compare with original raw
data_raw = raw.get_data(picks='eeg') * 1e6
print(f"raw max amplitude: {data_raw.max():.2f} µV")
print(f"raw min amplitude: {data_raw.min():.2f} µV")
```

### Cell 21 (code)
```python
# Visualize final class distribution
classes = list(epochs_final.event_id.keys())
counts = [len(epochs_final[cls]) for cls in classes]

plt.figure(figsize=(8, 5))
bars = plt.bar(classes, counts, color=['blue', 'red', 'green', 'purple'], 
               edgecolor='white', width=0.5)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', fontsize=12, fontweight='bold')

plt.axhline(y=30, color='orange', linestyle='--', linewidth=1.5, 
            label='Minimum recommended (30)')
plt.ylabel('Number of Clean Trials', fontsize=12)
plt.title('Final Clean Trial Count Per Class', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
```

### Cell 22 (code)
```python
# Save clean epochs
epochs_final.save('data/A01T_clean_epo.fif', overwrite=True)
print("✅ Saved to data/A01T_clean_epo.fif")

# Final summary
print()
print("=" * 45)
print("   PREPROCESSING COMPLETE — FINAL SUMMARY")
print("=" * 45)
print(f"Started with   : 288 trials")
print(f"After cleaning : 164 trials")
print(f"Removed        : 124 trials (43%)")
print(f"Shape          : {epochs_final.get_data().shape}")
print(f"               : (trials × channels × timepoints)")
print(f"Sampling rate  : {epochs_final.info['sfreq']} Hz")
print(f"Epoch window   : -0.5s to 4.5s")
print(f"Channels       : 22 EEG + 3 EOG = 25 total")
print()
print("Class balance:")
for cls, count in zip(classes, counts):
    bar = '█' * count
    print(f"  {cls:12}: {bar} {count}")
print()
print("Next step: CSP Feature Extraction + SVM Classification")
```

### Cell 23 (code)
```python
import seaborn as sns

# Get only EEG channels (first 22), exclude EOG
epoch_data = epochs_final.get_data(picks='eeg') * 1e6  # (164, 22, 1251)

# Mean power across time per trial per channel
epoch_mean = epoch_data.mean(axis=2)  # (164, 22)

# Correlation across trials
corr_matrix = np.corrcoef(epoch_mean.T)  # (22, 22)

plt.figure(figsize=(14, 11))
sns.heatmap(
    corr_matrix,
    xticklabels=epochs_final.ch_names[:22],
    yticklabels=epochs_final.ch_names[:22],
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 7},
    linewidths=0.3
)
plt.title('EEG Channel Correlation — Clean Epochs\n(Computed on 164 artifact-free trials)', 
          fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()
```

### Cell 24 (code)
```python
# Recompute statistics on clean epochs only
epoch_data = epochs_final.get_data(picks='eeg') * 1e6  # (164, 22, 1251)

# Reshape to (22, all_timepoints_across_all_trials)
n_trials, n_channels, n_times = epoch_data.shape
reshaped = epoch_data.transpose(1, 0, 2).reshape(n_channels, -1)  # (22, 164*1251)

stats = []
for i, ch in enumerate(epochs_final.ch_names[:22]):
    stats.append({
        'Channel': ch,
        'Mean (µV)': round(np.mean(reshaped[i]), 4),
        'Std (µV)': round(np.std(reshaped[i]), 4),
        'Min (µV)': round(np.min(reshaped[i]), 2),
        'Max (µV)': round(np.max(reshaped[i]), 2),
        'Peak-to-Peak (µV)': round(np.ptp(reshaped[i]), 2)
    })

stats_df = pd.DataFrame(stats)
print("=" * 60)
print("   EEG CHANNEL STATISTICS — CLEAN EPOCHS ONLY")
print("=" * 60)
print(stats_df.to_string(index=False))
```

### Cell 25 (code)
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
colors = ['blue', 'red', 'green', 'purple']

# Use epochs_final (clean) and get C3 index from it
c3_idx = epochs_final.ch_names.index('C3')

for ax, cls, color in zip(axes.flat, classes, colors):
    # Use epochs_final not epochs
    data = epochs_final[cls].get_data()[:, c3_idx, :] * 1e6
    times = epochs_final.times

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Plot first 10 trials lightly
    for trial in data[:10]:
        ax.plot(times, trial, color=color, alpha=0.1, linewidth=0.5)

    # Plot mean and std band
    ax.plot(times, mean, color=color, linewidth=2, label='Mean')
    ax.fill_between(times, mean - std, mean + std, alpha=0.2, color=color)

    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Cue onset')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title(f'{cls} Imagery — Channel C3 ({len(epochs_final[cls])} trials)', 
                 fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend(fontsize=8)

plt.suptitle('Epoched EEG — Mean ± Std per Class — Clean Epochs Only', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Cell 26 (code)
```python
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Use epochs_final (clean data)
classes = list(epochs_final.event_id.keys())

for ax, cls in zip(axes, classes):
    epochs_final[cls].compute_psd(fmin=8, fmax=30).plot_topomap(
        axes=ax,
        ch_type='eeg',
        contours=4,
        bands={'Mu/Beta (8-30Hz)': (8, 30)},
        show=False
    )
    ax.set_title(f'{cls}\n(8-30Hz Power, n={len(epochs_final[cls])})', fontsize=11)

plt.suptitle('Scalp Topomaps — Motor Imagery Power Distribution (Clean Epochs)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Cell 27 (code)
```python
# Load the saved clean epochs from preprocessing
epochs_final = mne.read_epochs('data/A01T_clean_epo.fif', preload=True)

print("✅ Epochs loaded successfully!")
print(f"Shape: {epochs_final.get_data().shape}")
print(f"  → (trials × channels × timepoints)")
print()
print(f"Channels: {epochs_final.ch_names}")
print()
print(f"Event IDs: {epochs_final.event_id}")
print()
print("Trials per class:")
for cls in epochs_final.event_id:
    print(f"  {cls}: {len(epochs_final[cls])} trials")
```

### Cell 28 (code)
```python
# Use only EEG channels (first 22), exclude EOG (last 3)
X = epochs_final.get_data(picks='eeg')  # (164, 22, 1251)

# Get labels from events array
y_raw = epochs_final.events[:, 2]  # raw integer codes

# Encode to 0,1,2,3
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"X shape: {X.shape}")
print(f"  → {X.shape[0]} trials")
print(f"  → {X.shape[1]} EEG channels")
print(f"  → {X.shape[2]} timepoints")
print()
print(f"y shape: {y.shape}")
print(f"Unique classes: {np.unique(y)}")
print(f"Class mapping:")
for i, cls in enumerate(le.classes_):
    name = list(epochs_final.event_id.keys())[list(epochs_final.event_id.values()).index(cls)]
    count = np.sum(y == i)
    print(f"  {i} → event code {cls} ({name}): {count} trials")
```

### Cell 29 (code)
```python
# CSP works on pairs of classes (binary)
# n_components=4 means 4 spatial filters per class pair

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Fit CSP on all data to visualize the spatial filters
csp.fit(X, y)

print("✅ CSP fitted successfully!")
print(f"CSP components shape: {csp.filters_.shape}")
print(f"  → {csp.filters_.shape[0]} spatial filters")
print(f"  → {csp.filters_.shape[1]} channels per filter")
```

### Cell 30 (code)
```python
# CSP patterns show which brain regions are most important
# for separating the classes

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, ax in enumerate(axes):
    mne.viz.plot_topomap(
        csp.patterns_[i],
        epochs_final.info,
        axes=ax,
        show=False,
        contours=4
    )
    ax.set_title(f'CSP Pattern {i+1}', fontsize=11)

plt.suptitle('CSP Spatial Patterns — What Each Filter Is Looking At',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Cell 31 (code)
```python
# Transform raw EEG data into CSP features

X_csp = csp.transform(X)

print(f"Before CSP: {X.shape}")
print(f"  → (trials × channels × timepoints)")
print()
print(f"After CSP:  {X_csp.shape}")
print(f"  → (trials × csp_features)")
print()
print(f"CSP feature values (first 5 trials):")
print(np.round(X_csp[:5], 4))
print()
print(f"Feature statistics:")
print(f"  Mean: {X_csp.mean(axis=0).round(4)}")
print(f"  Std:  {X_csp.std(axis=0).round(4)}")
print(f"  Min:  {X_csp.min(axis=0).round(4)}")
print(f"  Max:  {X_csp.max(axis=0).round(4)}")
```

