import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA


def load_subject(subject_id, data_path='data/raw/'):
    file_path = f'{data_path}{subject_id}.gdf'
    raw = mne.io.read_raw_gdf(
        file_path,
        preload=True,
        eog=['EOG-left', 'EOG-central', 'EOG-right']
    )
    print(f"Loaded {subject_id}: {len(raw.ch_names)} channels, "
          f"{raw.times[-1]:.1f}s duration")
    return raw


def apply_filter(raw):
    raw_filtered = raw.copy().filter(7., 30., fir_design='firwin', verbose=False)
    print(f"Bandpass filter applied (7-30Hz)")
    return raw_filtered


def set_montage(raw_filtered):
    channel_rename = {
        'EEG-Fz':  'Fz',
        'EEG-0':   'FC3',
        'EEG-1':   'FC1',
        'EEG-2':   'FCz',
        'EEG-3':   'FC2',
        'EEG-4':   'FC4',
        'EEG-5':   'C5',
        'EEG-C3':  'C3',
        'EEG-6':   'C1',
        'EEG-Cz':  'Cz',
        'EEG-7':   'C2',
        'EEG-C4':  'C4',
        'EEG-8':   'C6',
        'EEG-9':   'CP3',
        'EEG-10':  'CP1',
        'EEG-11':  'CPz',
        'EEG-12':  'CP2',
        'EEG-13':  'CP4',
        'EEG-14':  'P1',
        'EEG-Pz':  'Pz',
        'EEG-15':  'P2',
        'EEG-16':  'POz',
    }
    raw_filtered.rename_channels(channel_rename)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_filtered.set_montage(montage, on_missing='ignore')
    print(f"Montage set successfully")
    return raw_filtered


def run_ica(raw_filtered):
    ica = ICA(n_components=20, random_state=42, max_iter='auto')
    ica.fit(raw_filtered)

    idx_left, _    = ica.find_bads_eog(raw_filtered, ch_name='EOG-left',    verbose=False)
    idx_central, _ = ica.find_bads_eog(raw_filtered, ch_name='EOG-central', verbose=False)
    idx_right, _   = ica.find_bads_eog(raw_filtered, ch_name='EOG-right',   verbose=False)

    all_bad = list(set(idx_left + idx_central + idx_right))
    ica.exclude = all_bad

    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean)

    print(f"ICA complete - removed components {all_bad}")
    print(f"  Before: max={raw_filtered.get_data(picks='eeg').max()*1e6:.2f}uV")
    print(f"  After:  max={raw_clean.get_data(picks='eeg').max()*1e6:.2f}uV")
    return raw_clean


def create_epochs(raw_clean):
    events, event_id = mne.events_from_annotations(raw_clean)

    mi_event_id = {
        'Left Hand':  event_id['769'],
        'Right Hand': event_id['770'],
        'Feet':       event_id['771'],
        'Tongue':     event_id['772']
    }

    epochs = mne.Epochs(
        raw_clean, events,
        event_id=mi_event_id,
        tmin=-0.5, tmax=4.5,
        baseline=(-0.5, 0),
        preload=True
    )

    ep_data = epochs.get_data() * 1e6
    trial_max = ep_data.max(axis=(1, 2))
    clean_mask = trial_max <= 100
    epochs_final = epochs[clean_mask]

    print(f"Epochs created")
    print(f"  Before rejection: {len(epochs)} trials")
    print(f"  After rejection:  {len(epochs_final)} trials")
    print(f"  Dropped:          {len(epochs) - len(epochs_final)} trials")
    print(f"  Per class:")
    for cls in epochs_final.event_id:
        print(f"    {cls}: {len(epochs_final[cls])} trials")

    return epochs, epochs_final


def plot_amplitude_histogram(epochs, subject_id, figures_path='../results/figures/'):
    ep_data   = epochs.get_data() * 1e6
    trial_max = ep_data.max(axis=(1, 2))

    plt.figure(figsize=(10, 4))
    plt.hist(trial_max, bins=50, color='steelblue', edgecolor='white')
    plt.axvline(100, color='orange',  linestyle='--', linewidth=2, label='100 uV')
    plt.axvline(150, color='red',     linestyle='--', linewidth=2, label='150 uV')
    plt.axvline(200, color='darkred', linestyle='--', linewidth=2, label='200 uV')
    plt.xlabel('Max Amplitude per Trial (uV)')
    plt.ylabel('Number of Trials')
    plt.title(f'{subject_id} - Trial Amplitude Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_amplitude_histogram.png', dpi=100)
    plt.show()


def plot_trial_distribution(epochs_final, subject_id, figures_path='../results/figures/'):
    classes = list(epochs_final.event_id.keys())
    counts  = [len(epochs_final[cls]) for cls in classes]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(classes, counts,
                   color=['blue', 'red', 'green', 'purple'],
                   edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 str(count), ha='center', fontsize=12, fontweight='bold')
    plt.axhline(y=30, color='orange', linestyle='--',
                linewidth=1.5, label='Minimum recommended (30)')
    plt.ylabel('Number of Clean Trials')
    plt.title(f'{subject_id} - Final Clean Trial Count Per Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_trial_distribution.png', dpi=100)
    plt.show()


def plot_channel_correlation(epochs_final, subject_id, figures_path='../results/figures/'):
    epoch_data  = epochs_final.get_data()[:, :22, :] * 1e6
    epoch_mean  = epoch_data.mean(axis=2)
    corr_matrix = np.corrcoef(epoch_mean.T)

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
    plt.title(f'{subject_id} - EEG Channel Correlation (Clean Epochs)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_correlation.png', dpi=100)
    plt.show()


def plot_epoch_waveforms(epochs_final, subject_id, figures_path='../results/figures/'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    classes   = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors    = ['blue', 'red', 'green', 'purple']
    c3_idx    = epochs_final.ch_names.index('C3')

    for ax, cls, color in zip(axes.flat, classes, colors):
        data  = epochs_final[cls].get_data()[:, c3_idx, :] * 1e6
        times = epochs_final.times
        mean  = np.mean(data, axis=0)
        std   = np.std(data, axis=0)

        for trial in data[:10]:
            ax.plot(times, trial, color=color, alpha=0.1, linewidth=0.5)
        ax.plot(times, mean, color=color, linewidth=2, label='Mean')
        ax.fill_between(times, mean - std, mean + std, alpha=0.2, color=color)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Cue onset')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(f'{cls} - Channel C3 ({len(epochs_final[cls])} trials)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (uV)')
        ax.legend(fontsize=8)

    plt.suptitle(f'{subject_id} - Epoched EEG Mean +/- Std per Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_epoch_waveforms.png', dpi=100)
    plt.show()


def plot_topomaps(epochs_final, subject_id, figures_path='../results/figures/'):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    classes   = list(epochs_final.event_id.keys())

    for ax, cls in zip(axes, classes):
        epochs_final[cls].compute_psd(fmin=8, fmax=30).plot_topomap(
            axes=ax,
            ch_type='eeg',
            contours=4,
            bands={'Mu/Beta (8-30Hz)': (8, 30)},
            show=False
        )
        ax.set_title(f'{cls}\n(8-30Hz, n={len(epochs_final[cls])})', fontsize=11)

    plt.suptitle(f'{subject_id} - Scalp Topomaps Motor Imagery Power',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_topomaps.png', dpi=100)
    plt.show()


def plot_psd(raw, raw_clean, subject_id, figures_path='../results/figures/'):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    psd_raw = raw.compute_psd(fmin=1, fmax=50, picks='eeg')
    psd_raw.plot(axes=axes[0], show=False)
    axes[0].axvspan(8,  13, alpha=0.15, color='green',  label='Mu (8-13Hz)')
    axes[0].axvspan(13, 30, alpha=0.15, color='orange', label='Beta (13-30Hz)')
    axes[0].set_title(f'{subject_id} - Raw EEG PSD')
    axes[0].legend(loc='upper right', fontsize=8)

    psd_clean = raw_clean.compute_psd(fmin=1, fmax=50, picks='eeg')
    psd_clean.plot(axes=axes[1], show=False)
    axes[1].axvspan(8,  13, alpha=0.15, color='green')
    axes[1].axvspan(13, 30, alpha=0.15, color='orange')
    axes[1].set_title(f'{subject_id} - After ICA PSD')

    plt.suptitle(f'{subject_id} - Power Spectral Density Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_psd.png', dpi=100)
    plt.show()


def visualize_subject(raw, raw_clean, epochs, epochs_final, subject_id, figures_path='../results/figures/'):
    print(f"\nGenerating plots for {subject_id}...")
    plot_amplitude_histogram(epochs, subject_id, figures_path)
    plot_trial_distribution(epochs_final, subject_id, figures_path)
    plot_channel_correlation(epochs_final, subject_id, figures_path)
    plot_epoch_waveforms(epochs_final, subject_id, figures_path)
    plot_topomaps(epochs_final, subject_id, figures_path)
    plot_psd(raw, raw_clean, subject_id, figures_path)
    print(f"All plots saved for {subject_id}")


def preprocess_subject(subject_id, data_path='data/raw/', save_path='data/processed/'):
    print(f"\n{'='*50}")
    print(f"  Processing {subject_id}")
    print(f"{'='*50}")

    raw          = load_subject(subject_id, data_path)
    raw_filtered = apply_filter(raw)
    raw_filtered = set_montage(raw_filtered)
    raw_clean    = run_ica(raw_filtered)
    epochs, epochs_final = create_epochs(raw_clean)

    save_file = f'{save_path}{subject_id}_clean_epo.fif'
    epochs_final.save(save_file, overwrite=True)
    print(f"Saved to {save_file}")

    return raw, raw_clean, epochs, epochs_final