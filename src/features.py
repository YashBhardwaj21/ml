import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import CSP
from sklearn.preprocessing import LabelEncoder


def extract_csp_features(epochs_final, n_components=8, tmin=0.5, tmax=3.5):
    from sklearn.preprocessing import StandardScaler

    # Crop to active motor imagery window only (0.5s to 3.5s)
    epochs_cropped = epochs_final.copy().crop(tmin=tmin, tmax=tmax)

    # Extract Mu band (8-13 Hz) and Beta band (13-30 Hz) separately
    epochs_mu   = epochs_cropped.copy().filter(8.,  13., fir_design='firwin', verbose=False)
    epochs_beta = epochs_cropped.copy().filter(13., 30., fir_design='firwin', verbose=False)

    X_mu   = epochs_mu.get_data()[:,   :22, :]
    X_beta = epochs_beta.get_data()[:, :22, :]

    y_raw = epochs_final.events[:, 2]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Remove NaN/Inf trials from both bands
    bad_mu   = np.any(np.isnan(X_mu)   | np.isinf(X_mu),   axis=(1, 2))
    bad_beta = np.any(np.isnan(X_beta) | np.isinf(X_beta), axis=(1, 2))
    bad_trials = bad_mu | bad_beta

    if bad_trials.sum() > 0:
        print(f"Found {bad_trials.sum()} trials with NaN/Inf - removing them")
        X_mu   = X_mu[~bad_trials]
        X_beta = X_beta[~bad_trials]
        y      = y[~bad_trials]
    else:
        print(f"No NaN/Inf values found")

    print(f"X shape before CSP: {X_mu.shape}")
    print(f"y distribution:     {np.bincount(y)}")

    # Fit CSP on Mu band
    csp_mu = CSP(n_components=n_components, reg=0.05, log=True, norm_trace=False)
    X_csp_mu = csp_mu.fit_transform(X_mu, y)

    # Check for NaN after CSP on Mu
    if np.any(np.isnan(X_csp_mu)):
        print("Mu CSP produced NaN - increasing regularization")
        csp_mu = CSP(n_components=n_components, reg=0.2, log=True, norm_trace=False)
        X_csp_mu = csp_mu.fit_transform(X_mu, y)

    # Fit CSP on Beta band
    csp_beta = CSP(n_components=n_components, reg=0.05, log=True, norm_trace=False)
    X_csp_beta = csp_beta.fit_transform(X_beta, y)

    # Check for NaN after CSP on Beta
    if np.any(np.isnan(X_csp_beta)):
        print("Beta CSP produced NaN - increasing regularization")
        csp_beta = CSP(n_components=n_components, reg=0.2, log=True, norm_trace=False)
        X_csp_beta = csp_beta.fit_transform(X_beta, y)

    # Concatenate Mu and Beta features
    # Result: n_trials x (n_components * 2) = n_trials x 16
    X_csp = np.hstack([X_csp_mu, X_csp_beta])

    # Normalize features to zero mean and unit variance
    scaler = StandardScaler()
    X_csp = scaler.fit_transform(X_csp)

    print(f"X shape after CSP (Mu + Beta): {X_csp.shape}")
    print(f"  Mu features:   {X_csp_mu.shape[1]}")
    print(f"  Beta features: {X_csp_beta.shape[1]}")
    print(f"  Total features: {X_csp.shape[1]}")
    print(f"Classes:          {np.unique(y)}")
    print(f"Trials per class: {np.bincount(y)}")

    return X_csp, y, csp_mu, csp_beta, scaler, le


def save_features(subject_id, X_csp, y, save_path='../data/features/'):
    save_file = f'{save_path}{subject_id}_features.npz'
    np.savez(save_file, X=X_csp, y=y)
    print(f"Features saved to {save_file}")


def load_features(subject_id, load_path='../data/features/'):
    load_file = f'{load_path}{subject_id}_features.npz'
    data = np.load(load_file)
    print(f"Loaded features for {subject_id}: X={data['X'].shape}, y={data['y'].shape}")
    return data['X'], data['y']


def plot_csp_patterns(csp, epochs_info, subject_id, figures_path='../results/figures/'):
    """Topographic maps of all 4 CSP spatial patterns"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            csp.patterns_[i],
            epochs_info,
            axes=ax,
            show=False,
            contours=4
        )
        ax.set_title(f'CSP Pattern {i+1}', fontsize=11)

    plt.suptitle(f'{subject_id} - CSP Spatial Patterns',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_patterns.png', dpi=100)
    plt.show()


def plot_csp_filters(csp, epochs_info, subject_id, figures_path='../results/figures/'):
    """Topographic maps of all 4 CSP spatial filters"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            csp.filters_[i],
            epochs_info,
            axes=ax,
            show=False,
            contours=4
        )
        ax.set_title(f'CSP Filter {i+1}', fontsize=11)

    plt.suptitle(f'{subject_id} - CSP Spatial Filters',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_filters.png', dpi=100)
    plt.show()


def plot_csp_feature_distribution(X_csp, y, subject_id, figures_path='../results/figures/'):
    """Distribution of each CSP feature value per class"""
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors      = ['blue', 'red', 'green', 'purple']
    n_components = X_csp.shape[1]

    fig, axes = plt.subplots(1, n_components, figsize=(16, 4))

    for comp in range(n_components):
        ax = axes[comp]
        for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
            values = X_csp[y == cls_idx, comp]
            ax.hist(values, bins=20, alpha=0.5, color=color,
                    label=cls_name, edgecolor='white')
        ax.set_title(f'CSP Feature {comp+1}')
        ax.set_xlabel('Log Variance')
        ax.set_ylabel('Count')
        if comp == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f'{subject_id} - CSP Feature Distribution Per Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_feature_distribution.png', dpi=100)
    plt.show()


def plot_csp_scatter(X_csp, y, subject_id, figures_path='../results/figures/'):
    """Scatter plots of CSP feature pairs — shows class separability"""
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors      = ['blue', 'red', 'green', 'purple']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
            mask = y == cls_idx
            ax.scatter(X_csp[mask, i], X_csp[mask, j],
                       c=color, label=cls_name, alpha=0.6, s=30)
        ax.set_xlabel(f'CSP Feature {i+1}')
        ax.set_ylabel(f'CSP Feature {j+1}')
        ax.set_title(f'Feature {i+1} vs Feature {j+1}')
        ax.legend(fontsize=8)

    plt.suptitle(f'{subject_id} - CSP Feature Scatter Plots',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_scatter.png', dpi=100)
    plt.show()


def plot_csp_feature_boxplot(X_csp, y, subject_id, figures_path='../results/figures/'):
    """Boxplot of each CSP feature per class — shows median and spread"""
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors      = ['blue', 'red', 'green', 'purple']
    n_components = X_csp.shape[1]

    fig, axes = plt.subplots(1, n_components, figsize=(16, 5))

    for comp in range(n_components):
        ax = axes[comp]
        data_per_class = [X_csp[y == cls_idx, comp] for cls_idx in range(4)]

        bp = ax.boxplot(data_per_class, patch_artist=True, notch=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticklabels(['LH', 'RH', 'F', 'T'], fontsize=9)
        ax.set_title(f'CSP Feature {comp+1}')
        ax.set_ylabel('Log Variance')

    plt.suptitle(f'{subject_id} - CSP Feature Boxplot Per Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_boxplot.png', dpi=100)
    plt.show()


def plot_csp_feature_correlation(X_csp, subject_id, figures_path='../results/figures/'):
    """Correlation between CSP features — should be low for good features"""
    corr = np.corrcoef(X_csp.T)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        xticklabels=[f'F{i+1}' for i in range(X_csp.shape[1])],
        yticklabels=[f'F{i+1}' for i in range(X_csp.shape[1])]
    )
    plt.title(f'{subject_id} - CSP Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_feature_correlation.png', dpi=100)
    plt.show()


def plot_csp_variance_explained(X_csp, y, subject_id, figures_path='../results/figures/'):
    """Mean feature value per class per component — shows which features discriminate which classes"""
    class_names  = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors       = ['blue', 'red', 'green', 'purple']
    n_components = X_csp.shape[1]

    means = np.array([
        [X_csp[y == cls_idx, comp].mean() for comp in range(n_components)]
        for cls_idx in range(4)
    ])

    x = np.arange(n_components)
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        ax.bar(x + cls_idx * width, means[cls_idx],
               width, label=cls_name, color=color, alpha=0.7)

    ax.set_xlabel('CSP Component')
    ax.set_ylabel('Mean Log Variance')
    ax.set_title(f'{subject_id} - Mean CSP Feature Value Per Class')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Feature {i+1}' for i in range(n_components)])
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_csp_mean_per_class.png', dpi=100)
    plt.show()


def visualize_features(X_csp, y, csp, epochs_info, subject_id, figures_path='../results/figures/features/'):
    print(f"\nGenerating feature plots for {subject_id}...")
    plot_csp_patterns(csp, epochs_info, subject_id, figures_path)
    plot_csp_filters(csp, epochs_info, subject_id, figures_path)
    plot_csp_feature_distribution(X_csp, y, subject_id, figures_path)
    plot_csp_scatter(X_csp, y, subject_id, figures_path)
    plot_csp_feature_boxplot(X_csp, y, subject_id, figures_path)
    plot_csp_feature_correlation(X_csp, subject_id, figures_path)
    plot_csp_variance_explained(X_csp, y, subject_id, figures_path)
    print(f"All feature plots saved for {subject_id}")