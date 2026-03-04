import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
import os


CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']


def extract_riemannian_features(epochs_final, tmin=0.5, tmax=3.5):
    epochs_cropped = epochs_final.copy().crop(tmin=tmin, tmax=tmax)
    X     = epochs_cropped.get_data()[:, :22, :]
    y_raw = epochs_final.events[:, 2]
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw)

    bad = np.any(np.isnan(X) | np.isinf(X), axis=(1, 2))
    if bad.sum() > 0:
        print(f"  Removing {bad.sum()} bad trials")
        X = X[~bad]
        y = y[~bad]
    else:
        print(f"  No bad trials found")

    print(f"  X shape:        {X.shape}")
    print(f"  y distribution: {np.bincount(y)}")

    cov_estimator = Covariances(estimator='oas')
    X_cov         = cov_estimator.fit_transform(X)
    print(f"  Covariance matrices shape: {X_cov.shape}")

    ts   = TangentSpace(metric='riemann')
    X_ts = ts.fit_transform(X_cov)
    print(f"  Tangent space features shape: {X_ts.shape}")

    scaler = StandardScaler()
    X_ts   = scaler.fit_transform(X_ts)

    return X_ts, y, X_cov, cov_estimator, ts, scaler, le


def save_riemannian_features(subject_id, X_ts, y,
                              save_path='../data/features_riemannian/'):
    os.makedirs(save_path, exist_ok=True)
    save_file = f'{save_path}{subject_id}_riemannian.npz'
    np.savez(save_file, X=X_ts, y=y)
    print(f"  Saved to {save_file}")


def load_riemannian_features(subject_id,
                              load_path='../data/features_riemannian/'):
    data = np.load(f'{load_path}{subject_id}_riemannian.npz')
    print(f"  Loaded {subject_id}: X={data['X'].shape}, y={data['y'].shape}")
    return data['X'], data['y']


def plot_riemannian_scatter(X_ts, y, subject_id,
                            figures_path='../results/figures/features/riemannian/'):
    pca    = PCA(n_components=2)
    X_2d   = pca.fit_transform(X_ts)
    colors = ['blue', 'red', 'green', 'purple']

    plt.figure(figsize=(8, 6))
    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = y == cls_idx
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=color, label=cls_name, alpha=0.6, s=30)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    plt.xlabel(f'PC1 ({var1:.1f}% variance)')
    plt.ylabel(f'PC2 ({var2:.1f}% variance)')
    plt.title(f'{subject_id} - Riemannian Features PCA Projection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_riemannian_scatter.png', dpi=100)
    plt.show()


def plot_mean_covariance(X_cov, y, subject_id, epochs_info,
                         figures_path='../results/figures/features/riemannian/'):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ch_names  = [ch for ch in epochs_info['ch_names'] if 'EOG' not in ch][:22]

    for cls_idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        X_cls  = X_cov[y == cls_idx]
        mean_c = mean_covariance(X_cls, metric='riemann')
        diag   = np.sqrt(np.diag(mean_c))
        corr   = mean_c / np.outer(diag, diag)

        sns.heatmap(
            corr,
            ax=ax,
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            xticklabels=ch_names,
            yticklabels=ch_names,
            annot=False
        )
        ax.set_title(f'{cls_name}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', rotation=0,  labelsize=6)

    plt.suptitle(f'{subject_id} - Mean Riemannian Covariance Matrix Per Class',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_mean_covariance.png', dpi=100)
    plt.show()


def plot_feature_distribution(X_ts, y, subject_id,
                               figures_path='../results/figures/features/riemannian/'):
    colors = ['blue', 'red', 'green', 'purple']
    n_show = min(6, X_ts.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            break
        for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
            values = X_ts[y == cls_idx, i]
            ax.hist(values, bins=20, alpha=0.5, color=color,
                    label=cls_name, edgecolor='white')
        ax.set_title(f'Tangent Space Feature {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f'{subject_id} - Riemannian Feature Distribution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}{subject_id}_feature_distribution.png', dpi=100)
    plt.show()


def visualize_riemannian(X_ts, X_cov, y, subject_id, epochs_info,
                         figures_path='../results/figures/features/riemannian/'):
    os.makedirs(figures_path, exist_ok=True)
    print(f"  Generating Riemannian plots for {subject_id}...")
    plot_riemannian_scatter(X_ts, y, subject_id, figures_path)
    plot_mean_covariance(X_cov, y, subject_id, epochs_info, figures_path)
    plot_feature_distribution(X_ts, y, subject_id, figures_path)
    print(f"  Plots saved for {subject_id}")