import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pickle
import os
import json


CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']


def build_svm():
    return SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )


def build_lda():
    return LinearDiscriminantAnalysis(
        solver='lsqr',
        shrinkage='auto'
    )


def build_rf():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )


def evaluate_model(model, X, y, n_splits=5):
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    y_pred = cross_val_predict(model, X, y, cv=cv)
    cm     = confusion_matrix(y, y_pred)
    report = classification_report(
        y, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    return {
        'scores':  scores,
        'mean':    scores.mean(),
        'std':     scores.std(),
        'cm':      cm,
        'report':  report,
        'y_pred':  y_pred,
        'y_true':  y
    }


def save_model(model, subject_id, model_name, save_path='../results/models/baseline/'):
    os.makedirs(save_path, exist_ok=True)
    path = f'{save_path}{subject_id}_{model_name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(subject_id, model_name, load_path='../results/models/baseline'):
    path = f'{load_path}{subject_id}_{model_name}.pkl'
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_metrics(all_results, subjects, save_path='../results/metrics/baseline'):
    os.makedirs(save_path, exist_ok=True)

    # Accuracy summary CSV
    rows = []
    for subject_id in subjects:
        row = {'Subject': subject_id}
        for model_name in ['SVM', 'LDA', 'RF']:
            result = all_results[subject_id][model_name]
            row[f'{model_name}_mean'] = round(result['mean'] * 100, 2)
            row[f'{model_name}_std']  = round(result['std']  * 100, 2)
        rows.append(row)
    acc_df = pd.DataFrame(rows)
    acc_df.to_csv(f'{save_path}accuracy_summary.csv', index=False)
    print(f"Saved accuracy_summary.csv")

    # Per class F1 CSV
    f1_rows = []
    for subject_id in subjects:
        for model_name in ['SVM', 'LDA', 'RF']:
            report = all_results[subject_id][model_name]['report']
            for cls in CLASS_NAMES:
                f1_rows.append({
                    'Subject':   subject_id,
                    'Model':     model_name,
                    'Class':     cls,
                    'Precision': round(report[cls]['precision'], 4),
                    'Recall':    round(report[cls]['recall'],    4),
                    'F1':        round(report[cls]['f1-score'],  4),
                    'Support':   report[cls]['support']
                })
    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv(f'{save_path}per_class_metrics.csv', index=False)
    print(f"Saved per_class_metrics.csv")

    # Per fold scores CSV
    fold_rows = []
    for subject_id in subjects:
        for model_name in ['SVM', 'LDA', 'RF']:
            scores = all_results[subject_id][model_name]['scores']
            for fold_idx, score in enumerate(scores):
                fold_rows.append({
                    'Subject': subject_id,
                    'Model':   model_name,
                    'Fold':    fold_idx + 1,
                    'Accuracy': round(score * 100, 2)
                })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(f'{save_path}per_fold_scores.csv', index=False)
    print(f"Saved per_fold_scores.csv")

    return acc_df, f1_df, fold_df


def plot_accuracy_bars(all_results, subjects, figures_path='../results/figures/training/baseline/'):
    svm_means = [all_results[s]['SVM']['mean'] * 100 for s in subjects]
    lda_means = [all_results[s]['LDA']['mean'] * 100 for s in subjects]
    rf_means  = [all_results[s]['RF']['mean']  * 100 for s in subjects]
    svm_stds  = [all_results[s]['SVM']['std']  * 100 for s in subjects]
    lda_stds  = [all_results[s]['LDA']['std']  * 100 for s in subjects]
    rf_stds   = [all_results[s]['RF']['std']   * 100 for s in subjects]

    x     = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, svm_means, width, yerr=svm_stds, label='SVM',
           color='steelblue', alpha=0.8, capsize=4)
    ax.bar(x,         lda_means, width, yerr=lda_stds, label='LDA',
           color='tomato',    alpha=0.8, capsize=4)
    ax.bar(x + width, rf_means,  width, yerr=rf_stds,  label='Random Forest',
           color='seagreen',  alpha=0.8, capsize=4)

    ax.axhline(y=25, color='black', linestyle='--', linewidth=1.5, label='Chance (25%)')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy with Std Dev — All Subjects All Models')
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{figures_path}accuracy_bars.png', dpi=100)
    plt.show()


def plot_confusion_matrices(all_results, subjects, model_name='SVM', figures_path='../results/figures/training/baseline/'):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for ax, subject_id in zip(axes.flat, subjects):
        cm      = all_results[subject_id][model_name]['cm']
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
        acc = all_results[subject_id][model_name]['mean'] * 100
        ax.set_title(f"{subject_id} — {model_name} ({acc:.1f}%)", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Confusion Matrices — {model_name} — All 9 Subjects',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}confusion_matrices_{model_name}.png', dpi=100)
    plt.show()


def plot_f1_heatmap(all_results, subjects, figures_path='../results/figures/training/baseline/'):
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
        ax.set_title(f'{model_name} — F1 Score Per Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Subject')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('F1 Score Heatmap — All Models All Subjects',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}f1_heatmap.png', dpi=100)
    plt.show()


def plot_per_fold_scores(all_results, subjects, figures_path='../results/figures/training/baseline/'):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for ax, subject_id in zip(axes.flat, subjects):
        svm_scores = all_results[subject_id]['SVM']['scores'] * 100
        lda_scores = all_results[subject_id]['LDA']['scores'] * 100
        rf_scores  = all_results[subject_id]['RF']['scores']  * 100

        folds = np.arange(1, len(svm_scores) + 1)
        ax.plot(folds, svm_scores, 'o-', color='steelblue', label='SVM')
        ax.plot(folds, lda_scores, 's-', color='tomato',    label='LDA')
        ax.plot(folds, rf_scores,  '^-', color='seagreen',  label='RF')
        ax.axhline(y=25, color='black', linestyle='--', linewidth=1, label='Chance')
        ax.set_title(f'{subject_id}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(folds)
        if subject_id == 'A01T':
            ax.legend(fontsize=7)

    plt.suptitle('Per Fold Accuracy — All Subjects All Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}per_fold_scores.png', dpi=100)
    plt.show()


def plot_model_comparison_box(all_results, subjects, figures_path='../results/figures/training/baseline/'):
    svm_scores = [all_results[s]['SVM']['mean'] * 100 for s in subjects]
    lda_scores = [all_results[s]['LDA']['mean'] * 100 for s in subjects]
    rf_scores  = [all_results[s]['RF']['mean']  * 100 for s in subjects]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [svm_scores, lda_scores, rf_scores],
        labels=['SVM', 'LDA', 'Random Forest'],
        patch_artist=True,
        notch=False
    )
    colors = ['steelblue', 'tomato', 'seagreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=25, color='black', linestyle='--', linewidth=1.5, label='Chance (25%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison — Accuracy Distribution Across 9 Subjects')
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{figures_path}model_comparison_boxplot.png', dpi=100)
    plt.show()


def plot_class_accuracy(all_results, subjects, figures_path='../results/figures/training/baseline/'):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, model_name in zip(axes, ['SVM', 'LDA', 'RF']):
        recall_matrix = []
        for subject_id in subjects:
            report = all_results[subject_id][model_name]['report']
            recall_row = [report[cls]['recall'] for cls in CLASS_NAMES]
            recall_matrix.append(recall_row)

        sns.heatmap(
            np.array(recall_matrix),
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=CLASS_NAMES,
            yticklabels=subjects,
            ax=ax,
            vmin=0, vmax=1
        )
        ax.set_title(f'{model_name} — Recall Per Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Subject')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Recall Per Class Per Subject — All Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}recall_heatmap.png', dpi=100)
    plt.show()


def plot_best_subject_detail(all_results, subjects, figures_path='../results/figures/training/baseline/'):
    # Find best subject by SVM accuracy
    best_subject = max(subjects, key=lambda s: all_results[s]['SVM']['mean'])
    print(f"Best subject: {best_subject} — {all_results[best_subject]['SVM']['mean']*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model_name in zip(axes, ['SVM', 'LDA', 'RF']):
        cm      = all_results[best_subject][model_name]['cm']
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
        acc = all_results[best_subject][model_name]['mean'] * 100
        ax.set_title(f'{model_name} ({acc:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Best Subject ({best_subject}) — All Models Confusion Matrices',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{figures_path}best_subject_detail_{best_subject}.png', dpi=100)
    plt.show()


def run_all_visualizations(all_results, subjects, figures_path='../results/figures/training/baseline/'):
    print("\nGenerating all training visualizations...")
    plot_accuracy_bars(all_results, subjects, figures_path)
    plot_confusion_matrices(all_results, subjects, 'SVM', figures_path)
    plot_confusion_matrices(all_results, subjects, 'LDA', figures_path)
    plot_confusion_matrices(all_results, subjects, 'RF',  figures_path)
    plot_f1_heatmap(all_results, subjects, figures_path)
    plot_per_fold_scores(all_results, subjects, figures_path)
    plot_model_comparison_box(all_results, subjects, figures_path)
    plot_class_accuracy(all_results, subjects, figures_path)
    plot_best_subject_detail(all_results, subjects, figures_path)
    print("All training visualizations saved.")