"""
Breast Cancer Classification: Demonstrating the Accuracy Trap

This module implements a medical diagnosis classifier and exposes
the critical failure modes that occur when accuracy is used as the
primary metric for imbalanced medical datasets.

Key insight: A model that predicts "benign" for every patient achieves
~63% accuracy on this dataset, but misses 100% of malignant tumors.
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_data(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the breast cancer dataset."""
    if path is None:
        path = PROJECT_ROOT / "data" / "breast_cancer.csv"

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    X = df.drop(columns=["id", "diagnosis"])
    y = (df["diagnosis"] == "M").astype(int)

    return X, y


def get_class_distribution(y: pd.Series | np.ndarray) -> dict:
    """Return class counts and imbalance ratio."""
    y_array = np.asarray(y)
    benign = int((y_array == 0).sum())
    malignant = int((y_array == 1).sum())
    return {
        "benign": benign,
        "malignant": malignant,
        "imbalance_ratio": round(benign / malignant, 2)
    }


def plot_class_distribution(y: np.ndarray, save: bool = True) -> None:
    """Plot class distribution as a bar chart."""
    dist = get_class_distribution(y)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(['Benign (0)', 'Malignant (1)'],
                  [dist['benign'], dist['malignant']],
                  color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'Class Distribution (Imbalance Ratio: {dist["imbalance_ratio"]}:1)',
                 fontsize=14, fontweight='bold')

    for bar, val in zip(bars, [dist['benign'], dist['malignant']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, max(dist['benign'], dist['malignant']) * 1.15)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, title: str, save_name: str = None) -> None:
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'},
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add labels for TN, FP, FN, TP
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                   ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_name:
        plt.savefig(FIGURES_DIR / f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_confusion_matrices(results_dict: dict, save: bool = True) -> None:
    """Plot all confusion matrices in a single figure."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{name}\nAcc: {results["accuracy"]:.1%} | Recall: {results["recall"]:.1%}',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(models_data: dict, X_test: np.ndarray, y_test: np.ndarray,
                    save: bool = True) -> None:
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_data)))

    for (name, data), color in zip(models_data.items(), colors):
        model = data['model']

        if hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curves(models_data: dict, X_test: np.ndarray,
                                  y_test: np.ndarray, save: bool = True) -> None:
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_data)))

    for (name, data), color in zip(models_data.items(), colors):
        model = data['model']

        if hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)

        ax.plot(recall, precision, color=color, lw=2.5,
                label=f'{name} (AP = {ap:.3f})')

    # Baseline (random classifier)
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1.5,
               label=f'Baseline (AP = {baseline:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(results_dict: dict, save: bool = True) -> None:
    """Plot bar chart comparing all metrics across models."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results_dict.keys())

    x = np.arange(len(metrics))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for i, (name, results) in enumerate(results_dict.items()):
        values = [results[m] * 100 for m in metrics]
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.8)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        multiplier += 1

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cost_comparison(costs_dict: dict, save: bool = True) -> None:
    """Plot cost comparison across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(costs_dict.keys())
    fn_costs = [costs_dict[name]['false_negative_cost'] for name in model_names]
    fp_costs = [costs_dict[name]['false_positive_cost'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, fn_costs, width, label='False Negative Cost',
                   color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, fp_costs, width, label='False Positive Cost',
                   color='#3498db', edgecolor='black')

    ax.set_ylabel('Cost ($)', fontsize=12)
    ax.set_title('Cost Analysis by Error Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add total cost labels
    for i, name in enumerate(model_names):
        total = costs_dict[name]['total_cost']
        ax.annotate(f'Total: ${total:,.0f}',
                   xy=(i, max(fn_costs[i], fp_costs[i]) + 5000),
                   ha='center', fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(fn_costs) * 1.3)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cross_validation_scores(X: np.ndarray, y: np.ndarray, save: bool = True) -> dict:
    """Perform and plot cross-validation scores for all models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Naive SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
        ]),
        'Weighted SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42))
        ]),
        'SMOTE + SVM': ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
        ])
    }

    cv_results = {}
    all_scores = []
    labels = []

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        all_scores.append(scores)
        labels.append(name)

    # Plot boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(all_scores, tick_labels=labels, patch_artist=True)

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Recall Score', fontsize=12)
    ax.set_title('5-Fold Cross-Validation: Recall Scores', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    for i, (name, result) in enumerate(cv_results.items()):
        ax.annotate(f'μ={result["mean"]:.3f}\nσ={result["std"]:.3f}',
                   xy=(i + 1, result['mean']),
                   xytext=(i + 1.3, result['mean']),
                   fontsize=9, ha='left')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'cross_validation_recall.png', dpi=150, bbox_inches='tight')
    plt.show()

    return cv_results


def plot_learning_curves(X: np.ndarray, y: np.ndarray, save: bool = True) -> None:
    """Plot learning curves to diagnose bias/variance."""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42))
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='recall'
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#3498db')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color='#e74c3c')

    ax.plot(train_sizes, train_mean, 'o-', color='#3498db', lw=2, label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color='#e74c3c', lw=2, label='Cross-Validation Score')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Recall Score', fontsize=12)
    ax.set_title('Learning Curves (Weighted SVM)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.05)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def train_naive_model(X_train, X_test, y_train, y_test) -> dict:
    """Train SVM without addressing class imbalance."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, random_state=42, probability=True))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1])
    }


def train_weighted_model(X_train, X_test, y_train, y_test) -> dict:
    """Train SVM with class weighting."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42, probability=True))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1])
    }


def train_smote_model(X_train, X_test, y_train, y_test) -> dict:
    """Train SVM with SMOTE oversampling."""
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, random_state=42, probability=True))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Get resampled counts
    smote = SMOTE(random_state=42)
    _, y_resampled = smote.fit_resample(StandardScaler().fit_transform(X_train), y_train)

    return {
        "model": pipeline,
        "original_samples": len(y_train),
        "resampled_samples": len(y_resampled),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1])
    }


def cost_analysis(results: dict, cost_fn: float = 50000, cost_fp: float = 500) -> dict:
    """Translate confusion matrix into costs."""
    fn = results["false_negatives"]
    fp = results["false_positives"]
    total_cost = fn * cost_fn + fp * cost_fp

    return {
        "false_negative_cost": fn * cost_fn,
        "false_positive_cost": fp * cost_fp,
        "total_cost": total_cost,
        "cost_per_error": total_cost / (fn + fp) if (fn + fp) > 0 else 0
    }


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}")


def print_model_results(name: str, results: dict, cost: dict) -> None:
    """Print formatted model results."""
    print(f"\n{'Model:':<20} {name}")
    print(f"{'Accuracy:':<20} {results['accuracy']:.1%}")
    print(f"{'Precision:':<20} {results['precision']:.1%}")
    print(f"{'Recall:':<20} {results['recall']:.1%}")
    print(f"{'F1 Score:':<20} {results['f1']:.1%}")
    print(f"{'False Negatives:':<20} {results['false_negatives']} (missed cancers)")
    print(f"{'False Positives:':<20} {results['false_positives']} (unnecessary biopsies)")
    print(f"{'Estimated Cost:':<20} ${cost['total_cost']:,.0f}")


def run_comparison(show_plots: bool = True, save_plots: bool = True):
    """Execute full comparison with visualizations."""

    # Load data
    X, y = load_data()

    print_section("BREAST CANCER CLASSIFICATION: THE ACCURACY TRAP")

    dist = get_class_distribution(y)
    print(f"\n{'Dataset Summary':^60}")
    print(f"{'─' * 60}")
    print(f"  Benign samples:     {dist['benign']}")
    print(f"  Malignant samples:  {dist['malignant']}")
    print(f"  Imbalance ratio:    {dist['imbalance_ratio']}:1")
    print(f"  Total features:     {X.shape[1]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Plot class distribution
    if show_plots:
        print("\n[Generating class distribution plot...]")
        plot_class_distribution(y, save=save_plots)

    # Train models
    print_section("TRAINING MODELS", "-")

    print("\n[1/3] Training Naive SVM...")
    naive = train_naive_model(X_train, X_test, y_train, y_test)
    naive_cost = cost_analysis(naive)

    print("[2/3] Training Weighted SVM...")
    weighted = train_weighted_model(X_train, X_test, y_train, y_test)
    weighted_cost = cost_analysis(weighted)

    print("[3/3] Training SMOTE + SVM...")
    smote = train_smote_model(X_train, X_test, y_train, y_test)
    smote_cost = cost_analysis(smote)

    # Results dictionaries
    results_dict = {
        'Naive SVM': naive,
        'Weighted SVM': weighted,
        'SMOTE + SVM': smote
    }

    costs_dict = {
        'Naive SVM': naive_cost,
        'Weighted SVM': weighted_cost,
        'SMOTE + SVM': smote_cost
    }

    # Print results
    print_section("MODEL RESULTS", "-")

    for name, results in results_dict.items():
        print_model_results(name, results, costs_dict[name])

    # Generate visualizations
    if show_plots:
        print_section("GENERATING VISUALIZATIONS", "-")

        print("\n[1/6] Confusion matrices...")
        plot_all_confusion_matrices(results_dict, save=save_plots)

        print("[2/6] ROC curves...")
        plot_roc_curves(results_dict, X_test, y_test, save=save_plots)

        print("[3/6] Precision-Recall curves...")
        plot_precision_recall_curves(results_dict, X_test, y_test, save=save_plots)

        print("[4/6] Metrics comparison...")
        plot_metrics_comparison(results_dict, save=save_plots)

        print("[5/6] Cost analysis...")
        plot_cost_comparison(costs_dict, save=save_plots)

        print("[6/6] Cross-validation scores...")
        cv_results = plot_cross_validation_scores(X, y, save=save_plots)

        print("\n[Bonus] Learning curves...")
        plot_learning_curves(X, y, save=save_plots)

    # Summary
    print_section("DECISION SUMMARY")

    models = [(name, results_dict[name], costs_dict[name]) for name in results_dict]
    best = min(models, key=lambda x: x[2]['total_cost'])
    worst = max(models, key=lambda x: x[2]['total_cost'])

    print(f"\n{'BEST MODEL (lowest cost):'}")
    print(f"  {best[0]}")
    print(f"  • Accuracy: {best[1]['accuracy']:.1%}")
    print(f"  • Recall:   {best[1]['recall']:.1%}")
    print(f"  • Cost:     ${best[2]['total_cost']:,.0f}")

    print(f"\n{'WORST MODEL (highest cost):'}")
    print(f"  {worst[0]}")
    print(f"  • Accuracy: {worst[1]['accuracy']:.1%}")
    print(f"  • Recall:   {worst[1]['recall']:.1%}")
    print(f"  • Cost:     ${worst[2]['total_cost']:,.0f}")

    savings = worst[2]['total_cost'] - best[2]['total_cost']
    print(f"\n{'─' * 60}")
    print(f"  POTENTIAL SAVINGS: ${savings:,.0f} per 100 patients")
    print(f"{'─' * 60}")

    if best[1]['accuracy'] < worst[1]['accuracy']:
        print("\n  ⚠️  CRITICAL INSIGHT:")
        print("  The model with LOWER accuracy has BETTER patient outcomes!")
        print("  Optimizing for accuracy would choose the WRONG model.")

    if save_plots:
        print(f"\n  📁 Figures saved to: {FIGURES_DIR}")

    return results_dict, costs_dict


if __name__ == "__main__":
    results, costs = run_comparison(show_plots=True, save_plots=True)
