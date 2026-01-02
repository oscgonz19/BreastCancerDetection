# Technical Report: Imbalanced Classification for Medical Diagnosis

## 1. Problem Specification

**Task**: Binary classification of breast tumor samples (malignant vs benign)

**Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC)
- 569 samples (357 benign, 212 malignant)
- 30 numerical features (cell nucleus measurements)
- Imbalance ratio: 1.68:1

![Class Distribution](../../figures/class_distribution.png)
*Figure 1: Dataset class distribution showing the 1.68:1 imbalance ratio.*

**Constraints**:
- False negatives (missed cancers) are 100x more costly than false positives
- Model must achieve minimum 95% recall on malignant class
- Interpretability preferred for clinical adoption

## 2. Baseline Analysis

### 2.1 Trivial Classifier Performance

A constant predictor returning "benign" for all inputs:

```
Accuracy:  62.7%
Precision: 0.0%
Recall:    0.0%
F1 Score:  0.0%
```

This establishes the accuracy floor and demonstrates that accuracy alone is meaningless for this task.

### 2.2 Naive SVM Performance

RBF kernel SVM trained on original (imbalanced) data:

```
Accuracy:  97.4%
Precision: 100.0%
Recall:    92.9%
F1 Score:  96.3%
```

**Confusion Matrix**:
```
              Predicted
              Neg    Pos
Actual Neg    71      0
       Pos     3     40
```

False negatives: 3 (missed malignant tumors)

## 3. Model Comparison

### 3.1 Three Approaches Evaluated

We implemented and compared three strategies for handling class imbalance:

| Model | Approach | Key Parameter |
|-------|----------|---------------|
| Naive SVM | No imbalance handling | Default SVM |
| Weighted SVM | Class weight adjustment | `class_weight='balanced'` |
| SMOTE + SVM | Synthetic oversampling | SMOTE resampling |

### 3.2 Results Summary

![Metrics Comparison](../../figures/metrics_comparison.png)
*Figure 2: Comprehensive metrics comparison across all three models.*

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive SVM | 97.4% | 100.0% | 92.9% | 96.3% |
| Weighted SVM | 98.2% | 97.6% | 97.6% | 97.6% |
| SMOTE + SVM | 97.4% | 97.6% | 95.2% | 96.4% |

### 3.3 Confusion Matrix Analysis

![Confusion Matrices](../../figures/confusion_matrices_comparison.png)
*Figure 3: Side-by-side confusion matrices reveal the critical difference—false negatives.*

## 4. Resampling Strategies

### 4.1 SMOTE (Synthetic Minority Over-sampling Technique)

**Mechanism**: Generate synthetic samples by interpolating between existing minority class instances.

```python
from imblearn.over_sampling import SMOTE
resampler = SMOTE(random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
```

**Training set transformation**:
- Original: 286 benign, 169 malignant
- After SMOTE: 286 benign, 286 malignant

**Strengths**:
- No information loss (doesn't discard majority samples)
- Creates plausible interpolations in feature space

**Weaknesses**:
- May create noisy samples in sparse regions
- Assumes minority class is convex

### 4.2 Class Weight Adjustment

**Mechanism**: Modify SVM objective to penalize minority class errors more heavily.

```python
model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
```

**Strengths**:
- No synthetic data generation
- Preserves original data distribution
- Computationally efficient

## 5. ROC and Precision-Recall Analysis

### 5.1 ROC Curves

![ROC Curves](../../figures/roc_curves.png)
*Figure 4: ROC curves with AUC scores. All models achieve high AUC, but this metric doesn't capture cost asymmetry.*

### 5.2 Precision-Recall Curves

![Precision-Recall Curves](../../figures/precision_recall_curves.png)
*Figure 5: Precision-Recall curves. For medical screening, focus on the high-recall region.*

## 6. Cost-Weighted Evaluation

### 6.1 Cost Function Definition

```python
def total_cost(confusion_matrix, cost_fn=50000, cost_fp=500):
    fn = confusion_matrix[1, 0]
    fp = confusion_matrix[0, 1]
    return fn * cost_fn + fp * cost_fp
```

### 6.2 Cost Comparison

![Cost Comparison](../../figures/cost_comparison.png)
*Figure 6: Total cost breakdown by model. The Weighted SVM achieves lowest total cost.*

| Model | False Negatives | False Positives | Total Cost |
|-------|-----------------|-----------------|------------|
| Naive SVM | 3 | 0 | $150,000 |
| Weighted SVM | 1 | 1 | $50,500 |
| SMOTE + SVM | 2 | 1 | $100,500 |

**Selection**: Weighted SVM minimizes total cost.

## 7. Model Validation

### 7.1 Cross-Validation

![Cross-Validation](../../figures/cross_validation_recall.png)
*Figure 7: 5-fold cross-validation recall scores with mean and standard deviation.*

Cross-validation results confirm model stability:
- Weighted SVM shows consistently high recall
- Low variance indicates robust generalization

### 7.2 Learning Curves

![Learning Curves](../../figures/learning_curves.png)
*Figure 8: Learning curves diagnose bias/variance trade-off. Convergence indicates no overfitting.*

The learning curves show:
- Training and validation scores converge
- Model could benefit from additional data
- No evidence of overfitting

## 8. Implementation Architecture

```
src/
└── model.py
    ├── load_data()                    # Data loading and preprocessing
    ├── plot_class_distribution()      # Imbalance visualization
    ├── train_and_evaluate_model()     # Model training pipeline
    ├── plot_confusion_matrix()        # Confusion matrix heatmaps
    ├── plot_roc_curves()              # ROC curve analysis
    ├── plot_precision_recall_curves() # PR curve analysis
    ├── plot_metrics_comparison()      # Metrics bar charts
    ├── plot_cost_comparison()         # Cost analysis visualization
    ├── plot_cross_validation_scores() # CV boxplots
    ├── plot_learning_curves()         # Bias/variance diagnosis
    └── run_comparison()               # Full benchmark execution
```

**Design Decisions**:
1. Single file for simplicity (portfolio demonstration)
2. Pure functions with explicit I/O (testable, no hidden state)
3. Cost analysis decoupled from model training (separates concerns)
4. Comprehensive visualization suite for stakeholder communication

## 9. Reproducibility

```bash
# Environment
Python 3.9+
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.5.0
seaborn>=0.12.0

# Execution
python src/model.py
```

Random seeds fixed at 42 for all stochastic operations.

## 10. Limitations and Future Work

### Current Limitations

1. **Single dataset**: Results may not generalize to other medical imaging tasks
2. **Binary classification**: Real diagnoses include multiple tumor grades
3. **Static threshold**: Production system should calibrate threshold per deployment site

### Recommended Extensions

1. **Ensemble methods**: Combine multiple resampling strategies
2. **Calibration**: Ensure probability outputs reflect true class frequencies
3. **Monitoring**: Track recall degradation over time in production
4. **Explainability**: Add SHAP values for feature importance

## 11. Conclusion

This analysis demonstrates that:

1. Accuracy is inappropriate for imbalanced medical classification
2. Class weight adjustment achieves best cost-performance trade-off
3. Cost-weighted evaluation aligns model selection with business objectives
4. A focused recall optimization saves $99,500 per 100 patients

The key engineering insight: **metrics encode values**. Choosing the right metric is as important as choosing the right algorithm.
