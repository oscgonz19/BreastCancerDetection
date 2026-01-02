# ML Pipeline: Step-by-Step Walkthrough

This document explains the complete machine learning pipeline for imbalanced medical classification, with decision rationale at each stage.

---

## Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data Load  │───▶│  Imbalance  │───▶│  Resample   │───▶│   Train     │
│             │    │  Analysis   │    │  Strategy   │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│   Deploy    │◀───│    Cost     │◀───│  Evaluate   │◀─────────┘
│  Decision   │    │  Analysis   │    │  on Test    │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Stage 1: Data Loading and Exploration

### Input
- Raw CSV with 569 patient records
- 30 features from cell nucleus measurements
- Binary target: M (malignant) or B (benign)

### Operations

```python
def load_data(path: str = "data/breast_cancer.csv") -> tuple:
    df = pd.read_csv(path)
    X = df.drop(columns=["id", "diagnosis"])
    y = (df["diagnosis"] == "M").astype(int)
    return X, y
```

### Decision Point: Feature Selection

**Question**: Should we reduce dimensionality?

**Answer**: No for this dataset.
- 30 features is manageable for SVM
- Medical features are interpretable and may be useful for clinical review
- PCA would obscure feature importance

---

## Stage 2: Imbalance Analysis

### Operations

```python
def get_class_distribution(y) -> dict:
    benign = (y == 0).sum()
    malignant = (y == 1).sum()
    return {
        "benign": benign,
        "malignant": malignant,
        "imbalance_ratio": round(benign / malignant, 2)
    }
```

### Visual Output

![Class Distribution](../../figures/class_distribution.png)
*Figure 1: Class distribution visualization reveals the 1.68:1 imbalance ratio.*

### Output
```
benign: 357
malignant: 212
imbalance_ratio: 1.68
```

### Decision Point: Is Resampling Needed?

**Threshold heuristics**:
- Ratio < 1.5: Usually no resampling needed
- Ratio 1.5-3: Consider resampling based on cost asymmetry
- Ratio > 3: Resampling almost always required

**Our case**: 1.68 ratio + high cost asymmetry → **Resampling or class weights recommended**

---

## Stage 3: Train/Test Split

### Operations

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # Preserve class ratio in both sets
    random_state=42    # Reproducibility
)
```

### Decision Point: Split Ratio

**Question**: 80/20 or 70/30?

**Answer**: 80/20 is sufficient.
- 569 samples is small; need maximum training data
- Stratification ensures test set has representative malignant samples
- For production: use nested cross-validation

---

## Stage 4: Resampling Strategy

### Option A: Class Weights (Recommended)

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
```

**How it works**:
- Adjusts loss function to penalize minority class errors more heavily
- Weight inversely proportional to class frequency

**Strengths**:
- No synthetic data generation
- Preserves original data distribution
- Computationally efficient

### Option B: SMOTE

```python
from imblearn.over_sampling import SMOTE

resampler = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
```

**How SMOTE works**:
1. Select a minority sample
2. Find its k nearest minority neighbors (default k=5)
3. Create synthetic sample on line segment between them
4. Repeat until classes are balanced

**Strengths**:
- No information loss (doesn't discard majority samples)
- Creates plausible interpolations in feature space

**Weaknesses**:
- May create noisy samples in sparse regions
- Assumes minority class is convex

---

## Stage 5: Feature Scaling

### Operations

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training params!
```

### Critical: No Data Leakage

The scaler is fit **only** on training data. Test data is transformed using training statistics.

**Wrong**:
```python
scaler.fit(X_all)  # Leaks test distribution into training
```

**Right**:
```python
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Stage 6: Model Training

### Operations

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',           # Radial basis function for non-linear boundaries
    C=1.0,                  # Regularization (default)
    class_weight='balanced', # Handle imbalance
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

### Decision Point: Why SVM?

**Pros**:
- Effective in high-dimensional spaces (30 features)
- Memory efficient (stores only support vectors)
- Works well with clear margin of separation

**Cons**:
- Doesn't output calibrated probabilities by default
- Slower on very large datasets

**Alternative**: Random Forest or Logistic Regression for probability calibration.

---

## Stage 7: Evaluation

### Operations

```python
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
# [[TN, FP],
#  [FN, TP]]
```

### Visual Output

![Confusion Matrices](../../figures/confusion_matrices_comparison.png)
*Figure 2: Confusion matrices for all three models. Each cell shows the count of predictions.*

### Metrics Computed

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | "Of predicted malignant, how many are correct?" |
| Recall | TP/(TP+FN) | "Of actual malignant, how many did we find?" |
| F1 | 2×(Prec×Rec)/(Prec+Rec) | Harmonic mean |

![Metrics Comparison](../../figures/metrics_comparison.png)
*Figure 3: Metrics comparison across models. Note how accuracy masks differences in recall.*

### Critical: Why Recall Matters Most

In cancer screening:
- **False Negative** = Patient with cancer told they're healthy → delayed treatment → death
- **False Positive** = Healthy patient told to get biopsy → anxiety + $500

Recall directly measures the false negative rate.

---

## Stage 8: ROC and Precision-Recall Analysis

### ROC Curves

![ROC Curves](../../figures/roc_curves.png)
*Figure 4: ROC curves show discrimination ability. High AUC indicates good overall separation.*

### Precision-Recall Curves

![Precision-Recall Curves](../../figures/precision_recall_curves.png)
*Figure 5: PR curves are more informative for imbalanced data. Focus on high-recall region.*

---

## Stage 9: Cost Analysis

### Operations

```python
def cost_analysis(results: dict, cost_fn=50000, cost_fp=500) -> dict:
    fn = results["false_negatives"]
    fp = results["false_positives"]
    return {
        "total_cost": fn * cost_fn + fp * cost_fp
    }
```

### Visual Output

![Cost Comparison](../../figures/cost_comparison.png)
*Figure 6: Cost breakdown by model. False negative costs dominate total cost.*

### Decision Point: Setting Cost Ratio

**Question**: How to determine FN:FP cost ratio?

**Sources**:
1. **Medical literature**: Treatment delay costs, survival rates
2. **Legal data**: Average malpractice settlement for missed diagnosis
3. **Operational data**: Biopsy procedure costs

Our 100:1 ratio is conservative. Real medical settings may use 500:1 or higher.

---

## Stage 10: Model Validation

### Cross-Validation

![Cross-Validation](../../figures/cross_validation_recall.png)
*Figure 7: 5-fold cross-validation confirms model stability across data splits.*

### Learning Curves

![Learning Curves](../../figures/learning_curves.png)
*Figure 8: Learning curves diagnose bias/variance. Convergence indicates no overfitting.*

---

## Stage 11: Model Selection

### Comparison Matrix

| Model | Accuracy | Recall | Total Cost |
|-------|----------|--------|------------|
| Naive SVM | 97.4% | 92.9% | $150,000 |
| Weighted SVM | 98.2% | 97.6% | $50,500 |
| SMOTE + SVM | 97.4% | 95.2% | $100,500 |

### Decision

**Selected**: Weighted SVM

**Rationale**:
- Highest recall (97.6%)
- Lowest cost ($50,500)
- **66% cost reduction** vs naive approach

The accuracy metric would have been ambiguous—the cost metric is decisive.

---

## Stage 12: Production Considerations

### Threshold Tuning

Default threshold is 0.5. For high-stakes medical screening:

```python
# Get probability scores
model = SVC(probability=True)
y_scores = model.predict_proba(X_test)[:, 1]

# Lower threshold to catch more malignant cases
threshold = 0.3
y_pred = (y_scores >= threshold).astype(int)
```

### Monitoring

Track these metrics weekly in production:
1. Recall on confirmed malignant cases
2. False positive rate
3. Feature drift (are input distributions changing?)

### Retraining Triggers

- Recall drops below 95%
- New imaging equipment changes feature distributions
- New tumor subtypes identified

---

## Summary

The pipeline demonstrates that **metric selection is a design decision**, not a technical default. Each stage involves choices that must align with the business objective: minimizing patient harm, not maximizing a number on a dashboard.
