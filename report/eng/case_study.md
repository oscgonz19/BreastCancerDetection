# The Accuracy Trap in Medical Diagnosis

## The Scenario

A hospital system wants to deploy a machine learning model to assist radiologists in breast cancer screening. The model will classify tumor samples as **malignant** or **benign** based on cell nucleus measurements from fine needle aspirates.

The data science team presents two models:
- **Model A**: 97% accuracy
- **Model B**: 94% accuracy

The hospital chooses Model A. This decision will cost lives.

---

## What Went Wrong

### The Hidden Asymmetry

The dataset contains **357 benign** and **212 malignant** samples—a 1.7:1 imbalance. This seems modest, but the consequences are severe.

![Class Distribution](../../figures/class_distribution.png)
*Figure 1: The class imbalance in our breast cancer dataset. While 1.68:1 appears moderate, it creates systematic bias toward the majority class.*

A trivial "model" that predicts **benign for every patient** achieves:
- **63% accuracy** (correctly classifies all benign cases)
- **0% recall** on malignant tumors
- **Misses 100% of cancers**

This exposes the core problem: accuracy treats all errors as equal.

### Why Accuracy Lies

| Error Type | What Happened | Consequence |
|------------|---------------|-------------|
| False Positive | Benign → Malignant | Unnecessary biopsy, patient anxiety |
| False Negative | Malignant → Benign | Delayed treatment, metastasis, death |

In medical screening, **false negatives kill patients**. A model optimizing accuracy will minimize total errors by biasing toward the majority class, systematically missing rare but critical cases.

---

## The Cost Function Reality

Medical and legal costs are asymmetric:

| Error | Estimated Cost |
|-------|----------------|
| False Positive | $500 (biopsy procedure) |
| False Negative | $50,000+ (treatment delay, litigation) |

This 100:1 ratio is conservative. Wrongful death lawsuits from missed diagnoses regularly exceed $1M.

### Model Comparison Under Real Costs

We compared three approaches: a naive SVM, a weighted SVM with class balancing, and SMOTE resampling with SVM.

![Cost Comparison](../../figures/cost_comparison.png)
*Figure 2: Total cost breakdown by model. The Weighted SVM achieves the lowest total cost by minimizing false negatives.*

| Model | Accuracy | False Negatives | False Positives | Total Cost |
|-------|----------|-----------------|-----------------|------------|
| Naive SVM | 97.4% | 3 | 0 | $150,000 |
| Weighted SVM | 98.2% | 1 | 1 | $50,500 |
| SMOTE + SVM | 97.4% | 2 | 1 | $100,500 |

**The Weighted SVM saves $99,500 per 100 patients compared to the Naive model.**

---

## Visualizing Model Failures

### Confusion Matrices: Where Errors Hide

The confusion matrix reveals what accuracy conceals:

![Confusion Matrices](../../figures/confusion_matrices_comparison.png)
*Figure 3: Side-by-side confusion matrices for all three models. The Naive SVM (left) shows 3 false negatives—three missed cancers. The Weighted SVM (center) reduces this to just 1.*

Each false negative in the upper-right quadrant represents a patient with cancer who was told they were healthy.

### ROC and Precision-Recall Analysis

![ROC Curves](../../figures/roc_curves.png)
*Figure 4: ROC curves showing discrimination ability. All models achieve high AUC, but this masks critical differences in the cost-sensitive region.*

![Precision-Recall Curves](../../figures/precision_recall_curves.png)
*Figure 5: Precision-Recall curves. For medical screening, we prioritize the high-recall region (right side of curve), accepting lower precision to catch more cancers.*

---

## What Goes Wrong With Each Approach

### Approach 1: Ignoring Imbalance

The standard SVM optimizes the decision boundary to minimize total misclassifications. With more benign samples, the boundary shifts toward predicting benign, sacrificing recall on malignant cases.

**Failure mode**: High accuracy, missed cancers.

### Approach 2: Random Undersampling

Discarding majority class samples to achieve balance.

**Failure mode**: With severe imbalance (e.g., 380 benign vs 17 malignant), undersampling discards 95% of data. The model lacks statistical power and overfits to the small remaining sample.

### Approach 3: SMOTE Oversampling

Synthesizing new minority samples by interpolating between existing ones.

**Failure mode**: SMOTE assumes the minority class forms convex clusters. If malignant tumors have multiple distinct subtypes, interpolation creates biologically implausible synthetic cases.

### Approach 4: Class Weights

Adjusting the loss function to penalize minority class errors more heavily.

**Success mode**: Achieves high recall without synthetic data generation, maintaining the original data distribution.

---

## Model Validation

### Cross-Validation Results

![Cross-Validation](../../figures/cross_validation_recall.png)
*Figure 6: 5-fold cross-validation recall scores. The Weighted SVM shows consistently high recall with low variance, indicating robust performance.*

### Learning Curves: Bias vs Variance

![Learning Curves](../../figures/learning_curves.png)
*Figure 7: Learning curves for the Weighted SVM. The convergence of training and validation scores indicates the model is not overfitting and could benefit from additional data.*

---

## Comprehensive Metrics Comparison

![Metrics Comparison](../../figures/metrics_comparison.png)
*Figure 8: Full metrics comparison across all models. While accuracy appears similar, recall (the critical metric for medical screening) varies significantly.*

---

## The Right Question

The data science team asked: *"Which model has higher accuracy?"*

They should have asked: *"What is the cost of each type of error, and how do we minimize total harm?"*

### Metric Selection for Imbalanced Medical Data

| Metric | What It Measures | When to Use |
|--------|------------------|-------------|
| **Recall** | % of actual positives correctly identified | When false negatives are costly |
| **Precision** | % of predicted positives that are correct | When false positives are costly |
| **F1 Score** | Harmonic mean of precision and recall | Balanced trade-off |
| **F2 Score** | Weighted F-score favoring recall | When recall matters more |
| **AUC-ROC** | Discrimination ability across thresholds | Comparing models overall |

For cancer screening: **optimize recall**, accept lower precision, monitor false positive rate for operational feasibility.

---

## Decision Framework

Before training any model on imbalanced data:

1. **Quantify the cost ratio** of false negatives vs false positives
2. **Set a minimum recall threshold** based on domain requirements
3. **Choose resampling strategy** based on dataset characteristics:
   - Small dataset + severe imbalance → SMOTE with care
   - Large dataset + moderate imbalance → class weights or Tomek Links
   - Multiple minority subgroups → cluster-based SMOTE variants
4. **Evaluate on cost-weighted metrics**, not raw accuracy
5. **Report confusion matrices**, not single-number scores

---

## Conclusion

The accuracy metric is dangerous in imbalanced classification because it:
- Hides systematic failures on minority classes
- Creates perverse incentives to ignore rare cases
- Gives false confidence to stakeholders unfamiliar with ML limitations

In medical diagnosis, the minority class is often the one that matters most. A model that achieves 99% accuracy by missing 100% of cancers is not a success—it's a liability.

**The lesson**: Metrics encode values. Choose metrics that encode *your* values, not the algorithm's convenience.
