# Executive Summary: Cost-Aware Medical Classification

## Business Problem

A hospital system needed an ML model to assist with breast cancer screening. The standard approach—optimizing for accuracy—would have deployed a model that systematically missed malignant tumors.

## Key Insight

**Accuracy is the wrong metric for medical diagnosis.**

A model predicting "benign" for every patient achieves 63% accuracy while missing 100% of cancers. When class costs are asymmetric, accuracy optimization produces harmful outcomes.

![Class Distribution](../../figures/class_distribution.png)
*The dataset imbalance: 357 benign vs 212 malignant samples creates hidden bias.*

## Results

| Metric | Naive SVM | Weighted SVM | SMOTE + SVM |
|--------|-----------|--------------|-------------|
| Accuracy | 97.4% | 98.2% | 97.4% |
| Recall | 92.9% | 97.6% | 95.2% |
| Missed Cancers | 3 | 1 | 2 |
| Estimated Cost | $150,000 | $50,500 | $100,500 |

The Weighted SVM saves **$99,500 per 100 patients** and catches 2 additional cancers compared to the naive approach.

![Cost Comparison](../../figures/cost_comparison.png)
*Cost breakdown: False negatives (missed cancers) dominate total cost.*

## Visual Evidence

### Model Performance Comparison

![Metrics Comparison](../../figures/metrics_comparison.png)
*While accuracy appears similar across models, recall—the metric that matters for patient safety—varies significantly.*

### Confusion Matrix Analysis

![Confusion Matrices](../../figures/confusion_matrices_comparison.png)
*The confusion matrices reveal what accuracy hides: the Naive SVM misses 3 cancers, while the Weighted SVM misses only 1.*

## Technical Approach

1. Identified class imbalance (1.68:1 benign to malignant ratio)
2. Quantified asymmetric error costs (false negative: $50,000 vs false positive: $500)
3. Compared three approaches: naive SVM, weighted SVM, and SMOTE resampling
4. Evaluated models using recall and cost-weighted metrics instead of accuracy

## Model Validation

![Cross-Validation](../../figures/cross_validation_recall.png)
*5-fold cross-validation confirms the Weighted SVM maintains high recall consistently across data splits.*

## Skills Demonstrated

- **Decision Analysis**: Translating confusion matrices into business costs
- **Risk Assessment**: Identifying failure modes in standard ML pipelines
- **Stakeholder Communication**: Explaining why "better" metrics can mean worse outcomes
- **Domain Awareness**: Understanding medical and legal implications of classification errors
- **Visual Analytics**: Creating compelling visualizations that tell the cost story

## Recommendation

For any classification task with asymmetric error costs:
1. Define the cost ratio before model selection
2. Set minimum recall thresholds based on domain requirements
3. Report confusion matrices, not single-number scores
4. Validate that accuracy improvements don't come at the expense of critical minority cases

---

*This case study demonstrates that effective ML deployment requires understanding the business context, not just optimizing metrics.*
