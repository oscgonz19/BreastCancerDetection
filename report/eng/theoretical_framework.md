# Theoretical Framework: Imbalanced Classification in Medical Diagnosis

A comprehensive academic reference covering the statistical and machine learning foundations required to understand cost-aware classification for imbalanced datasets.

---

## Table of Contents

1. [Statistical Learning Theory](#1-statistical-learning-theory)
2. [Binary Classification](#2-binary-classification)
3. [Evaluation Metrics and Their Limitations](#3-evaluation-metrics-and-their-limitations)
4. [The Class Imbalance Problem](#4-the-class-imbalance-problem)
5. [Support Vector Machines](#5-support-vector-machines)
6. [Resampling Methods](#6-resampling-methods)
7. [Cost-Sensitive Learning](#7-cost-sensitive-learning)
8. [Decision Theory and Risk](#8-decision-theory-and-risk)
9. [Model Selection and Validation](#9-model-selection-and-validation)
10. [References](#10-references)

---

## 1. Statistical Learning Theory

### 1.1 The Learning Problem

Statistical learning theory provides the mathematical foundation for machine learning. The fundamental problem can be stated as follows:

Given a training set $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$ drawn independently from an unknown probability distribution $P(\mathbf{X}, Y)$, find a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that minimizes the expected risk.

### 1.2 Risk and Loss Functions

The **expected risk** (or true risk) of a hypothesis $f$ is defined as:

$$R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}[L(y, f(\mathbf{x}))]$$

where $L(y, \hat{y})$ is a loss function measuring the cost of predicting $\hat{y}$ when the true label is $y$.

Common loss functions include:

| Loss Function | Formula | Use Case |
|---------------|---------|----------|
| 0-1 Loss | $L(y, \hat{y}) = \mathbb{1}[y \neq \hat{y}]$ | Classification |
| Squared Loss | $L(y, \hat{y}) = (y - \hat{y})^2$ | Regression |
| Hinge Loss | $L(y, \hat{y}) = \max(0, 1 - y\hat{y})$ | SVM |
| Log Loss | $L(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$ | Logistic Regression |

### 1.3 Empirical Risk Minimization

Since $P(\mathbf{X}, Y)$ is unknown, we approximate the expected risk with the **empirical risk**:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(\mathbf{x}_i))$$

The principle of **Empirical Risk Minimization (ERM)** states that we should choose the hypothesis that minimizes empirical risk:

$$f^* = \arg\min_{f \in \mathcal{H}} \hat{R}(f)$$

where $\mathcal{H}$ is the hypothesis space.

### 1.4 Bias-Variance Tradeoff

The expected prediction error can be decomposed into three components:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$

where:
- **Bias**: Error from erroneous assumptions in the learning algorithm
- **Variance**: Error from sensitivity to fluctuations in the training set
- **Irreducible error** ($\sigma^2$): Noise inherent in the problem

This tradeoff is fundamental: complex models have low bias but high variance (overfitting), while simple models have high bias but low variance (underfitting).

### 1.5 Generalization and VC Dimension

The **Vapnik-Chervonenkis (VC) dimension** measures the capacity of a hypothesis space. For a hypothesis space $\mathcal{H}$, the VC dimension is the largest number of points that can be shattered (perfectly separated) by $\mathcal{H}$.

The generalization bound states that with probability at least $1 - \delta$:

$$R(f) \leq \hat{R}(f) + \sqrt{\frac{h(\log(2n/h) + 1) - \log(\delta/4)}{n}}$$

where $h$ is the VC dimension and $n$ is the sample size.

---

## 2. Binary Classification

### 2.1 Problem Definition

In binary classification, we have:
- Input space: $\mathcal{X} \subseteq \mathbb{R}^d$
- Output space: $\mathcal{Y} = \{0, 1\}$ or $\{-1, +1\}$
- Goal: Learn $f: \mathcal{X} \rightarrow \mathcal{Y}$

### 2.2 Decision Boundaries

A classifier partitions the input space into decision regions. The **decision boundary** is the surface where:

$$P(Y = 1 | \mathbf{X} = \mathbf{x}) = P(Y = 0 | \mathbf{X} = \mathbf{x}) = 0.5$$

For linear classifiers, this boundary is a hyperplane:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

### 2.3 Probabilistic Classification

Many classifiers output probability estimates $\hat{p}(\mathbf{x}) = P(Y = 1 | \mathbf{X} = \mathbf{x})$. The final prediction is made by thresholding:

$$\hat{y} = \begin{cases} 1 & \text{if } \hat{p}(\mathbf{x}) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

The threshold $\tau$ (typically 0.5) can be adjusted to trade off between different types of errors.

### 2.4 Bayes Optimal Classifier

The **Bayes optimal classifier** minimizes the expected 0-1 loss:

$$f^*(\mathbf{x}) = \arg\max_{y \in \{0, 1\}} P(Y = y | \mathbf{X} = \mathbf{x})$$

This classifier achieves the lowest possible error rate, called the **Bayes error rate**:

$$R^* = \mathbb{E}_{\mathbf{x}}[\min(P(Y=1|\mathbf{x}), P(Y=0|\mathbf{x}))]$$

---

## 3. Evaluation Metrics and Their Limitations

### 3.1 The Confusion Matrix

For binary classification, predictions fall into four categories:

|  | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

### 3.2 Basic Metrics

**Accuracy**:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Error Rate**:
$$\text{Error Rate} = 1 - \text{Accuracy} = \frac{FP + FN}{TP + TN + FP + FN}$$

### 3.3 Class-Specific Metrics

**Precision** (Positive Predictive Value):
$$\text{Precision} = \frac{TP}{TP + FP} = P(\text{Actual Positive} | \text{Predicted Positive})$$

**Recall** (Sensitivity, True Positive Rate):
$$\text{Recall} = \frac{TP}{TP + FN} = P(\text{Predicted Positive} | \text{Actual Positive})$$

**Specificity** (True Negative Rate):
$$\text{Specificity} = \frac{TN}{TN + FP} = P(\text{Predicted Negative} | \text{Actual Negative})$$

**False Positive Rate**:
$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

### 3.4 Combined Metrics

**F1 Score** (Harmonic mean of precision and recall):
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**F-beta Score** (Weighted harmonic mean):
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta > 1$: Emphasizes recall
- $\beta < 1$: Emphasizes precision
- $\beta = 2$: Recall is twice as important as precision

**Matthews Correlation Coefficient**:
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

MCC ranges from -1 (perfect misclassification) to +1 (perfect classification), with 0 indicating random performance.

### 3.5 Threshold-Independent Metrics

**ROC Curve**: Plot of TPR vs FPR across all thresholds.

**AUC-ROC** (Area Under ROC Curve):
$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

Probabilistic interpretation:
$$\text{AUC} = P(\hat{p}(\mathbf{x}^+) > \hat{p}(\mathbf{x}^-))$$

where $\mathbf{x}^+$ is a random positive sample and $\mathbf{x}^-$ is a random negative sample.

**Precision-Recall Curve**: Plot of Precision vs Recall across all thresholds.

**Average Precision**:
$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

### 3.6 Why Accuracy Fails for Imbalanced Data

Consider a dataset with 95% negative and 5% positive samples. A trivial classifier predicting "negative" for all inputs achieves:

- Accuracy: 95%
- Precision: undefined (0/0)
- Recall: 0%
- F1: 0%

**Key insight**: Accuracy is dominated by majority class performance and can hide complete failure on the minority class.

---

## 4. The Class Imbalance Problem

### 4.1 Definition and Prevalence

Class imbalance occurs when the class distribution is significantly skewed:

$$\frac{n_{\text{majority}}}{n_{\text{minority}}} >> 1$$

Imbalance ratios in real applications:
- Medical diagnosis: 10:1 to 1000:1
- Fraud detection: 100:1 to 10000:1
- Defect detection: 50:1 to 500:1

### 4.2 Why Imbalance Causes Problems

Standard learning algorithms assume balanced class distributions. With imbalanced data:

1. **Prior probability shift**: The classifier learns to favor the majority class
2. **Boundary distortion**: Decision boundaries shift toward the minority class
3. **Metric deception**: Accuracy appears high while minority class performance is poor
4. **Insufficient representation**: Too few minority samples to learn the underlying distribution

### 4.3 Types of Imbalance

**Between-class imbalance**: Unequal number of samples per class.

**Within-class imbalance**: The minority class contains multiple sub-concepts with unequal representation.

**Relative imbalance** vs **Absolute rarity**: A 100:1 ratio with 10,000 minority samples is different from 100:1 with 100 minority samples.

### 4.4 The Small Disjuncts Problem

In imbalanced datasets, minority class samples often form small, disconnected clusters (**disjuncts**). These are difficult to learn because:

- Small sample size leads to unreliable estimates
- Classifiers may treat them as noise
- Error rates on small disjuncts are disproportionately high

---

## 5. Support Vector Machines

### 5.1 Linear SVM: Maximum Margin Classifier

Given linearly separable data, SVM finds the hyperplane that maximizes the margin between classes.

**Primal formulation**:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

The margin width is $\frac{2}{\|\mathbf{w}\|}$, so minimizing $\|\mathbf{w}\|^2$ maximizes the margin.

### 5.2 Soft-Margin SVM

For non-separable data, introduce slack variables $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

The parameter $C$ controls the trade-off:
- Large $C$: Less tolerance for misclassification (may overfit)
- Small $C$: More tolerance for misclassification (may underfit)

### 5.3 Dual Formulation

Using Lagrangian duality, the problem becomes:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

subject to:
$$\sum_{i=1}^{n} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C$$

The solution depends only on inner products $\mathbf{x}_i^T \mathbf{x}_j$, enabling the kernel trick.

### 5.4 Kernel Methods

Replace inner products with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$:

**Linear kernel**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$

**Polynomial kernel**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$

**Radial Basis Function (RBF) kernel**:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

The RBF kernel maps data to an infinite-dimensional space, enabling complex decision boundaries.

### 5.5 Decision Function

The classification decision is:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

where $SV$ is the set of support vectors (samples with $\alpha_i > 0$).

### 5.6 SVM and Class Imbalance

Standard SVM optimization treats all errors equally. With imbalanced data:
- The margin is pushed toward the minority class
- Minority class samples are more likely to be misclassified
- The classifier favors the majority class

**Solutions**:
1. Different misclassification costs per class
2. Resampling before training
3. Adjusting the decision threshold

---

## 6. Resampling Methods

### 6.1 Overview of Approaches

Resampling methods modify the training data to address class imbalance:

| Approach | Method | Effect |
|----------|--------|--------|
| Oversampling | Add minority samples | Increases minority representation |
| Undersampling | Remove majority samples | Reduces majority dominance |
| Hybrid | Combine both | Balances trade-offs |

### 6.2 Random Oversampling

Randomly duplicate minority class samples until classes are balanced.

**Algorithm**:
1. Calculate the difference: $N = n_{\text{majority}} - n_{\text{minority}}$
2. Randomly select $N$ samples from minority class (with replacement)
3. Add copies to training set

**Advantages**: Simple, preserves information

**Disadvantages**: Exact copies lead to overfitting; decision boundaries become overly specific

### 6.3 Random Undersampling

Randomly remove majority class samples until classes are balanced.

**Algorithm**:
1. Calculate target: $N = n_{\text{minority}}$
2. Randomly select $N$ samples from majority class
3. Discard remaining majority samples

**Advantages**: Reduces training time, may remove noisy samples

**Disadvantages**: Loses potentially useful information; may discard important majority samples near the boundary

### 6.4 SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generates synthetic minority samples by interpolation.

**Algorithm**:
```
For each minority sample x_i:
    1. Find k nearest minority neighbors
    2. Randomly select one neighbor x_j
    3. Generate synthetic sample:
       x_new = x_i + λ(x_j - x_i)
       where λ ~ Uniform(0, 1)
```

**Mathematical formulation**:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_j - \mathbf{x}_i), \quad \lambda \in [0, 1]$$

**Advantages**:
- Creates new, plausible samples
- Reduces overfitting compared to random oversampling
- Expands the minority class region

**Disadvantages**:
- May create noisy samples if minority class is not convex
- Can generate samples in majority class regions (overlap)
- Assumes feature space is meaningful for interpolation

### 6.5 SMOTE Variants

**Borderline-SMOTE**: Only oversample minority samples near the decision boundary.

**SMOTE-ENN**: Apply SMOTE, then clean with Edited Nearest Neighbors.

**ADASYN** (Adaptive Synthetic Sampling): Generate more synthetic samples for minority instances that are harder to classify.

### 6.6 Tomek Links

A Tomek link is a pair of samples $(\mathbf{x}_i, \mathbf{x}_j)$ where:
- $y_i \neq y_j$ (different classes)
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_i, \mathbf{x}_k)$ for all $k$ with $y_k = y_j$
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_j, \mathbf{x}_l)$ for all $l$ with $y_l = y_i$

In words: each sample is the other's nearest neighbor from the opposite class.

**Usage**:
- Remove majority samples in Tomek links (cleans boundary)
- Remove both samples (removes ambiguous regions)

**Limitation**: Only affects boundary samples; does not address core imbalance.

### 6.7 Edited Nearest Neighbors (ENN)

Remove samples whose class differs from the majority of their k nearest neighbors.

**Algorithm**:
```
For each sample (x_i, y_i):
    Find k nearest neighbors
    If majority of neighbors have class ≠ y_i:
        Remove (x_i, y_i)
```

This cleans noisy and borderline samples from both classes.

### 6.8 Hybrid Methods

**SMOTE + Tomek Links**: Apply SMOTE, then remove Tomek links to clean the boundary.

**SMOTE + ENN**: Apply SMOTE, then apply ENN to remove noisy synthetic samples.

---

## 7. Cost-Sensitive Learning

### 7.1 Motivation

In many applications, different types of errors have different costs:

| Application | Costly Error | Less Costly Error |
|-------------|--------------|-------------------|
| Cancer screening | False Negative (missed cancer) | False Positive (unnecessary biopsy) |
| Spam filtering | False Positive (lost email) | False Negative (spam in inbox) |
| Fraud detection | False Negative (undetected fraud) | False Positive (blocked transaction) |

### 7.2 Cost Matrix

Define costs for each outcome:

|  | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** | $C_{TN}$ (usually 0) | $C_{FP}$ |
| **Actual Positive** | $C_{FN}$ | $C_{TP}$ (usually 0) |

### 7.3 Expected Cost

The expected cost of a classifier is:

$$\mathbb{E}[\text{Cost}] = C_{FP} \cdot FP + C_{FN} \cdot FN + C_{TP} \cdot TP + C_{TN} \cdot TN$$

With $C_{TP} = C_{TN} = 0$:

$$\mathbb{E}[\text{Cost}] = C_{FP} \cdot FP + C_{FN} \cdot FN$$

### 7.4 Cost-Sensitive Learning Approaches

**1. Resampling to reflect costs**:

Oversample minority class by factor proportional to cost ratio:
$$\text{Oversampling ratio} = \frac{C_{FN}}{C_{FP}}$$

**2. Cost-sensitive algorithms**:

Modify the learning algorithm to incorporate costs directly:
$$\min_{\mathbf{w}} \sum_{i: y_i = 0} C_{FP} \cdot L_i + \sum_{i: y_i = 1} C_{FN} \cdot L_i$$

**3. Threshold adjustment**:

For probabilistic classifiers, the cost-optimal threshold is:
$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

### 7.5 Class Weighting in SVM

Modify the soft-margin objective:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C_0 \sum_{i: y_i=0} \xi_i + C_1 \sum_{i: y_i=1} \xi_i$$

Setting $C_1 > C_0$ penalizes false negatives more heavily.

Common heuristic (balanced weighting):
$$C_k = C \cdot \frac{n}{2 \cdot n_k}$$

where $n$ is total samples and $n_k$ is samples in class $k$.

---

## 8. Decision Theory and Risk

### 8.1 Statistical Decision Theory

A **decision rule** $\delta: \mathcal{X} \rightarrow \mathcal{A}$ maps observations to actions. In classification:
- Observations: Feature vectors $\mathbf{x}$
- Actions: Class predictions $\{0, 1\}$

### 8.2 Loss Functions and Risk

The **loss function** $L(y, a)$ quantifies the cost of taking action $a$ when the true state is $y$.

The **risk** of a decision rule is its expected loss:
$$R(\delta) = \mathbb{E}[L(Y, \delta(\mathbf{X}))]$$

### 8.3 Bayes Risk and Optimal Decisions

For a given prior $P(Y)$ and likelihood $P(\mathbf{X}|Y)$, the **Bayes optimal decision** minimizes expected loss:

$$\delta^*(\mathbf{x}) = \arg\min_a \sum_y L(y, a) P(Y = y | \mathbf{X} = \mathbf{x})$$

The **Bayes risk** is the minimum achievable risk:
$$R^* = \mathbb{E}[L(Y, \delta^*(\mathbf{X}))]$$

### 8.4 Cost-Sensitive Bayes Optimal Decision

With asymmetric costs, the optimal decision rule is:

Predict $Y = 1$ if:
$$\frac{P(Y = 1 | \mathbf{x})}{P(Y = 0 | \mathbf{x})} > \frac{C_{FP}}{C_{FN}}$$

Equivalently:
$$P(Y = 1 | \mathbf{x}) > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

### 8.5 Risk Analysis in Medical Diagnosis

For cancer screening with:
- $C_{FN} = \$50,000$ (missed cancer)
- $C_{FP} = \$500$ (unnecessary biopsy)

The cost ratio is $100:1$, so the optimal threshold is:
$$\tau^* = \frac{500}{500 + 50000} = 0.0099 \approx 1\%$$

This means: predict malignant if $P(\text{malignant}|\mathbf{x}) > 1\%$.

---

## 9. Model Selection and Validation

### 9.1 The Holdout Method

Split data into training and test sets:
- Training set: Used to fit the model
- Test set: Used to estimate generalization performance

**Limitation**: Wastes data; estimate has high variance with small datasets.

### 9.2 K-Fold Cross-Validation

1. Partition data into $K$ equal folds
2. For $k = 1, \ldots, K$:
   - Train on folds $\{1, \ldots, K\} \setminus \{k\}$
   - Test on fold $k$
3. Average performance across folds

**K-fold CV estimate**:
$$\hat{R}_{CV} = \frac{1}{K} \sum_{k=1}^{K} \hat{R}_k$$

Common choices: $K = 5$ or $K = 10$.

### 9.3 Stratified Cross-Validation

For imbalanced data, use **stratified** sampling to preserve class ratios in each fold.

Without stratification, some folds may have very few or no minority samples.

### 9.4 Nested Cross-Validation

For model selection with hyperparameter tuning:

**Outer loop**: Estimate generalization performance
**Inner loop**: Select hyperparameters

This prevents information leakage from test set to model selection.

### 9.5 Statistical Comparison of Classifiers

**McNemar's Test**: Compare two classifiers on the same test set.

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

where:
- $n_{01}$: Samples correct by classifier 1, incorrect by classifier 2
- $n_{10}$: Samples correct by classifier 2, incorrect by classifier 1

Under $H_0$ (equal performance): $\chi^2 \sim \chi^2_1$

### 9.6 Calibration

A classifier is **well-calibrated** if:
$$P(Y = 1 | \hat{p}(\mathbf{x}) = p) = p$$

Calibration can be assessed using:
- Reliability diagrams
- Expected Calibration Error (ECE)
- Brier score

**Platt scaling** and **isotonic regression** are common post-hoc calibration methods.

---

## 10. References

### Foundational Texts

1. Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

### Class Imbalance

4. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

5. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

6. Tomek, I. (1976). Two Modifications of CNN. *IEEE Transactions on Systems, Man, and Cybernetics*, 6(11), 769-772.

### Cost-Sensitive Learning

7. Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973-978.

8. Domingos, P. (1999). MetaCost: A General Method for Making Classifiers Cost-Sensitive. *Proceedings of the 5th International Conference on Knowledge Discovery and Data Mining*, 155-164.

### Support Vector Machines

9. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.

10. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.

### Evaluation Metrics

11. Powers, D. M. (2011). Evaluation: From Precision, Recall and F-measure to ROC, Informedness, Markedness and Correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

12. Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning*, 233-240.

---

## Summary

This theoretical framework establishes that:

1. **Statistical learning theory** provides bounds on generalization but assumes certain loss functions
2. **Accuracy is not a universal metric**—it encodes the assumption that all errors are equally costly
3. **Class imbalance** breaks standard assumptions and requires specialized techniques
4. **Resampling methods** (SMOTE, Tomek Links) address imbalance at the data level
5. **Cost-sensitive learning** addresses imbalance at the algorithm level
6. **Decision theory** provides the formal framework for choosing metrics that align with real-world costs
7. **Model selection** must use stratified validation and appropriate metrics

The key insight is that **metric selection is not a technical detail—it encodes values and priorities** that must be aligned with the application domain.
