# Mathematical Foundations

This document provides formal mathematical definitions for the methods and metrics used in the imbalanced classification case study.

---

## 1. Problem Formalization

### Binary Classification

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ where:
- $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector (d = 30 cell nucleus measurements)
- $y_i \in \{0, 1\}$ is the class label (0 = benign, 1 = malignant)

The goal is to learn a function $f: \mathbb{R}^d \rightarrow \{0, 1\}$ that minimizes expected loss.

### Class Imbalance

Let $n_0$ and $n_1$ denote the number of samples in each class. The imbalance ratio is:

$$\rho = \frac{n_0}{n_1} = \frac{357}{212} \approx 1.68$$

---

## 2. Support Vector Machine (SVM)

### Primal Formulation

For linearly separable data, SVM solves:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

subject to:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i$$

### Soft-Margin SVM

For non-separable data, introduce slack variables $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

subject to:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \forall i$$

where $C > 0$ is the regularization parameter controlling the trade-off between margin maximization and classification error.

### Dual Formulation

The dual problem is:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

subject to:

$$\sum_{i=1}^{n} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C$$

### Kernel Extension (RBF)

Replace inner product with kernel function:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

where $\gamma = \frac{1}{2\sigma^2}$ controls the kernel width.

Decision function:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

where $SV$ is the set of support vectors (points with $\alpha_i > 0$).

---

## 3. SMOTE Algorithm

### Synthetic Minority Over-sampling Technique

For each minority class sample $\mathbf{x}_i$:

1. Find k nearest neighbors in minority class: $\{\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_k}\}$

2. Randomly select one neighbor $\mathbf{x}_{i_j}$

3. Generate synthetic sample:

$$\mathbf{x}_{new} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{i_j} - \mathbf{x}_i)$$

where $\lambda \sim \text{Uniform}(0, 1)$

### Oversampling Rate

To balance classes, generate $N$ synthetic samples where:

$$N = n_0 - n_1 = 357 - 212 = 145$$

After SMOTE: $n_0 = n_1 = 357$

---

## 4. Tomek Links

### Definition

A pair $(\mathbf{x}_i, \mathbf{x}_j)$ forms a Tomek link if:
- $y_i \neq y_j$ (different classes)
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_i, \mathbf{x}_k)$ for all $\mathbf{x}_k$ with $y_k = y_j$
- $d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_j, \mathbf{x}_l)$ for all $\mathbf{x}_l$ with $y_l = y_i$

where $d(\cdot, \cdot)$ is Euclidean distance.

### Cleaning Strategy

Remove majority class samples that participate in Tomek links, cleaning the decision boundary.

---

## 5. Evaluation Metrics

### Confusion Matrix

$$\begin{array}{c|cc}
& \hat{y}=0 & \hat{y}=1 \\
\hline
y=0 & TN & FP \\
y=1 & FN & TP
\end{array}$$

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Limitation**: For imbalanced data, accuracy is dominated by majority class performance.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP} = P(\text{correct} | \text{predicted positive})$$

### Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN} = P(\text{detected} | \text{actually positive})$$

### Specificity (True Negative Rate)

$$\text{Specificity} = \frac{TN}{TN + FP} = P(\text{detected} | \text{actually negative})$$

### F1 Score

Harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

### F-beta Score

Generalization with recall weighting:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta > 1$: emphasizes recall
- $\beta < 1$: emphasizes precision
- $\beta = 2$: recall is twice as important as precision

---

## 6. Cost-Sensitive Evaluation

### Expected Cost

Given cost matrix:

$$\begin{array}{c|cc}
& \hat{y}=0 & \hat{y}=1 \\
\hline
y=0 & 0 & c_{FP} \\
y=1 & c_{FN} & 0
\end{array}$$

Total expected cost:

$$\mathcal{L} = c_{FN} \cdot FN + c_{FP} \cdot FP$$

### Cost Ratio

$$r = \frac{c_{FN}}{c_{FP}} = \frac{50000}{500} = 100$$

### Optimal Threshold

For probabilistic classifier $P(y=1|\mathbf{x})$, the cost-optimal threshold is:

$$\tau^* = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{500}{500 + 50000} \approx 0.01$$

This suggests classifying as malignant when $P(y=1|\mathbf{x}) > 0.01$.

---

## 7. ROC Analysis

### ROC Curve

Plot of True Positive Rate vs False Positive Rate across all thresholds:

$$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}$$

$$\text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

### AUC (Area Under Curve)

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

Equivalent to probability that a random positive sample is ranked higher than a random negative sample:

$$\text{AUC} = P(f(\mathbf{x}^+) > f(\mathbf{x}^-))$$

---

## 8. Class Weighting in SVM

### Weighted Soft-Margin

Modify the objective to penalize minority class errors more:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C_0 \sum_{i: y_i=0} \xi_i + C_1 \sum_{i: y_i=1} \xi_i$$

where:

$$C_1 = C \cdot \frac{n_0}{n_1}, \quad C_0 = C$$

This is equivalent to `class_weight='balanced'` in scikit-learn.

---

## 9. Statistical Significance

### McNemar's Test

For comparing two classifiers on the same test set:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where:
- $b$ = samples correctly classified by model 1 but not model 2
- $c$ = samples correctly classified by model 2 but not model 1

Under $H_0$ (equal performance), $\chi^2 \sim \chi^2_1$.

---

## Summary Table

| Metric | Formula | Range | Optimal |
|--------|---------|-------|---------|
| Accuracy | $(TP+TN)/N$ | [0, 1] | 1 |
| Precision | $TP/(TP+FP)$ | [0, 1] | 1 |
| Recall | $TP/(TP+FN)$ | [0, 1] | 1 |
| F1 | $2PR/(P+R)$ | [0, 1] | 1 |
| AUC | $\int \text{ROC}$ | [0.5, 1] | 1 |
| Cost | $c_{FN} \cdot FN + c_{FP} \cdot FP$ | $[0, \infty)$ | 0 |
