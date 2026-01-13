# ML Metrics That Matter
<img width="975" height="614" alt="image" src="https://github.com/user-attachments/assets/9f2bfae4-ac74-48f0-acb3-d6d9d65d6be5" />

<p align="right">
  <a href="README_ESP.md">🇪🇸 Leer en Español</a>
</p>

### When 97% Accuracy Costs Lives

<p align="center">
  <img src="figures/cost_comparison.png" alt="Cost Comparison" width="700"/>
</p>

<p align="center">
  <strong>The "best" model by accuracy costs 3x more than the "worst" one.</strong><br>
  <em>This repository explains why—and what to do about it.</em>
</p>

<p align="center">
  <a href="#the-story">The Story</a> •
  <a href="#key-results">Results</a> •
  <a href="#visualizations">Visualizations</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Docs</a>
</p>

---

## The Story

A hospital needs an ML model for breast cancer screening. The data science team presents their results:

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON                         │
├─────────────────┬──────────────┬──────────────┬────────────┤
│     Model       │   Accuracy   │    Recall    │    Cost    │
├─────────────────┼──────────────┼──────────────┼────────────┤
│  Naive SVM      │    97.4%     │    92.9%     │  $150,000  │
│  Weighted SVM   │    98.2%     │    97.6%     │   $50,500  │
│  SMOTE + SVM    │    97.4%     │    95.2%     │  $100,500  │
└─────────────────┴──────────────┴──────────────┴────────────┘
```

**The question:** Which model should the hospital deploy?

**The trap:** If you look only at accuracy, you might pick a model that misses more cancers.

**The insight:** The Weighted SVM saves **$99,500 per 100 patients** by catching 2 additional cancers.

---

## The Problem: Class Imbalance

<p align="center">
  <img src="figures/class_distribution.png" alt="Class Distribution" width="600"/>
</p>

The dataset has **357 benign** and **212 malignant** samples—a 1.68:1 ratio. This seems harmless, but:

> A trivial model predicting "benign" for everyone achieves **63% accuracy** while missing **100% of cancers**.

This is the **accuracy trap**: when classes have different costs, accuracy optimization produces harmful outcomes.

---

## The Real Cost of Errors

Not all errors are equal:

| Error Type | What Happens | Cost |
|------------|--------------|------|
| **False Positive** | Healthy patient gets biopsy | $500 |
| **False Negative** | Cancer patient sent home | $50,000+ |

This **100:1 cost ratio** means one missed cancer costs as much as 100 unnecessary biopsies.

---

## Key Results

### Confusion Matrices: Where Errors Hide

<p align="center">
  <img src="figures/confusion_matrices_comparison.png" alt="Confusion Matrices" width="800"/>
</p>

Each false negative (top-right of each matrix) represents a patient with cancer who was told they're healthy.

- **Naive SVM:** 3 missed cancers → $150,000 cost
- **Weighted SVM:** 1 missed cancer → $50,500 cost
- **SMOTE + SVM:** 2 missed cancers → $100,500 cost

### Performance Metrics

<p align="center">
  <img src="figures/metrics_comparison.png" alt="Metrics Comparison" width="700"/>
</p>

Notice how **accuracy looks similar** across models, but **recall varies significantly**. For medical screening, recall is what matters.

---

## ROC & Precision-Recall Analysis

<p align="center">
  <img src="figures/roc_curves.png" alt="ROC Curves" width="700"/>
</p>

All models achieve high AUC—but AUC doesn't capture cost asymmetry.

<p align="center">
  <img src="figures/precision_recall_curves.png" alt="Precision-Recall Curves" width="700"/>
</p>

For medical screening, we prioritize the **high-recall region** (right side), accepting lower precision to catch more cancers.

---

## Model Validation

### Cross-Validation

<p align="center">
  <img src="figures/cross_validation_recall.png" alt="Cross-Validation" width="700"/>
</p>

5-fold cross-validation confirms the Weighted SVM maintains **consistently high recall** with low variance.

### Learning Curves

<p align="center">
  <img src="figures/learning_curves.png" alt="Learning Curves" width="700"/>
</p>

The convergence of training and validation scores indicates **no overfitting**—the model generalizes well.

---

## The Decision Framework

```
┌──────────────────────────────────────────────────────────────────┐
│                 BEFORE TRAINING ON IMBALANCED DATA               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. QUANTIFY THE COST RATIO                                     │
│      └── How much worse is a false negative vs false positive?   │
│                                                                  │
│   2. SET MINIMUM RECALL THRESHOLDS                               │
│      └── What's the minimum acceptable detection rate?           │
│                                                                  │
│   3. CHOOSE YOUR STRATEGY                                        │
│      ├── Class weights (simple, preserves data)                  │
│      ├── SMOTE (synthetic oversampling)                          │
│      └── Tomek Links (boundary cleaning)                         │
│                                                                  │
│   4. EVALUATE ON COST-WEIGHTED METRICS                           │
│      └── Total Cost = (FN × Cost_FN) + (FP × Cost_FP)            │
│                                                                  │
│   5. REPORT CONFUSION MATRICES                                   │
│      └── Single scores hide critical failure modes               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/oscgonz19/ml-metrics-that-matter.git
cd ml-metrics-that-matter

# Install dependencies
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn

# Run the analysis
python src/model.py
```

**Output:**

```
============================================================
      BREAST CANCER CLASSIFICATION: THE ACCURACY TRAP
============================================================

                      Dataset Summary
────────────────────────────────────────────────────────────
  Benign samples:     357
  Malignant samples:  212
  Imbalance ratio:    1.68:1

------------------------------------------------------------
                       MODEL RESULTS
------------------------------------------------------------

Model:               Weighted SVM
Accuracy:            98.2%
Recall:              97.6%
False Negatives:     1 (missed cancers)
Estimated Cost:      $50,500

============================================================
                      DECISION SUMMARY
============================================================

  POTENTIAL SAVINGS: $99,500 per 100 patients
```

---

## Repository Structure

```
ml-metrics-that-matter/
│
├── src/
│   └── model.py                 # Complete analysis pipeline
│
├── data/
│   └── breast_cancer.csv        # Wisconsin Breast Cancer dataset
│
├── figures/                     # All visualizations
│   ├── class_distribution.png
│   ├── confusion_matrices_comparison.png
│   ├── cost_comparison.png
│   ├── cross_validation_recall.png
│   ├── learning_curves.png
│   ├── metrics_comparison.png
│   ├── precision_recall_curves.png
│   └── roc_curves.png
│
└── report/
    ├── eng/                     # English documentation
    │   ├── executive_summary.md
    │   ├── case_study.md
    │   ├── technical_report.md
    │   ├── pipeline_explained.md
    │   ├── mathematical_formulas.md
    │   └── theoretical_framework.md
    │
    └── esp/                     # Documentación en español
        └── ... (same structure)
```

---

## Documentation

### English

| Document | Audience | Description |
|----------|----------|-------------|
| [Executive Summary](report/eng/executive_summary.md) | Recruiters, Managers | Business impact, key results |
| [Case Study](report/eng/case_study.md) | Portfolio viewers | Full narrative with trade-offs |
| [Technical Report](report/eng/technical_report.md) | Engineers | Implementation details |
| [Pipeline Explained](report/eng/pipeline_explained.md) | Data Scientists | Step-by-step decisions |
| [Mathematical Formulas](report/eng/mathematical_formulas.md) | Statisticians | Formal definitions |
| [Theoretical Framework](report/eng/theoretical_framework.md) | Researchers | Complete theory |

### Español

| Documento | Audiencia | Descripción |
|-----------|-----------|-------------|
| [Resumen Ejecutivo](report/esp/executive_summary.md) | Reclutadores, Gerentes | Impacto de negocio |
| [Caso de Estudio](report/esp/case_study.md) | Portafolio | Narrativa completa |
| [Reporte Técnico](report/esp/technical_report.md) | Ingenieros | Detalles de implementación |
| [Pipeline Explicado](report/esp/pipeline_explained.md) | Científicos de Datos | Decisiones paso a paso |
| [Fórmulas Matemáticas](report/esp/mathematical_formulas.md) | Estadísticos | Definiciones formales |
| [Marco Teórico](report/esp/theoretical_framework.md) | Investigadores | Teoría completa |

---

## Methods Compared

| Approach | How It Works | Best For | Limitations |
|----------|--------------|----------|-------------|
| **Naive SVM** | Standard optimization | Baseline | Biases toward majority |
| **Class Weights** | Penalizes minority errors | Most cases | Requires tuning |
| **SMOTE** | Synthetic oversampling | Small datasets | Creates artifacts |
| **Tomek Links** | Boundary cleaning | Combined with SMOTE | Minimal standalone impact |

---

## Key Takeaways

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   "Accuracy is a proxy for what we care about.            │
│    When the proxy diverges from reality,                   │
│    optimize reality—not the proxy."                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

1. **Accuracy lies** when classes have different costs
2. **Recall saves lives** in medical screening
3. **Cost analysis** aligns ML with business objectives
4. **Confusion matrices** reveal what single metrics hide

---

## Tech Stack

- **Python 3.9+**
- **scikit-learn** - ML models and evaluation
- **imbalanced-learn** - SMOTE, Tomek Links
- **matplotlib/seaborn** - Visualizations
- **pandas/numpy** - Data processing

---

## References

- He, H., & Garcia, E. A. (2009). [Learning from Imbalanced Data](https://www.jair.org/index.php/jair/article/view/10302). *Journal of Artificial Intelligence Research*
- Chawla, N. V., et al. (2002). [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813). *Journal of Artificial Intelligence Research*
- Hastie, T., Tibshirani, R., & Friedman, J. [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

---

<p align="center">
  <strong>The lesson:</strong> Metrics encode values. Choose metrics that encode <em>your</em> values.
</p>

