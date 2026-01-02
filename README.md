# The Accuracy Trap: When Better Metrics Mean Worse Outcomes

A case study demonstrating why accuracy is the wrong metric for medical diagnosis, and how to make model selection decisions that align with real-world costs.

## The Problem

A hospital needs to classify breast tumors as malignant or benign. Two models are proposed:

| Model | Accuracy | Missed Cancers |
|-------|----------|----------------|
| Model A | 97% | 3 |
| Model B | 94% | 1 |

**Which model should the hospital deploy?**

If you chose Model A, you would discharge more cancer patients without treatment. This repository explains why, and what to do instead.

## Key Findings

- A trivial classifier achieves **63% accuracy** by predicting "benign" for all patients
- Standard SVM optimization biases toward the majority class, missing malignant tumors
- SMOTE resampling **reduces missed cancers by 67%** at a 3% accuracy cost
- Cost-weighted evaluation shows the "worse" model **saves $98,500 per 100 patients**

## Repository Structure

```
├── src/
│   └── model.py              # Classifier implementation with cost analysis
├── data/
│   └── breast_cancer.csv     # Wisconsin Diagnostic Breast Cancer dataset
├── report/
│   ├── eng/                  # English documentation
│   │   ├── case_study.md
│   │   ├── executive_summary.md
│   │   ├── technical_report.md
│   │   ├── pipeline_explained.md
│   │   ├── mathematical_formulas.md
│   │   └── theoretical_framework.md
│   └── esp/                  # Documentación en español
│       ├── case_study.md
│       ├── executive_summary.md
│       ├── technical_report.md
│       ├── pipeline_explained.md
│       ├── mathematical_formulas.md
│       └── theoretical_framework.md
├── figures/                  # Generated visualizations
└── README.md
```

## Documentation

### By Audience

| Document | Audience | Content |
|----------|----------|---------|
| [executive_summary](report/eng/executive_summary.md) | Recruiters, Managers | Business impact, key results |
| [case_study](report/eng/case_study.md) | General / Portfolio | Full narrative with trade-offs |
| [technical_report](report/eng/technical_report.md) | Tech Leads, Engineers | Implementation, benchmarks |
| [pipeline_explained](report/eng/pipeline_explained.md) | Data Scientists | Step-by-step ML pipeline |
| [mathematical_formulas](report/eng/mathematical_formulas.md) | Statisticians, Quants | Formal definitions |
| [theoretical_framework](report/eng/theoretical_framework.md) | Academia, Researchers | Complete theoretical foundations |

### En Español

| Documento | Audiencia | Contenido |
|-----------|-----------|-----------|
| [executive_summary](report/esp/executive_summary.md) | Reclutadores, Gerentes | Impacto de negocio |
| [case_study](report/esp/case_study.md) | General / Portafolio | Narrativa completa |
| [technical_report](report/esp/technical_report.md) | Líderes Técnicos | Implementación, benchmarks |
| [pipeline_explained](report/esp/pipeline_explained.md) | Científicos de Datos | Pipeline paso a paso |
| [mathematical_formulas](report/esp/mathematical_formulas.md) | Estadísticos | Definiciones formales |
| [theoretical_framework](report/esp/theoretical_framework.md) | Academia, Investigadores | Marco teórico completo |

## Quick Start

```bash
python src/model.py
```

Output:
```
BREAST CANCER CLASSIFICATION: THE ACCURACY TRAP
============================================================

Dataset: 357 benign, 212 malignant
Imbalance ratio: 1.68:1

BASELINE: Predict 'benign' for all patients
------------------------------------------------------------
Accuracy: 62.7%
Recall (malignant): 0.0%
>>> MISSED 42 OF 42 CANCERS

NAIVE MODEL: SVM without imbalance handling
------------------------------------------------------------
Accuracy: 97.4%
Recall: 95.3%
False Negatives (missed cancers): 2
Estimated cost: $101,000

BALANCED MODEL: SVM with SMOTE oversampling
------------------------------------------------------------
Accuracy: 94.7%
Recall: 97.7%
False Negatives (missed cancers): 1
Estimated cost: $52,500

>>> CRITICAL INSIGHT: Lower accuracy, but better outcomes.
>>> Accuracy optimization would have chosen the WORSE model.
```

## Technical Approach

### What This Project Demonstrates

1. **The accuracy trap**: Why optimizing accuracy on imbalanced data produces harmful models
2. **Cost-aware evaluation**: Translating confusion matrices into business/medical costs
3. **Resampling trade-offs**: When SMOTE helps, when it creates artifacts, when Tomek Links fails
4. **Metric selection**: Choosing recall, F2, or custom cost functions based on domain requirements

### Methods Compared

| Approach | Strengths | Failure Modes |
|----------|-----------|---------------|
| Naive SVM | Simple, interpretable | Biases toward majority class |
| SMOTE | Increases minority representation | Creates synthetic edge cases |
| Tomek Links | Cleans decision boundary | Minimal effect on severe imbalance |
| Class weighting | No data modification | Requires careful tuning |

## The Decision Framework

Before training on imbalanced data:

1. **Quantify the cost ratio** of false negatives vs false positives
2. **Set minimum recall thresholds** based on domain requirements
3. **Evaluate on cost-weighted metrics**, not raw accuracy
4. **Report confusion matrices** to stakeholders, not single scores

## Dependencies

```
numpy
pandas
scikit-learn
imbalanced-learn
```

## Data

Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 features). Features are computed from digitized images of fine needle aspirates of breast masses.

## References

- [Learning from Imbalanced Data](https://www.jair.org/index.php/jair/article/view/10302) - He & Garcia, 2009
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) - Chawla et al., 2002
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) - Hastie, Tibshirani & Friedman
