# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A portfolio case study demonstrating cost-aware model selection for imbalanced medical classification. The project shows why accuracy is dangerous for cancer diagnosis and how to make decisions that align with real-world consequences.

## Structure

```
├── src/
│   └── model.py              # Main classifier with cost analysis
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
│   └── esp/                  # Spanish documentation
│       ├── case_study.md
│       ├── executive_summary.md
│       ├── technical_report.md
│       ├── pipeline_explained.md
│       ├── mathematical_formulas.md
│       └── theoretical_framework.md
├── figures/                  # Generated visualizations
├── README.md                 # Project overview
└── CLAUDE.md                 # This file
```

## Running the Code

```bash
python src/model.py
```

Dependencies: numpy, pandas, scikit-learn, imbalanced-learn

## Documentation by Audience

| Document | Audience | Focus |
|----------|----------|-------|
| executive_summary | Recruiters, Managers | Business impact, skills demonstrated |
| case_study | General / Portfolio | Full narrative with trade-offs |
| technical_report | Tech Leads, Engineers | Implementation details, benchmarks |
| pipeline_explained | Data Scientists, ML Engineers | Step-by-step pipeline decisions |
| mathematical_formulas | Statisticians, Quants | Formal definitions, proofs |
| theoretical_framework | Academia, Researchers | Complete theoretical foundations |

## Key Concepts

- **Accuracy trap**: Standard metrics hide failures on minority classes
- **Cost-weighted evaluation**: False negatives and false positives have asymmetric costs
- **Resampling strategies**: SMOTE, Tomek Links, and their failure modes

## Editing Guidelines

- Keep the focus on decision-making, not algorithm mechanics
- All code changes should preserve the cost analysis framework
- Documentation exists in English and Spanish; update both when making changes
- Maintain clear separation between modeling code and evaluation logic
