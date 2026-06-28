
# CodigoFinal.py — Hybrid Bayesian Networks for Cardiovascular Diagnosis

**Author:** María Cañas Francia | UCM Mathematics Engineering BSc Thesis, 2025/2026

## Description
Full implementation of a hybrid CLG Bayesian network pipeline applied to the UCI Cleveland Heart Disease Dataset (303 patients, 14 variables), including structure learning, parameter estimation, exact/approximate inference, and evaluation against classical ML models.

## Requirements
```bash
pip install pandas numpy matplotlib seaborn networkx scipy scikit-learn xgboost imbalanced-learn pgmpy ucimlrepo shap
```

## Structure
| Block | Content |
|-------|---------|
| 1–3 | Data loading, EDA, discretization, train/test split |
| 4 | Structure learning: expert network + PC, HC-BIC, HC-BDeu, MMHC |
| 5 | Parameter estimation: CPTs (ESS=5) + CLG Gaussian parameters (OLS) |
| 6 | Exact inference: Variable Elimination & Belief Propagation |
| 7 | Approximate inference: Gibbs Sampling & Likelihood Weighting |
| 8–10 | Evaluation, SHAP, Bayesian inversion, robustness analysis |

## Usage
```bash
python CodigoFinal.py
```
Dataset is downloaded automatically from UCI ML Repository.
