# Customer Churn Prediction

[![CI](https://github.com/JoelAlfred1997/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/JoelAlfred1997/customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade machine learning system for predicting customer churn in a telecommunications business, built as a portfolio project demonstrating end-to-end ML engineering best practices.

## Business Problem

Acquiring a new customer costs 5–25× more than retaining an existing one. For a telco with hundreds of thousands of subscribers, even a modest reduction in monthly churn translates to millions in recovered revenue. This project frames churn prediction not as an academic classification task but as a cost-sensitive decision problem: the cost of a false negative (letting a churner leave) is materially higher than the cost of a false positive (an unnecessary retention offer).

**Goal:** Identify customers at high churn risk before they leave, rank them by expected retention value, and provide interpretable explanations so customer-success teams can act with confidence.

## Status

Data pipeline, feature engineering, baseline and gradient-boosted models, hyperparameter tuning, cost-sensitive threshold optimisation, and comprehensive evaluation complete. SHAP interpretability and Streamlit dashboard in progress.

## Results

Final evaluation on the held-out **test set** (20 % of data, never seen during
training or tuning). All gradient-boosting models use early stopping on the
validation split.

### Test-set metrics — default threshold (0.50)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier ↓ |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.843 | 0.671 | 0.612 | 0.775 | 0.506 | 0.146 |
| Decision Tree | 0.728 | 0.551 | 0.567 | 0.708 | 0.472 | 0.177 |
| XGBoost | 0.858 | 0.697 | 0.641 | 0.790 | 0.536 | 0.132 |
| LightGBM | 0.855 | 0.693 | 0.636 | 0.783 | 0.532 | 0.134 |

### Cost-optimal threshold (FN cost = $500, FP cost = $50)

Lowering the decision threshold increases recall at the cost of precision —
appropriate when missing a churner (FN) costs 10× more than a wasted offer (FP).

| Model | Threshold | Recall | Precision | F1 | Expected Cost ↓ |
|---|---|---|---|---|---|
| Logistic Regression | ~0.28 | 0.848 | 0.478 | 0.612 | — |
| Decision Tree | ~0.32 | 0.768 | 0.455 | 0.572 | — |
| XGBoost | ~0.26 | 0.866 | 0.505 | 0.638 | lowest |
| LightGBM | ~0.27 | 0.860 | 0.499 | 0.633 | — |

### Key findings

- XGBoost and LightGBM improve ROC-AUC by ~1.5 pp over the LR baseline,
  confirming the complexity trade-off is worthwhile.
- At 30 % customer targeting, gradient-boosted models capture ~78 % of all
  churners — 2.5× better than random selection.
- Tree-based models output compressed probabilities; Platt or isotonic
  calibration is recommended before deploying raw scores.
- Decision-curve analysis shows all models outperform "treat all" and "treat
  none" strategies for threshold probabilities above ~0.10 (aligned with the
  1:10 FP/FN cost ratio).

See [`notebooks/08_model_evaluation.ipynb`](notebooks/08_model_evaluation.ipynb)
for full evaluation plots (ROC, PR, calibration, lift/gains, confusion matrices,
decision curves).

Key LR interpretability findings (top coefficients):
- **Churn drivers**: electronic check payment, month-to-month contract, fibre-optic
  internet without add-on services
- **Retention drivers**: two-year contract, autopay flag, long tenure with
  multiple service subscriptions

Run `python scripts/train_baselines.py` to reproduce baseline results and log to MLflow.

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features covering demographics, account information, and service subscriptions. Target: `Churn` (Yes/No).

Source: [IBM Sample Data Sets](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/03/04/updated-ibm-telco-customer-churn-dataset)

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost, LightGBM |
| Interpretability | SHAP |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Validation | pydantic |
| Dashboard | Streamlit |
| Linting / formatting | ruff, black |
| Type checking | mypy |
| Testing | pytest |
| Config | PyYAML |

## Project Layout

```
customer-churn-prediction/
├── configs/           # YAML configuration files
├── data/
│   ├── raw/           # Immutable source data (not tracked)
│   └── processed/     # Feature-engineered datasets (not tracked)
├── models/            # Serialised model artefacts (not tracked)
├── notebooks/         # Exploratory analysis (not run in CI)
├── src/
│   ├── data/          # Ingestion and validation
│   ├── features/      # Feature engineering
│   ├── models/        # Training and inference
│   ├── evaluation/    # Metrics and reporting
│   └── utils/         # Shared utilities
├── tests/             # pytest test suite
├── app.py             # Streamlit dashboard entry point
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/JoelAlfred1997/customer-churn-prediction.git
cd customer-churn-prediction
make install

# 2. Place the raw dataset
#    Download WA_Fn-UseC_-Telco-Customer-Churn.csv from IBM and drop it in:
mkdir -p data/raw
mv WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# 3. Lint, format, test
make lint
make test

# 4. Launch dashboard
make app
```

## Development

```bash
make format   # black + ruff --fix
make lint     # ruff check + mypy
make test     # pytest -v
```

## License

MIT © 2026 — see [LICENSE](LICENSE).
