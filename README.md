# Customer Churn Prediction

[![CI](https://github.com/JoelAlfred1997/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/JoelAlfred1997/customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade machine learning system for predicting customer churn in a telecommunications business, built as a portfolio project demonstrating end-to-end ML engineering best practices.

## Business Problem

Acquiring a new customer costs 5–25× more than retaining an existing one. For a telco with hundreds of thousands of subscribers, even a modest reduction in monthly churn translates to millions in recovered revenue. This project frames churn prediction not as an academic classification task but as a cost-sensitive decision problem: the cost of a false negative (letting a churner leave) is materially higher than the cost of a false positive (an unnecessary retention offer).

**Goal:** Identify customers at high churn risk before they leave, rank them by expected retention value, and provide interpretable explanations so customer-success teams can act with confidence.

## Status

Active development — data pipeline, feature engineering, and baseline models complete. Hyperparameter tuning, SHAP interpretability, and dashboard in progress.

## Results so far

Baseline models trained with stratified 5-fold CV on the train+validation split
(80 % of data). The test set is held out for final evaluation.

| Model | CV AUC | CV F1 | CV Recall | Notes |
|---|---|---|---|---|
| Logistic Regression | 0.843 ± 0.011 | 0.617 ± 0.018 | 0.778 ± 0.024 | L2, class_weight=balanced |
| Decision Tree | 0.726 ± 0.019 | 0.568 ± 0.024 | 0.706 ± 0.031 | max_depth=5, class_weight=balanced |

> **Target:** any more complex model must exceed the LR CV AUC by at least 0.01
> to justify the added complexity.

Key LR interpretability findings (top coefficients):
- **Churn drivers**: electronic check payment, month-to-month contract, fibre-optic
  internet without add-on services
- **Retention drivers**: two-year contract, autopay flag, long tenure with
  multiple service subscriptions

Run `python scripts/train_baselines.py` to reproduce and log results to MLflow.

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
