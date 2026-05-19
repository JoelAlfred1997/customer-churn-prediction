# Customer Churn Prediction

[![CI](https://github.com/JoelAlfred1997/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/JoelAlfred1997/customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade machine learning system for predicting customer churn in a telecommunications business, built as a portfolio project demonstrating end-to-end ML engineering best practices.

## Business Problem

Acquiring a new customer costs 5–25× more than retaining an existing one. For a telco with hundreds of thousands of subscribers, even a modest reduction in monthly churn translates to millions in recovered revenue. This project frames churn prediction not as an academic classification task but as a cost-sensitive decision problem: the cost of a false negative (letting a churner leave) is materially higher than the cost of a false positive (an unnecessary retention offer).

**Goal:** Identify customers at high churn risk before they leave, rank them by expected retention value, and provide interpretable explanations so customer-success teams can act with confidence.

## Status

Active development — core data pipeline and feature engineering complete. Model training, interpretability, and dashboard in progress.

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
