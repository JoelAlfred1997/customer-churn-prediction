# Dataset: IBM Telco Customer Churn

## Overview

The IBM Telco Customer Churn dataset contains information about a fictional telecom company's customers and whether each customer left ("churned") within the last month.

- **Rows:** 7,043 customers
- **Columns:** 21 (20 features + 1 target)
- **Target:** `Churn` — binary `Yes` / `No`
- **Class imbalance:** ~26.5 % positive (churned)

## Source and License

| Attribute | Detail |
|-----------|--------|
| Original publisher | IBM Watson Analytics |
| Kaggle page | <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |
| IBM GitHub | <https://github.com/IBM/telco-customer-churn-on-icp4d> |
| License | Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0) |

The raw file is not committed to this repository.  See **Download** below.

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | string | Unique customer identifier |
| `gender` | categorical | `Male` / `Female` |
| `SeniorCitizen` | int (0/1) | Whether the customer is a senior citizen |
| `Partner` | Yes/No | Has a partner |
| `Dependents` | Yes/No | Has dependents |
| `tenure` | int | Months with the company |
| `PhoneService` | Yes/No | Has phone service |
| `MultipleLines` | categorical | `Yes` / `No` / `No phone service` |
| `InternetService` | categorical | `DSL` / `Fiber optic` / `No` |
| `OnlineSecurity` | categorical | `Yes` / `No` / `No internet service` |
| `OnlineBackup` | categorical | `Yes` / `No` / `No internet service` |
| `DeviceProtection` | categorical | `Yes` / `No` / `No internet service` |
| `TechSupport` | categorical | `Yes` / `No` / `No internet service` |
| `StreamingTV` | categorical | `Yes` / `No` / `No internet service` |
| `StreamingMovies` | categorical | `Yes` / `No` / `No internet service` |
| `Contract` | categorical | `Month-to-month` / `One year` / `Two year` |
| `PaperlessBilling` | Yes/No | Enrolled in paperless billing |
| `PaymentMethod` | categorical | `Electronic check` / `Mailed check` / `Bank transfer (automatic)` / `Credit card (automatic)` |
| `MonthlyCharges` | float | Current monthly charge (USD) |
| `TotalCharges` | float* | Total charges to date (USD) |
| `Churn` | Yes/No | **Target** — left in the last month |

\* `TotalCharges` is stored as a string in the raw file; ~11 rows with `tenure = 0`
contain a blank string instead of `0.0`.  This is handled in `src/data/validate.py`
(warning) and corrected in the preprocessing step.

## Download

### Option A — automatic (recommended)

```python
from src.data import download_telco_data

download_telco_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

This tries two IBM GitHub mirrors in sequence.

### Option B — Kaggle CLI

Requires a `~/.kaggle/kaggle.json` credentials file.

```bash
kaggle datasets download -d blastchar/telco-customer-churn --unzip \
  -p data/raw/
```

### Option C — direct browser download

1. Visit <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place the file in `data/raw/`

## Directory Layout

```
data/
├── raw/          # original, unmodified source file (gitignored)
└── processed/    # feature-engineered parquet files (gitignored)
```
