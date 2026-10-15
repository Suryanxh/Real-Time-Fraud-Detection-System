# Fraud Detection System

A fraud-detection proof of concept with LightGBM and XGBoost model comparison,
a FastAPI scoring service, a React monitoring dashboard, and a labeled
transaction replay. The replay sends HTTP requests to the API; it is not a
Kafka consumer.

## Project structure

```text
data/
  raw/                  Source dataset (not committed)
  processed/            Train/test and engineered feature CSVs (not committed)
models/                 Trained models and generated evaluation artifacts
src/
  api.py                FastAPI scoring service
  config.py             Stream and API settings
  dashboard.py          Optional Streamlit dashboard
  db.py                 PostgreSQL helper used for optional metric writes
  evaluation.py         Validation-based threshold selection
  features.py           Feature engineering
  stream.py             HTTP replay of labeled test transactions
  train.py              LightGBM and XGBoost training/evaluation
frontend/               React dashboard (Vite)
notebooks/              Exploratory analysis and preprocessing
tests/                  Pytest tests
db/schema.sql           PostgreSQL schema
requirements.txt        Python dependencies
```

## Requirements and setup

- Python 3.10 or newer
- Node.js 18 or newer and npm (for the React dashboard)
- The source credit-card dataset at `data/raw/creditcard.csv`

From the repository root, create and activate a virtual environment, then
install Python dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For preprocessing, run `notebooks/01_eda_preprocessing.ipynb` to create
`data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`.
Update the notebook's project-root path for your checkout if necessary. Then
run feature engineering from the repository root:

```powershell
python -m src.features
```

This creates `train_features.csv` and `test_features.csv`, plus the merchant
risk and feature-statistics files required by the API.

## Train and compare models

Run from the repository root after feature engineering:

```powershell
python -m src.train
```

Training uses the same split and features for both classifiers. It reports
ROC-AUC, average precision, and confusion matrices on the held-out test set,
including the default 0.5 threshold and validation-selected block/review
thresholds. Generated comparison artifacts:

- `models/model_comparison_roc.html` - interactive test-set ROC curves
- `models/model_comparison_metrics.json` - test metrics and confusion matrices
- `models/xgboost_fraud_model.pkl` - separately saved XGBoost model
- `models/lgbm_fraud_model.pkl` and `models/model_metrics.json` - LightGBM
  model and metrics consumed by the API

The API continues to use LightGBM; XGBoost is trained for offline comparison.

## Run the API and React dashboard

From the repository root, start the API in one terminal:

```powershell
python -m uvicorn src.api:app --reload --port 8000
```

In a second terminal, start the React dashboard:

```powershell
cd frontend
npm install
npm run dev
```

Open the local Vite URL printed in the terminal, usually
`http://localhost:5173`. The dashboard polls the API at `http://localhost:8000`.
Keep both processes running.

The optional Streamlit dashboard can also be run from the repository root,
with the API running:

```powershell
python -m streamlit run src/dashboard.py
```

It normally opens at `http://localhost:8501`.

## Replay sample transactions

Start the API first, then run this from the repository root in another
terminal:

```powershell
python -m src.stream
```

By default, the replay sends up to 50 transactions (a balanced fraud/legitimate
demo sample) at 0.5-second intervals. Settings can be changed with
`STREAM_LIMIT_ROWS`, `STREAM_EVENT_DELAY`, and `FRAUD_API_BASE_URL`. The replay
sends the historical label only as demo evaluation metadata; it is not used as
a model input. Keep a dashboard open to view decisions as they arrive.

## Database and Docker status

The local API keeps decisions in memory; they are cleared when the API process
restarts. Training also attempts to write metrics to PostgreSQL, but a database
is not required for local training, scoring, or dashboard use. The schema in
`db/schema.sql` is PostgreSQL.

Docker Compose is not currently a verified run path: `docker-compose.yml`
references a `Dockerfile` that is not present in this repository, configures
MySQL while the checked-in schema is PostgreSQL, and its streaming service
does not match the documented HTTP replay. Use the local commands above
instead.

## Tests

Run the project tests from the repository root:

```powershell
python -m pytest tests -q
```

## Evaluation snapshot

The latest recorded held-out test comparison (56,746 transactions, 95 fraud
cases) was:

| Model | ROC-AUC | Average precision | Confusion matrix at 0.5 (`[[TN, FP], [FN, TP]]`) |
|---|---:|---:|---|
| LightGBM | 0.9763 | 0.8210 | `[[56643, 8], [22, 73]]` |
| XGBoost | 0.9693 | 0.8213 | `[[56642, 9], [22, 73]]` |

These are proof-of-concept results, not evidence of production performance.
Merchant, device, and IP fields are synthetic demo data generated independently
of fraud labels, not real signals from the source dataset.
