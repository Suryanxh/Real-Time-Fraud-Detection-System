# Fraud Detection System

A comprehensive fraud detection system using machine learning, real-time streaming, and REST APIs to identify and prevent fraudulent transactions.

## Project Structure

```
fraud-detection-System/
├── data/
│   ├── raw/              # Raw transaction data
│   └── processed/        # Processed and cleaned data
├── models/               # Trained model artifacts
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── features.py       # Feature engineering
│   ├── train.py          # Model training
│   ├── api.py            # REST API (FastAPI)
│   └── stream.py         # Labeled transaction replay into the API (HTTP)
├── frontend/             # React monitoring dashboard (Vite)
├── db/                   # Database schema and scripts
│   └── schema.sql
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit and integration tests
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Docker services configuration
└── README.md            # This file
```

## Features

- **Machine Learning Model**: XGBoost/LightGBM for fraud classification
- **REST API**: FastAPI for real-time predictions
- **Transaction Replay**: Replays labeled test transactions into the scoring API
- **Dashboard**: React-based live monitoring and visualization
- **Database**: MySQL/PostgreSQL for transaction storage
- **Docker Support**: Containerized deployment

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Installation

1. Clone the repository
```bash
cd fraud-detection-System
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure database
```bash
mysql -u root -p < db/schema.sql
```

## Usage

### Training the Model
```bash
python src/train.py
```

### Running the API
```bash
uvicorn src.api:app --reload
```

### Running the React Dashboard

Start the API in one terminal:

```bash
python -m uvicorn src.api:app --reload --port 8000
```

In another terminal, install the frontend dependencies and start Vite:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The
dashboard polls the API for health, summary metrics, and recent decisions.

### Real-time Streaming

This is a small HTTP replay of labeled test transactions to the API (not a
Kafka consumer). Start the API in one terminal, then in a second terminal run:

```bash
python -m uvicorn src.api:app --reload --port 8000
```

```bash
python src/stream.py
```

The replay sends a balanced sample of 25 fraud-labeled and 25 legitimate
transactions at half-second intervals by default, ordered by timestamp. It
includes the highest-amount legitimate example to demonstrate a model flag.
Labels are demo evaluation metadata and are not model inputs. Keep the React
dashboard running to watch decisions arrive live.

## Docker Deployment

```bash
docker-compose up -d
```

## Model Performance

On the held-out test set (56,746 transactions; 95 fraud cases):

- ROC-AUC: **0.9763**
- Average precision: **0.8210**
- Block threshold: precision **0.9125**, recall **0.7684** (73 fraud cases caught,
  22 missed, 7 legitimate transactions flagged)
- Review threshold: precision **0.0852**, recall **0.8632** (82 fraud cases
  caught, 13 missed, 881 legitimate transactions flagged)

Training selects the block and review thresholds on validation data, then
reports performance on a separate held-out test set. Merchant, device, and IP
fields are synthetic demo data and are generated independently of fraud labels;
they are not real signals from the source dataset. Metrics are proof-of-concept
results, not evidence of production performance.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open an issue in the repository.
