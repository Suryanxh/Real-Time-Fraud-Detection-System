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
│   ├── stream.py         # Real-time streaming (Kafka)
│   └── dashboard.py      # Monitoring dashboard (Streamlit)
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
- **Streaming Processing**: Kafka integration for real-time transaction processing
- **Dashboard**: Streamlit-based monitoring and visualization
- **Database**: MySQL/PostgreSQL for transaction storage
- **Docker Support**: Containerized deployment

## Setup

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- MySQL or PostgreSQL
- Kafka (for streaming features)

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

### Running the Dashboard
```bash
streamlit run src/dashboard.py
```

### Real-time Streaming
```bash
python src/stream.py
```

## Docker Deployment

```bash
docker-compose up -d
```

## Model Performance

- Accuracy: [To be added]
- Precision: [To be added]
- Recall: [To be added]
- AUC-ROC: [To be added]

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open an issue in the repository.
