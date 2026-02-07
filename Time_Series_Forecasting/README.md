# Time Series Forecasting & Anomaly Detection

End-to-end MLOps project for e-commerce sales forecasting with automated drift detection, model retraining, and production serving.

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **XGBoost (tuned)** | **307** | **754** | **8.2%** |
| XGBoost (default) | 323 | 1277 | 7.7% |
| Moving Avg (7d) | 1453 | 1847 | 43.5% |
| Naive (yesterday) | 1826 | 2489 | 51.9% |
| SARIMA | 2458 | 3061 | 55.5% |
| Prophet | 2671 | 3263 | 61.4% |

Champion model: **XGBoost** tuned with Optuna (20 trials), achieving **79% MAE improvement** over the best baseline (Moving Avg).

## Architecture

```
XML Data  -->  Ingestion  -->  Feature Engineering  -->  Model Training  -->  Serving API
                  |                    |                       |                  |
             ingest.py         build_features.py      train_model.py         app.py
                                                            |                  |
                                                      MLflow Registry    Prometheus
                                                            |              Metrics
                                                      Drift Detection  -->  Grafana
                                                            |
                                                    Auto-Retraining
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | Pandas, NumPy, lxml |
| Modeling | XGBoost, Prophet, SARIMA (statsmodels), Isolation Forest |
| Experiment tracking | MLflow |
| Hyperparameter tuning | Optuna |
| Serving | FastAPI, Pydantic, uvicorn |
| Monitoring | Prometheus, Grafana, Evidently (drift detection) |
| Security | API key auth, slowapi (rate limiting), CORS |
| Infrastructure | Docker, Docker Compose |
| CI/CD | GitHub Actions (lint, test, build, push) |
| Code quality | ruff, pytest (31 tests), pytest-cov |
| Visualization | Plotly, Streamlit |

## Project Structure

```
├── src/
│   ├── config.py                 # Centralized settings (pydantic-settings)
│   ├── data/
│   │   └── ingest.py             # XML parsing, daily aggregation
│   ├── features/
│   │   └── build_features.py     # Calendar, lag, rolling features
│   ├── models/
│   │   ├── train_model.py        # ChampionChallenger multi-model training
│   │   ├── detect_anomalies.py   # Isolation Forest anomaly detection
│   │   └── registry.py           # MLflow Model Registry wrapper
│   ├── serving/
│   │   ├── app.py                # FastAPI v2.0 with Prometheus middleware
│   │   └── auth.py               # API key authentication
│   ├── monitoring/
│   │   ├── drift_detector.py     # Evidently + PSI drift detection
│   │   ├── data_quality.py       # Statistical data quality checks
│   │   ├── retraining.py         # Automated retraining pipeline
│   │   └── scheduler.py          # APScheduler (24h evaluation cycle)
│   └── visualization/
│       └── plot_results.py       # Plotly dashboard generation
├── tests/                        # 31 tests (pytest)
├── notebooks/                    # EDA, baseline, advanced modeling
├── monitoring/                   # Prometheus + Grafana configs
├── .github/workflows/            # CI (ruff+pytest+pip-audit) + CD (Docker)
├── docker-compose.yml            # 5 services: api, mlflow, scheduler, prometheus, grafana
├── Dockerfile                    # python:3.10-slim, non-root user
├── main.py                       # Full pipeline orchestrator
├── streamlit_app.py              # Interactive demo dashboard
└── requirements.txt
```

## Setup

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the Pipeline

```bash
# Full pipeline: ingest -> features -> train -> anomalies -> visualize
python main.py
```

### Start the API

```bash
uvicorn src.serving.app:app --reload
```

### Run Tests

```bash
pytest tests/ -v
ruff check src/ tests/
```

### Docker

```bash
cp .env.example .env    # configure secrets
docker-compose up -d    # starts api, mlflow, scheduler, prometheus, grafana
```

## API

Base URL: `http://localhost:8000`

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | API info |
| `/health` | GET | No | Health check + model status |
| `/metrics` | GET | No | Prometheus metrics |
| `/forecast` | POST | `X-API-Key` | Generate sales forecast |
| `/model-info` | GET | `X-API-Key` | Model metadata |
| `/reload-model` | POST | `X-API-Key` | Hot-reload model |

Example:
```bash
curl -X POST http://localhost:8000/forecast \
  -H "X-API-Key: change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"days_ahead": 7}'
```

Rate limits: `/forecast` 60/min, `/reload-model` 5/min.

## Environment Variables

All settings use the `TSF_` prefix (via pydantic-settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `TSF_API_KEY_SECRET` | `change-me-in-production` | API key for protected endpoints |
| `TSF_API_KEY_ENABLED` | `true` | Enable/disable auth |
| `TSF_RATE_LIMIT_PER_MINUTE` | `60` | Rate limit for forecast endpoint |
| `TSF_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `TSF_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `TSF_RETRAINING_SCHEDULE_HOURS` | `168` | Retraining evaluation interval |

## Data

The project expects XML files with e-commerce order data in `Data/` directory (not tracked in git). Each XML contains orders with `date_add` and `order_amount_brutto` fields. The pipeline aggregates them into daily sales time series.

To reproduce results, place your XML data files in `Data/` and run `python main.py`.
