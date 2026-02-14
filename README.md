# ML in E-Commerce

A portfolio of machine learning projects applied to real-world e-commerce data. Each project tackles a different ML domain — from time series forecasting through customer segmentation to NLP-based product categorization — forming a comprehensive toolkit for data-driven e-commerce.

## Projects

### 1. [Time Series Forecasting & Anomaly Detection](Time_Series_Forecasting/) — Completed

End-to-end MLOps pipeline for e-commerce sales forecasting with automated drift detection, model retraining, and production serving.

**Pipeline:** XML ingestion → feature engineering → model training → API serving → monitoring

**Key results:**

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **XGBoost (Optuna-tuned)** | **307** | **754** | **8.2%** |
| Moving Avg (7d) | 1453 | 1847 | 43.5% |
| SARIMA | 2458 | 3061 | 55.5% |
| Prophet | 2671 | 3263 | 61.4% |

Champion model achieves **79% MAE improvement** over the best baseline.

**Highlights:**
- XGBoost with Optuna hyperparameter tuning (20 trials)
- FastAPI serving with API key auth, rate limiting, Prometheus metrics
- Automated drift detection (Evidently, PSI) with scheduled retraining
- Isolation Forest anomaly detection on sales data
- Docker Compose stack: API, MLflow, Scheduler, Prometheus, Grafana
- CI/CD via GitHub Actions (lint, test, build, push)
- 31 pytest tests, ruff linting, pip-audit security checks

**Tech stack:** Python, XGBoost, Prophet, statsmodels, FastAPI, MLflow, Optuna, Evidently, Docker, Prometheus, Grafana, GitHub Actions

### 2. [Customer Segmentation Engine](Clustering/) — Completed

Transaction-level clustering pipeline with automated segment labeling and interactive Streamlit dashboard.

**Pipeline:** XML parsing → feature engineering → K-Means / DBSCAN → segment labeling → dashboard

**Key results:**

| Feature Set | Features | Best k | Silhouette |
|-------------|----------|--------|------------|
| Numerical only | 4 | auto | highest |
| Numerical + Cyclical | 8 | auto | high |
| Reduced cardinality | ~25 | auto | moderate |
| Baseline (full one-hot) | 126 | auto | lowest |

Fewer, well-chosen features produce cleaner segments than the full 126-dim one-hot space. Pipeline auto-labels **7 business segments** (Premium, High-Value, Budget, Standard, etc.).

**Highlights:**
- K-Means (elbow + silhouette analysis) and DBSCAN (noise detection)
- Rank-based auto-labeling of 7 business segment types
- Streamlit dashboard with real-time parameter tuning and 3D PCA visualization
- FeaturePipeline: log1p, sin/cos cyclical, one-hot, StandardScaler
- EDA + modeling experiment notebooks with executed outputs
- CI/CD via GitHub Actions (ruff + pytest + pip-audit)
- 151 pytest tests, 79% coverage

**Tech stack:** Python, scikit-learn, Pandas, NumPy, Plotly, Streamlit, lxml, PyArrow, GitHub Actions

### 3. [NLP Feature Extraction & Product Categorization](NLP_Feature_Extraction_and_Product_Categorization/) — Completed

Rule-based + ML hybrid pipeline for extracting structured attributes from Polish e-commerce product names and automated product categorization.

**Pipeline:** XML parsing → text preprocessing → rule-based extraction → TF-IDF features → ML categorization → enriched Parquet + model

**Key results:**

| Feature | Coverage |
|---------|----------|
| Product type | 84.2% |
| Brand | 78.8% |
| Size | 73.4% |
| Color | 55.4% |
| Material | 46.0% |
| Season | 8.7% |

| Model | CV F1-macro | Accuracy |
|-------|-------------|----------|
| **LinearSVC** | **1.000** | **1.000** |
| Random Forest | 0.999 | 1.000 |
| XGBoost | 0.997 | 1.000 |

**Highlights:**
- Rule-based extraction of 6 product attributes from Polish text (color, material, size, type, brand, season)
- TF-IDF + classifier pipelines (LinearSVC, Random Forest, XGBoost) for product categorization
- 20 product type categories classified with perfect CV F1
- Champion model predicts categories for unlabeled items
- EDA + modeling experiment notebooks with Plotly visualizations
- CI/CD via GitHub Actions (ruff + pytest + pip-audit)
- 176 pytest tests, 94% coverage

**Tech stack:** Python, spaCy, scikit-learn, XGBoost, Pandas, lxml, Plotly, Matplotlib, WordCloud, GitHub Actions

## Repository Structure

```
ML_in_E_Commerce/
├── Time_Series_Forecasting/   # Project 1 (completed)
├── Clustering/                # Project 2 (completed)
├── NLP_Feature_Extraction_and_Product_Categorization/  # Project 3 (completed)
├── Data/                      # Shared e-commerce datasets (not tracked)
└── README.md
```

## Data

All projects use real e-commerce order data (XML format) stored in the `Data/` directory. Data files are not tracked in git due to size and privacy. See individual project READMEs for data format details and reproduction instructions.
