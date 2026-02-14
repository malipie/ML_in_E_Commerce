# Customer Segmentation Engine

Transaction-level clustering of e-commerce orders with automated segment labeling and an interactive Streamlit dashboard.

## Results

| Feature Set | Features | Best k | Silhouette |
|-------------|----------|--------|------------|
| Numerical only | 4 | auto | highest |
| Numerical + Cyclical | 8 | auto | high |
| Reduced cardinality | ~25 | auto | moderate |
| Baseline (full) | 126 | auto | lowest |

Key finding: fewer, well-chosen features (monetary + basket) produce cleaner segments than the full 126-dimensional one-hot feature space.

**7 auto-labeled segments:** Premium Single-Item, High-Value Multi-Item, High Spenders, Budget Multi-Item, Small Quick Purchases, Budget Shoppers, Standard Orders.

## Architecture

```
XML Data  -->  ETL  -->  Feature Engineering  -->  Clustering  -->  Labeling  -->  Dashboard
                |               |                      |              |               |
            etl.py    feature_engineering.py    clustering.py    labeling.py    streamlit_app.py
                                |                      |
                        log1p, sin/cos,         K-Means (elbow),
                        one-hot, scale          DBSCAN (noise)
                                                       |
                                                 PCA / t-SNE
                                               (visualization)
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | Pandas, NumPy, lxml, PyArrow |
| ML | scikit-learn (K-Means, DBSCAN, PCA, t-SNE), SciPy |
| Visualization | Plotly (7 chart types), Kaleido |
| Dashboard | Streamlit (real-time parameter tuning) |
| CI/CD | GitHub Actions (ruff + pytest + pip-audit) |
| Code quality | ruff, pytest (151 tests, 79% coverage) |
| Config | YAML-driven pipeline |

## Project Structure

```
├── src/
│   ├── etl.py                   # XML parsing, data cleaning, parquet export
│   ├── feature_engineering.py   # log1p, sin/cos, one-hot, StandardScaler pipeline
│   ├── clustering.py            # K-Means (elbow/silhouette), DBSCAN, metrics
│   ├── dimensionality.py        # PCA + t-SNE with variance diagnostics
│   ├── labeling.py              # Rank-based segment labeling (7 types)
│   └── visualization.py         # Plotly charts (elbow, 3D PCA, radar, box, heatmap)
├── tests/                       # 151 tests (pytest), 79% coverage
├── notebooks/
│   ├── 01_eda.ipynb             # Distributions, outliers, correlations, PCA variance
│   └── 02_modeling_experiments.ipynb  # 4 feature set experiments, DBSCAN tuning, t-SNE
├── .github/workflows/ci.yml    # CI: lint, test, security audit
├── main.py                      # Full 6-step pipeline orchestrator
├── streamlit_app.py             # Interactive dashboard
├── config.yaml                  # Pipeline configuration
├── Makefile                     # install, test, lint, run, dashboard
└── requirements.txt
```

## Setup

```bash
make install
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the Pipeline

```bash
make run
# or: python main.py
```

### Launch Dashboard

```bash
make dashboard
# or: streamlit run streamlit_app.py
```

### Run Tests

```bash
make test
make lint
```

## Data

The project expects XML files with e-commerce order data in `../Data/` (not tracked in git). Each XML contains orders with fields like `order_id`, `date_add`, `order_amount_brutto`, delivery and payment info, and item details. The ETL pipeline parses XML, aggregates items per order, and exports to Parquet.

To reproduce: place XML data in `../Data/` and run `make run`.
