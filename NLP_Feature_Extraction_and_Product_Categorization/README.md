# NLP Feature Extraction & Product Categorization

Rule-based + ML hybrid pipeline for extracting structured attributes from Polish e-commerce product names and automated product categorization.

## Key Results

**Feature Extraction Coverage (9,879 items):**

| Feature | Coverage |
|---------|----------|
| Product type | 84.2% |
| Brand | 78.8% |
| Size | 73.4% |
| Color | 55.4% |
| Material | 46.0% |
| Season | 8.7% |

**ML Categorization (20 product types):**

| Model | CV F1-macro | Accuracy |
|-------|-------------|----------|
| **LinearSVC** | **1.000** | **1.000** |
| Random Forest | 0.999 | 1.000 |
| XGBoost | 0.997 | 1.000 |

## Architecture

```
XML (9 files) → ETL → Text Preprocessing → Rule-Based Extraction → NLP Features → ML Categorization → Parquet + Model
                 │          │                      │                      │               │
            item-level   normalize           color, material,        TF-IDF,        SVM / RF /
            DataFrame    tokenize            size, type,             text stats     XGBoost
                         stopwords           brand, season
```

## Pipeline

```
STEP 1/6 — ETL:                 XML → item-level DataFrame (9,879 items)
STEP 2/6 — Text Preprocessing:  normalize, tokenize, stopword removal
STEP 3/6 — Feature Extraction:  rule-based color/material/size/type/brand/season
STEP 4/6 — NLP Features:        TF-IDF (500 features) + text statistics
STEP 5/6 — ML Categorization:   3-model comparison, champion selection, unlabeled prediction
STEP 6/6 — Export:               enriched Parquet + serialized champion model
```

## Tech Stack

| Category | Tools |
|----------|-------|
| NLP | spaCy (pl_core_news_sm), scikit-learn TF-IDF |
| ML | LinearSVC, Random Forest, XGBoost |
| Data | Pandas, lxml, PyArrow |
| Visualization | Plotly, Matplotlib, WordCloud |
| Testing | pytest (176 tests, 94% coverage) |
| CI/CD | GitHub Actions (ruff + pytest + pip-audit) |

## Project Structure

```
NLP_Feature_Extraction_and_Product_Categorization/
├── src/
│   ├── etl.py                    # XML parsing → item-level DataFrame
│   ├── text_preprocessing.py     # Polish text normalization & tokenization
│   ├── feature_extraction.py     # Rule-based attribute extraction
│   ├── nlp_features.py           # TF-IDF & text statistics
│   └── categorization.py         # ML product type classification
├── tests/                         # 176 pytest tests (94% coverage)
├── notebooks/
│   ├── 01_eda.ipynb              # Text analysis & extraction coverage
│   └── 02_nlp_experiments.ipynb  # Model comparison & error analysis
├── data/processed/                # Parquet outputs
├── models/                        # Serialized champion pipeline
├── main.py                        # Pipeline orchestrator
├── config.yaml                    # Extraction dictionaries & model config
├── requirements.txt
├── Makefile
└── README.md
```

## Setup

```bash
make install    # creates .venv, installs deps + spaCy model
```

## Usage

```bash
make run        # run full pipeline
make test       # run tests with coverage
make lint       # lint with ruff
```

## Data

Uses shared e-commerce order data (XML) from the `Data/` directory. Product domain: fashion/footwear from Allegro marketplace (Polish). Product names contain color, material, size, brand, and product type information in structured and semi-structured Polish text.
