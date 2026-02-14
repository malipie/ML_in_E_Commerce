"""NLP Feature Extraction & Product Categorization — main pipeline orchestrator.

Executes the full NLP pipeline:
    1. ETL:  XML files → item-level DataFrame → Parquet
    2. Text Preprocessing:  normalize, tokenize, stopwords
    3. Feature Extraction:  color, material, size, product_type, brand, season
    4. NLP Features:  TF-IDF, text statistics
    5. ML Categorization:  SVM / RF / XGBoost → champion model
    6. Export:  enriched DataFrame + serialized model

Usage:
    python main.py                   # use default config.yaml
    python main.py --config my.yaml  # custom config path
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.categorization import (
    evaluate_models,
    predict_unlabeled,
    prepare_labeled_data,
    save_model,
    select_champion,
)
from src.etl import run_etl, save_parquet
from src.feature_extraction import FeatureExtractor
from src.nlp_features import compute_text_statistics, compute_tfidf_features
from src.text_preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to config file.

    Returns:
        Parsed config dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Config loaded from %s", path)
    return config


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: dict[str, Any]) -> None:
    """Execute the full NLP pipeline.

    Args:
        config: Parsed YAML config dict.
    """
    data_cfg = config["data"]
    nlp_cfg = config.get("nlp", {})
    extraction_cfg = config["extraction"]
    classification_cfg = config.get("classification", {})
    model_cfg = config.get("models", {})

    # ── 1. ETL ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/6 — ETL")
    logger.info("=" * 60)

    df = run_etl(
        raw_dir=data_cfg["raw_dir"],
        output_path=data_cfg["items_parquet"],
        pattern=data_cfg.get("xml_glob", "*.xml"),
    )
    logger.info("ETL complete: %d items loaded.", len(df))

    # ── 2. Text Preprocessing ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/6 — Text Preprocessing")
    logger.info("=" * 60)

    df = preprocess_dataframe(
        df,
        use_lemma=nlp_cfg.get("use_lemma", False),
        min_length=nlp_cfg.get("min_token_length", 2),
    )
    logger.info("Preprocessing complete: %d items.", len(df))

    # ── 3. Rule-Based Feature Extraction ───────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/6 — Rule-Based Feature Extraction")
    logger.info("=" * 60)

    extractor = FeatureExtractor.from_config(extraction_cfg)
    df = extractor.extract_dataframe(df)

    for col in ["color", "material", "size", "product_type", "brand", "season"]:
        coverage = df[col].notna().mean() * 100
        logger.info("  %s: %.1f%% coverage", col, coverage)

    # ── 4. NLP Features ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4/6 — NLP Feature Engineering")
    logger.info("=" * 60)

    tfidf_cfg = classification_cfg.get("tfidf", {})
    ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
    tfidf_matrix, vectorizer = compute_tfidf_features(
        df["name_clean"],
        max_features=tfidf_cfg.get("max_features", 500),
        ngram_range=ngram_range,
        min_df=tfidf_cfg.get("min_df", 2),
    )
    df = compute_text_statistics(df)
    logger.info(
        "TF-IDF: %d features. Text stats added.",
        tfidf_matrix.shape[1],
    )

    # ── 5. ML Categorization ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5/6 — ML Product Categorization")
    logger.info("=" * 60)

    df_labeled, df_unlabeled = prepare_labeled_data(
        df,
        min_class_size=classification_cfg.get("min_class_size", 10),
    )

    if len(df_labeled) > 0:
        results = evaluate_models(
            df_labeled["name_clean"],
            df_labeled["product_type"],
            config=classification_cfg,
        )
        champion = select_champion(results)

        logger.info("Champion: %s (CV F1=%.3f)", champion.model_name,
                     champion.cv_scores.mean())

        for r in results:
            logger.info(
                "  %s: CV F1=%.3f (±%.3f), Acc=%.3f",
                r.model_name, r.cv_scores.mean(), r.cv_scores.std(), r.accuracy,
            )

        # Predict unlabeled
        if not df_unlabeled.empty:
            df_unlabeled = predict_unlabeled(
                champion.model, df_unlabeled, name_col="name_clean"
            )
            df = pd.concat([df_labeled, df_unlabeled], ignore_index=True)
            logger.info("Predicted %d unlabeled items.", len(df_unlabeled))
        else:
            df = df_labeled
    else:
        logger.warning("No labeled data found. Skipping ML categorization.")
        champion = None

    # ── 6. Export ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6/6 — Export")
    logger.info("=" * 60)

    save_parquet(df, data_cfg["categorized_parquet"])

    if champion is not None:
        model_dir = Path(model_cfg.get("output_dir", "models"))
        save_model(champion.model, model_dir / "champion_pipeline.pkl")

    # ── Summary ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total items:        %d", len(df))
    logger.info("Product types:      %d unique",
                df["product_type"].nunique() if "product_type" in df.columns else 0)
    if champion is not None:
        logger.info("Champion model:     %s (CV F1=%.3f)",
                     champion.model_name, champion.cv_scores.mean())
    logger.info("Output:             %s", data_cfg["categorized_parquet"])
    logger.info("Model:              %s/champion_pipeline.pkl",
                model_cfg.get("output_dir", "models"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NLP Feature Extraction & Product Categorization — pipeline runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
