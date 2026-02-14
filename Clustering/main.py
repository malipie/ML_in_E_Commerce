"""Customer Segmentation Engine — main pipeline orchestrator.

Executes the full segmentation pipeline:
    1. ETL:  XML files → clean DataFrame → Parquet
    2. Feature Engineering:  log1p, sin/cos, one-hot, StandardScaler
    3. Clustering:  K-Means (elbow + silhouette) → optimal k → fit
    4. DBSCAN:  noise / outlier detection
    5. Dimensionality Reduction:  PCA 3D
    6. Labeling:  centroid analysis → business segment names
    7. Export:  enriched DataFrame + serialized models

Usage:
    python main.py                   # use default config.yaml
    python main.py --config my.yaml  # custom config path
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import yaml

from src.clustering import (
    compute_cluster_summary,
    evaluate_kmeans_range,
    fit_dbscan,
    fit_kmeans,
)
from src.dimensionality import fit_pca
from src.etl import run_etl
from src.feature_engineering import FeaturePipeline
from src.labeling import build_label_map, label_clusters

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
# Model persistence
# ---------------------------------------------------------------------------


def save_artifacts(
    output_dir: str | Path,
    pipeline: FeaturePipeline,
    kmeans_model: Any,
    pca_model: Any,
) -> None:
    """Serialize fitted models to disk.

    Args:
        output_dir: Directory to save into.
        pipeline: Fitted FeaturePipeline (scaler + categories).
        kmeans_model: Fitted KMeans estimator.
        pca_model: Fitted PCA estimator.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "feature_pipeline.pkl": pipeline,
        "kmeans_model.pkl": kmeans_model,
        "pca_model.pkl": pca_model,
    }

    for name, obj in artifacts.items():
        path = output_dir / name
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved %s → %s", name, path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: dict[str, Any]) -> None:
    """Execute the full segmentation pipeline.

    Args:
        config: Parsed YAML config dict.
    """
    data_cfg = config["data"]
    feat_cfg = config["features"]
    clust_cfg = config["clustering"]
    dim_cfg = config["dimensionality"]
    model_cfg = config["models"]

    # ── 1. ETL ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/6 — ETL")
    logger.info("=" * 60)

    df = run_etl(
        raw_dir=data_cfg["raw_dir"],
        output_path=data_cfg["output_parquet"],
        pattern=data_cfg.get("xml_glob", "*.xml"),
    )
    logger.info("ETL complete: %d orders loaded.", len(df))

    # ── 2. Feature Engineering ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/6 — Feature Engineering")
    logger.info("=" * 60)

    pipeline = FeaturePipeline.from_config(feat_cfg)
    X_scaled = pipeline.fit_transform(df)
    logger.info(
        "Features engineered: %d samples × %d features.",
        X_scaled.shape[0], X_scaled.shape[1],
    )

    # ── 3. K-Means Clustering ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/6 — K-Means Clustering (Elbow + Silhouette)")
    logger.info("=" * 60)

    km_cfg = clust_cfg["kmeans"]
    k_min, k_max = km_cfg["k_range"]

    elbow = evaluate_kmeans_range(
        X_scaled,
        k_min=k_min,
        k_max=k_max,
        n_init=km_cfg.get("n_init", 10),
        random_state=km_cfg.get("random_state", 42),
    )
    best_k = elbow.best_k_silhouette
    logger.info("Optimal k=%d (silhouette=%.4f)", best_k, max(elbow.silhouette_scores))

    km_model, km_labels = fit_kmeans(
        X_scaled,
        n_clusters=best_k,
        n_init=km_cfg.get("n_init", 10),
        random_state=km_cfg.get("random_state", 42),
    )

    # ── 4. DBSCAN ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4/6 — DBSCAN (noise detection)")
    logger.info("=" * 60)

    db_cfg = clust_cfg["dbscan"]
    dbscan_result = fit_dbscan(
        X_scaled,
        eps=db_cfg.get("eps", 0.5),
        min_samples=db_cfg.get("min_samples", 5),
    )
    logger.info(
        "DBSCAN: %d clusters, %d noise points (%.1f%%).",
        dbscan_result.n_clusters,
        dbscan_result.n_noise,
        100 * dbscan_result.n_noise / len(X_scaled),
    )

    # ── 5. PCA ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5/6 — PCA Dimensionality Reduction")
    logger.info("=" * 60)

    pca_cfg = dim_cfg["pca"]
    pca_result = fit_pca(X_scaled, n_components=pca_cfg.get("n_components", 3))
    logger.info(
        "PCA: %.1f%% variance explained by %d components.",
        pca_result.cumulative_variance[-1] * 100,
        pca_cfg.get("n_components", 3),
    )

    # ── 6. Labeling ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6/6 — Business Labeling")
    logger.info("=" * 60)

    # Compute summary on UNSCALED features for interpretability
    raw_features = feat_cfg["numerical"]
    available_raw = [c for c in raw_features if c in df.columns]
    raw_matrix = df[available_raw].values

    cluster_summary = compute_cluster_summary(
        raw_matrix, km_labels, available_raw
    )
    profiles = label_clusters(cluster_summary)
    label_map = build_label_map(profiles)

    for p in profiles:
        logger.info(
            "  Cluster %d → '%s' (n=%d)", p.cluster_id, p.name, p.size
        )

    # ── Export enriched DataFrame ──────────────────────────────────
    df["cluster_id"] = km_labels
    df["segment"] = df["cluster_id"].map(label_map)
    df["dbscan_label"] = dbscan_result.labels
    df["is_noise"] = dbscan_result.labels == -1
    df["pca_1"] = pca_result.embedding[:, 0]
    df["pca_2"] = pca_result.embedding[:, 1]
    df["pca_3"] = pca_result.embedding[:, 2]

    enriched_path = Path(data_cfg["processed_dir"]) / "orders_segmented.parquet"
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(enriched_path, index=False, engine="pyarrow")
    logger.info("Enriched DataFrame saved → %s (%d rows)", enriched_path, len(df))

    # ── Save models ────────────────────────────────────────────────
    save_artifacts(model_cfg["output_dir"], pipeline, km_model, pca_result.model)

    # ── Summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total orders:      %d", len(df))
    logger.info("K-Means clusters:  %d (best k by silhouette)", best_k)
    logger.info("DBSCAN noise:      %d (%.1f%%)",
                dbscan_result.n_noise, 100 * dbscan_result.n_noise / len(df))
    logger.info("PCA variance:      %.1f%%", pca_result.cumulative_variance[-1] * 100)
    logger.info("Output:            %s", enriched_path)
    logger.info("Models:            %s/", model_cfg["output_dir"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Customer Segmentation Engine — pipeline runner"
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
