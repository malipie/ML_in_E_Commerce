"""Clustering module: K-Means (elbow + silhouette) and DBSCAN.

Provides functions for fitting clustering models on scaled feature matrices,
evaluating cluster quality, and selecting the optimal number of clusters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elbow & Silhouette analysis
# ---------------------------------------------------------------------------


@dataclass
class ElbowResult:
    """Results from an elbow / silhouette sweep over k values.

    Attributes:
        k_values: Tested cluster counts.
        inertias: Within-cluster sum of squares per k.
        silhouette_scores: Mean silhouette coefficient per k.
        calinski_harabasz_scores: CH index per k.
        best_k_silhouette: k with highest silhouette score.
    """

    k_values: list[int]
    inertias: list[float]
    silhouette_scores: list[float]
    calinski_harabasz_scores: list[float]
    best_k_silhouette: int


def evaluate_kmeans_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    n_init: int = 10,
    random_state: int = 42,
) -> ElbowResult:
    """Run K-Means for a range of k and compute quality metrics.

    For each k in ``[k_min, k_max]`` fits a K-Means model and records:
    - **Inertia** (within-cluster sum of squares) — for elbow method.
    - **Silhouette score** — higher is better, range [-1, 1].
    - **Calinski-Harabasz index** — higher is better.

    Args:
        X: Scaled feature matrix of shape ``(n_samples, n_features)``.
        k_min: Minimum number of clusters (inclusive, >= 2).
        k_max: Maximum number of clusters (inclusive).
        n_init: Number of K-Means restarts per k.
        random_state: Random seed for reproducibility.

    Returns:
        ``ElbowResult`` containing metrics for each k and the best k
        according to silhouette score.

    Raises:
        ValueError: If ``k_min < 2`` or ``k_max < k_min`` or not enough
            samples.
    """
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, got {k_min}.")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min}).")
    if X.shape[0] < k_max:
        raise ValueError(
            f"Need at least k_max={k_max} samples, got {X.shape[0]}."
        )

    k_values: list[int] = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    sil_scores: list[float] = []
    ch_scores: list[float] = []

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)

        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(X, labels)))
        ch_scores.append(float(calinski_harabasz_score(X, labels)))

        logger.info(
            "k=%d  inertia=%.2f  silhouette=%.4f  CH=%.2f",
            k, km.inertia_, sil_scores[-1], ch_scores[-1],
        )

    best_k = k_values[int(np.argmax(sil_scores))]
    logger.info("Best k by silhouette: %d (score=%.4f)", best_k, max(sil_scores))

    return ElbowResult(
        k_values=k_values,
        inertias=inertias,
        silhouette_scores=sil_scores,
        calinski_harabasz_scores=ch_scores,
        best_k_silhouette=best_k,
    )


# ---------------------------------------------------------------------------
# K-Means fitting
# ---------------------------------------------------------------------------


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """Fit a K-Means model and return it with cluster labels.

    Args:
        X: Scaled feature matrix ``(n_samples, n_features)``.
        n_clusters: Number of clusters.
        n_init: Number of restarts.
        random_state: Random seed.

    Returns:
        Tuple of (fitted KMeans model, labels array of shape (n_samples,)).
    """
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    logger.info(
        "K-Means fitted: k=%d, inertia=%.2f, silhouette=%.4f",
        n_clusters, km.inertia_, silhouette_score(X, labels),
    )
    return km, labels


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


@dataclass
class DBSCANResult:
    """Results from a DBSCAN fit.

    Attributes:
        model: Fitted DBSCAN estimator.
        labels: Cluster labels (``-1`` = noise).
        n_clusters: Number of clusters found (excluding noise).
        n_noise: Number of noise points.
        silhouette: Silhouette score (``None`` if < 2 clusters or all noise).
    """

    model: DBSCAN
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    silhouette: float | None


def fit_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> DBSCANResult:
    """Fit DBSCAN for density-based clustering and noise detection.

    DBSCAN identifies dense regions as clusters and marks sparse points
    as noise (label ``-1``).  Useful for detecting outlier transactions.

    Args:
        X: Scaled feature matrix ``(n_samples, n_features)``.
        eps: Maximum distance between two samples in the same neighborhood.
        min_samples: Minimum points required to form a dense region.

    Returns:
        ``DBSCANResult`` with model, labels, and quality metrics.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    sil: float | None = None
    if n_clusters >= 2 and n_noise < len(labels):
        # Compute silhouette only on non-noise points if possible
        non_noise_mask = labels != -1
        if len(set(labels[non_noise_mask])) >= 2:
            sil = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))

    logger.info(
        "DBSCAN fitted: eps=%.3f, min_samples=%d → %d clusters, %d noise points, "
        "silhouette=%s",
        eps, min_samples, n_clusters, n_noise,
        f"{sil:.4f}" if sil is not None else "N/A",
    )

    return DBSCANResult(
        model=db,
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=sil,
    )


# ---------------------------------------------------------------------------
# Cluster summary statistics
# ---------------------------------------------------------------------------


def compute_cluster_summary(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> dict[int, dict[str, float]]:
    """Compute per-cluster mean values for each feature.

    Useful for interpreting cluster centroids and assigning business labels.

    Args:
        X: Feature matrix (scaled or unscaled).
        labels: Cluster assignments.
        feature_names: Names corresponding to columns of ``X``.

    Returns:
        Dict mapping cluster label → dict of ``{feature_name: mean_value}``.
        Includes a ``"size"`` key with the cluster population count.
    """
    unique_labels = sorted(set(labels))
    summary: dict[int, dict[str, float]] = {}

    for label in unique_labels:
        mask = labels == label
        cluster_data = X[mask]
        stats: dict[str, float] = {"size": int(mask.sum())}
        for i, name in enumerate(feature_names):
            stats[name] = float(cluster_data[:, i].mean())
        summary[label] = stats

    return summary
