"""Dimensionality reduction module: PCA and t-SNE.

Reduces high-dimensional scaled feature matrices to 2D or 3D
representations for visualization.

- **PCA** (fast, deterministic) — used in the Streamlit dashboard for
  real-time 3D scatter plots.
- **t-SNE** (slow, stochastic) — used in notebooks for exploratory 2D
  embeddings that better preserve local structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Container for PCA output.

    Attributes:
        embedding: Projected array of shape ``(n_samples, n_components)``.
        explained_variance_ratio: Variance explained by each component.
        cumulative_variance: Cumulative sum of explained variance.
        model: Fitted ``PCA`` estimator (for later ``transform`` calls).
    """

    embedding: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    model: PCA


def fit_pca(
    X: np.ndarray,
    n_components: int = 3,
) -> PCAResult:
    """Fit PCA and return the reduced embedding with diagnostics.

    PCA is a linear method that projects data onto the directions of
    maximum variance.  It is deterministic and fast — suitable for
    real-time dashboard use.

    Args:
        X: Scaled feature matrix ``(n_samples, n_features)``.
        n_components: Number of principal components to retain.

    Returns:
        ``PCAResult`` containing the embedding, explained variance,
        and the fitted model.

    Raises:
        ValueError: If ``n_components`` exceeds number of features or
            samples.
    """
    max_components = min(X.shape[0], X.shape[1])
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds max allowable "
            f"{max_components} (min of n_samples, n_features)."
        )

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    logger.info(
        "PCA fitted: %d components, cumulative variance=%.2f%%",
        n_components,
        cumulative[-1] * 100,
    )
    for i, (v, c) in enumerate(zip(evr, cumulative)):
        logger.debug("  PC%d: %.2f%% (cumulative: %.2f%%)", i + 1, v * 100, c * 100)

    return PCAResult(
        embedding=embedding,
        explained_variance_ratio=evr,
        cumulative_variance=cumulative,
        model=pca,
    )


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------


@dataclass
class TSNEResult:
    """Container for t-SNE output.

    Attributes:
        embedding: Projected array of shape ``(n_samples, n_components)``.
        kl_divergence: Final Kullback-Leibler divergence of the embedding.
        perplexity: Perplexity value used.
    """

    embedding: np.ndarray
    kl_divergence: float
    perplexity: float


def fit_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> TSNEResult:
    """Fit t-SNE and return a low-dimensional embedding.

    t-SNE is a non-linear method that preserves local neighborhood
    structure.  It is stochastic and computationally expensive — best
    suited for exploratory notebooks, not real-time dashboards.

    The perplexity parameter roughly controls how many neighbors each
    point considers.  Rule of thumb: ``5 <= perplexity <= 50``.  If
    ``perplexity >= n_samples``, it is automatically clamped.

    Args:
        X: Scaled feature matrix ``(n_samples, n_features)``.
        n_components: Output dimensionality (typically 2).
        perplexity: Effective number of neighbors.
        random_state: Seed for reproducibility.

    Returns:
        ``TSNEResult`` with the embedding and convergence diagnostics.
    """
    # Clamp perplexity to a safe value if dataset is small
    safe_perplexity = min(perplexity, max(1.0, (X.shape[0] - 1) / 3.0))
    if safe_perplexity != perplexity:
        logger.warning(
            "Perplexity clamped from %.1f to %.1f (n_samples=%d).",
            perplexity,
            safe_perplexity,
            X.shape[0],
        )

    tsne = TSNE(
        n_components=n_components,
        perplexity=safe_perplexity,
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X)

    kl_div = float(tsne.kl_divergence_)
    logger.info(
        "t-SNE fitted: %d components, perplexity=%.1f, KL divergence=%.4f",
        n_components,
        safe_perplexity,
        kl_div,
    )

    return TSNEResult(
        embedding=embedding,
        kl_divergence=kl_div,
        perplexity=safe_perplexity,
    )


# ---------------------------------------------------------------------------
# Convenience: embedding → DataFrame
# ---------------------------------------------------------------------------


def embedding_to_dataframe(
    embedding: np.ndarray,
    labels: np.ndarray | None = None,
    prefix: str = "dim",
) -> "pd.DataFrame":
    """Convert an embedding array to a labeled DataFrame.

    Args:
        embedding: Array of shape ``(n_samples, n_components)``.
        labels: Optional cluster labels to attach.
        prefix: Column name prefix (e.g. ``"PC"`` or ``"tSNE"``).

    Returns:
        DataFrame with columns ``{prefix}_1``, ``{prefix}_2``, ...
        and optionally a ``cluster`` column.
    """
    import pandas as pd

    n_components = embedding.shape[1]
    columns = [f"{prefix}_{i + 1}" for i in range(n_components)]
    df = pd.DataFrame(embedding, columns=columns)

    if labels is not None:
        df["cluster"] = labels

    return df
