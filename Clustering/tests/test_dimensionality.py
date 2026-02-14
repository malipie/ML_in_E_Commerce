"""Unit tests for src/dimensionality.py — PCA, t-SNE, helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.dimensionality import (
    PCAResult,
    TSNEResult,
    embedding_to_dataframe,
    fit_pca,
    fit_tsne,
)

RANDOM_STATE = 42


@pytest.fixture
def scaled_data() -> np.ndarray:
    """Generate 100 samples with 8 features (3 informative blobs)."""
    X, _ = make_blobs(
        n_samples=100,
        n_features=8,
        centers=3,
        cluster_std=1.0,
        random_state=RANDOM_STATE,
    )
    return X


@pytest.fixture
def small_data() -> np.ndarray:
    """Very small dataset (10 samples, 4 features)."""
    X, _ = make_blobs(
        n_samples=10,
        n_features=4,
        centers=2,
        random_state=RANDOM_STATE,
    )
    return X


# ===================================================================
# PCA
# ===================================================================


class TestFitPca:
    """Tests for fit_pca()."""

    def test_returns_pca_result(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=3)
        assert isinstance(result, PCAResult)

    def test_embedding_shape(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=3)
        assert result.embedding.shape == (100, 3)

    def test_embedding_shape_2d(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=2)
        assert result.embedding.shape == (100, 2)

    def test_explained_variance_sums_to_at_most_one(
        self, scaled_data: np.ndarray
    ) -> None:
        result = fit_pca(scaled_data, n_components=3)
        assert result.explained_variance_ratio.sum() <= 1.0 + 1e-10

    def test_explained_variance_length(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=3)
        assert len(result.explained_variance_ratio) == 3

    def test_cumulative_variance_monotonic(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=5)
        for i in range(1, len(result.cumulative_variance)):
            assert result.cumulative_variance[i] >= result.cumulative_variance[i - 1]

    def test_cumulative_variance_last_equals_sum(
        self, scaled_data: np.ndarray
    ) -> None:
        result = fit_pca(scaled_data, n_components=3)
        assert result.cumulative_variance[-1] == pytest.approx(
            result.explained_variance_ratio.sum()
        )

    def test_model_can_transform(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=3)
        new_data = scaled_data[:5]
        projected = result.model.transform(new_data)
        assert projected.shape == (5, 3)

    def test_deterministic(self, scaled_data: np.ndarray) -> None:
        r1 = fit_pca(scaled_data, n_components=3)
        r2 = fit_pca(scaled_data, n_components=3)
        np.testing.assert_array_almost_equal(r1.embedding, r2.embedding)

    def test_too_many_components_raises(self, small_data: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_components.*exceeds"):
            fit_pca(small_data, n_components=20)

    def test_no_nan_in_embedding(self, scaled_data: np.ndarray) -> None:
        result = fit_pca(scaled_data, n_components=3)
        assert not np.isnan(result.embedding).any()


# ===================================================================
# t-SNE
# ===================================================================


class TestFitTsne:
    """Tests for fit_tsne()."""

    def test_returns_tsne_result(self, scaled_data: np.ndarray) -> None:
        result = fit_tsne(scaled_data, n_components=2)
        assert isinstance(result, TSNEResult)

    def test_embedding_shape(self, scaled_data: np.ndarray) -> None:
        result = fit_tsne(scaled_data, n_components=2)
        assert result.embedding.shape == (100, 2)

    def test_kl_divergence_non_negative(self, scaled_data: np.ndarray) -> None:
        result = fit_tsne(scaled_data, n_components=2)
        assert result.kl_divergence >= 0.0

    def test_perplexity_clamped_for_small_data(self, small_data: np.ndarray) -> None:
        result = fit_tsne(small_data, perplexity=50.0)
        # With 10 samples, perplexity should be clamped to (10-1)/3 = 3.0
        assert result.perplexity <= (small_data.shape[0] - 1) / 3.0

    def test_reproducible_with_same_seed(self, scaled_data: np.ndarray) -> None:
        r1 = fit_tsne(scaled_data, random_state=42)
        r2 = fit_tsne(scaled_data, random_state=42)
        np.testing.assert_array_almost_equal(r1.embedding, r2.embedding)

    def test_no_nan_in_embedding(self, scaled_data: np.ndarray) -> None:
        result = fit_tsne(scaled_data, n_components=2)
        assert not np.isnan(result.embedding).any()


# ===================================================================
# embedding_to_dataframe
# ===================================================================


class TestEmbeddingToDataframe:
    """Tests for embedding_to_dataframe()."""

    def test_returns_dataframe(self) -> None:
        emb = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = embedding_to_dataframe(emb)
        assert isinstance(result, pd.DataFrame)

    def test_column_names_default_prefix(self) -> None:
        emb = np.array([[1.0, 2.0, 3.0]])
        result = embedding_to_dataframe(emb)
        assert list(result.columns) == ["dim_1", "dim_2", "dim_3"]

    def test_column_names_custom_prefix(self) -> None:
        emb = np.array([[1.0, 2.0]])
        result = embedding_to_dataframe(emb, prefix="PC")
        assert list(result.columns) == ["PC_1", "PC_2"]

    def test_with_labels(self) -> None:
        emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([0, 1, 0])
        result = embedding_to_dataframe(emb, labels=labels, prefix="tSNE")
        assert "cluster" in result.columns
        assert list(result["cluster"]) == [0, 1, 0]

    def test_without_labels_no_cluster_column(self) -> None:
        emb = np.array([[1.0, 2.0]])
        result = embedding_to_dataframe(emb)
        assert "cluster" not in result.columns

    def test_row_count_matches(self) -> None:
        emb = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = embedding_to_dataframe(emb)
        assert len(result) == 5
