"""Unit tests for src/clustering.py — K-Means, DBSCAN, metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.clustering import (
    DBSCANResult,
    ElbowResult,
    compute_cluster_summary,
    evaluate_kmeans_range,
    fit_dbscan,
    fit_kmeans,
)

# ===================================================================
# Fixtures
# ===================================================================

RANDOM_STATE = 42


@pytest.fixture
def blob_data_3() -> np.ndarray:
    """Generate 3 well-separated blobs (150 samples, 4 features)."""
    X, _ = make_blobs(
        n_samples=150,
        n_features=4,
        centers=3,
        cluster_std=0.5,
        random_state=RANDOM_STATE,
    )
    return X


@pytest.fixture
def blob_data_5() -> np.ndarray:
    """Generate 5 well-separated blobs (250 samples, 4 features)."""
    X, _ = make_blobs(
        n_samples=250,
        n_features=4,
        centers=5,
        cluster_std=0.5,
        random_state=RANDOM_STATE,
    )
    return X


@pytest.fixture
def uniform_noise() -> np.ndarray:
    """Generate uniform random data (no clusters)."""
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.uniform(-10, 10, size=(100, 4))


# ===================================================================
# Elbow / Silhouette analysis
# ===================================================================


class TestEvaluateKmeansRange:
    """Tests for evaluate_kmeans_range()."""

    def test_returns_elbow_result(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=5)
        assert isinstance(result, ElbowResult)

    def test_correct_k_values(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=6)
        assert result.k_values == [2, 3, 4, 5, 6]

    def test_inertias_decrease(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=6)
        # Inertia must decrease monotonically as k grows
        for i in range(1, len(result.inertias)):
            assert result.inertias[i] <= result.inertias[i - 1]

    def test_best_k_finds_true_clusters(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=6)
        # With well-separated blobs, best k by silhouette should be 3
        assert result.best_k_silhouette == 3

    def test_silhouette_scores_bounded(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=5)
        for s in result.silhouette_scores:
            assert -1.0 <= s <= 1.0

    def test_calinski_harabasz_positive(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=5)
        for ch in result.calinski_harabasz_scores:
            assert ch > 0

    def test_k_min_less_than_2_raises(self, blob_data_3: np.ndarray) -> None:
        with pytest.raises(ValueError, match="k_min must be >= 2"):
            evaluate_kmeans_range(blob_data_3, k_min=1, k_max=5)

    def test_k_max_less_than_k_min_raises(self, blob_data_3: np.ndarray) -> None:
        with pytest.raises(ValueError, match="k_max.*must be >= k_min"):
            evaluate_kmeans_range(blob_data_3, k_min=5, k_max=3)

    def test_not_enough_samples_raises(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(ValueError, match="Need at least"):
            evaluate_kmeans_range(X, k_min=2, k_max=10)

    def test_lists_same_length(self, blob_data_3: np.ndarray) -> None:
        result = evaluate_kmeans_range(blob_data_3, k_min=2, k_max=7)
        n = len(result.k_values)
        assert len(result.inertias) == n
        assert len(result.silhouette_scores) == n
        assert len(result.calinski_harabasz_scores) == n


# ===================================================================
# K-Means fitting
# ===================================================================


class TestFitKmeans:
    """Tests for fit_kmeans()."""

    def test_returns_model_and_labels(self, blob_data_3: np.ndarray) -> None:
        model, labels = fit_kmeans(blob_data_3, n_clusters=3)
        assert hasattr(model, "cluster_centers_")
        assert labels.shape == (blob_data_3.shape[0],)

    def test_correct_number_of_clusters(self, blob_data_3: np.ndarray) -> None:
        _, labels = fit_kmeans(blob_data_3, n_clusters=3)
        assert len(set(labels)) == 3

    def test_labels_are_integers(self, blob_data_3: np.ndarray) -> None:
        _, labels = fit_kmeans(blob_data_3, n_clusters=3)
        assert labels.dtype in (np.int32, np.int64)

    def test_reproducible_with_same_seed(self, blob_data_3: np.ndarray) -> None:
        _, labels1 = fit_kmeans(blob_data_3, n_clusters=3, random_state=42)
        _, labels2 = fit_kmeans(blob_data_3, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(labels1, labels2)

    def test_centroids_shape(self, blob_data_3: np.ndarray) -> None:
        model, _ = fit_kmeans(blob_data_3, n_clusters=4)
        assert model.cluster_centers_.shape == (4, blob_data_3.shape[1])


# ===================================================================
# DBSCAN
# ===================================================================


class TestFitDbscan:
    """Tests for fit_dbscan()."""

    def test_returns_dbscan_result(self, blob_data_3: np.ndarray) -> None:
        result = fit_dbscan(blob_data_3, eps=1.0, min_samples=5)
        assert isinstance(result, DBSCANResult)

    def test_finds_clusters_in_blobs(self, blob_data_3: np.ndarray) -> None:
        result = fit_dbscan(blob_data_3, eps=1.0, min_samples=5)
        assert result.n_clusters >= 2  # should find at least 2 clusters

    def test_labels_shape(self, blob_data_3: np.ndarray) -> None:
        result = fit_dbscan(blob_data_3, eps=1.0, min_samples=5)
        assert result.labels.shape == (blob_data_3.shape[0],)

    def test_noise_label_is_minus_one(self, uniform_noise: np.ndarray) -> None:
        # With tight eps on uniform data, most points should be noise
        result = fit_dbscan(uniform_noise, eps=0.1, min_samples=10)
        assert result.n_noise > 0
        assert -1 in result.labels

    def test_n_noise_count_consistent(self, blob_data_3: np.ndarray) -> None:
        result = fit_dbscan(blob_data_3, eps=1.0, min_samples=5)
        actual_noise = int(np.sum(result.labels == -1))
        assert result.n_noise == actual_noise

    def test_silhouette_computed_when_possible(self, blob_data_3: np.ndarray) -> None:
        result = fit_dbscan(blob_data_3, eps=1.0, min_samples=5)
        if result.n_clusters >= 2:
            assert result.silhouette is not None
            assert -1.0 <= result.silhouette <= 1.0

    def test_silhouette_none_when_single_cluster(self) -> None:
        # Very large eps → everything in one cluster
        X = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]], dtype=float)
        result = fit_dbscan(X, eps=100.0, min_samples=2)
        assert result.n_clusters <= 1
        assert result.silhouette is None

    def test_all_noise_silhouette_none(self) -> None:
        X = np.array([[0, 0], [100, 100], [200, 200]], dtype=float)
        result = fit_dbscan(X, eps=0.001, min_samples=5)
        assert result.n_noise == len(X)
        assert result.silhouette is None


# ===================================================================
# Cluster summary
# ===================================================================


class TestComputeClusterSummary:
    """Tests for compute_cluster_summary()."""

    def test_returns_dict_per_cluster(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        labels = np.array([0, 0, 1, 1])
        summary = compute_cluster_summary(X, labels, ["feat_a", "feat_b"])
        assert 0 in summary
        assert 1 in summary

    def test_size_correct(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        labels = np.array([0, 0, 0, 1, 1])
        summary = compute_cluster_summary(X, labels, ["val"])
        assert summary[0]["size"] == 3
        assert summary[1]["size"] == 2

    def test_mean_values_correct(self) -> None:
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        labels = np.array([0, 0])
        summary = compute_cluster_summary(X, labels, ["a", "b"])
        assert summary[0]["a"] == pytest.approx(20.0)
        assert summary[0]["b"] == pytest.approx(30.0)

    def test_noise_cluster_included(self) -> None:
        X = np.array([[1.0], [2.0], [100.0]])
        labels = np.array([0, 0, -1])
        summary = compute_cluster_summary(X, labels, ["val"])
        assert -1 in summary
        assert summary[-1]["size"] == 1

    def test_all_features_present(self) -> None:
        X = np.array([[1.0, 2.0, 3.0]])
        labels = np.array([0])
        names = ["x", "y", "z"]
        summary = compute_cluster_summary(X, labels, names)
        for name in names:
            assert name in summary[0]
        assert "size" in summary[0]
