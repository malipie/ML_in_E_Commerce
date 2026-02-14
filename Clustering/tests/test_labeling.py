"""Unit tests for src/labeling.py — cluster labeling and ranking."""

from __future__ import annotations

import pytest

from src.labeling import (
    NOISE_LABEL,
    NOISE_NAME,
    SegmentProfile,
    _classify_segment,
    _rank_clusters,
    build_label_map,
    label_clusters,
)


# ===================================================================
# Fixtures — synthetic cluster summaries
# ===================================================================


@pytest.fixture
def three_cluster_summary() -> dict[int, dict[str, float]]:
    """Three distinct clusters: premium, budget, standard."""
    return {
        0: {
            "size": 100,
            "order_amount_brutto": 500.0,
            "n_items": 1.2,
            "avg_item_price": 450.0,
        },
        1: {
            "size": 300,
            "order_amount_brutto": 80.0,
            "n_items": 4.5,
            "avg_item_price": 18.0,
        },
        2: {
            "size": 200,
            "order_amount_brutto": 180.0,
            "n_items": 2.0,
            "avg_item_price": 90.0,
        },
    }


@pytest.fixture
def summary_with_noise() -> dict[int, dict[str, float]]:
    """Two clusters + noise cluster (-1)."""
    return {
        -1: {
            "size": 15,
            "order_amount_brutto": 3000.0,
            "n_items": 1.0,
            "avg_item_price": 3000.0,
        },
        0: {
            "size": 200,
            "order_amount_brutto": 150.0,
            "n_items": 2.0,
            "avg_item_price": 75.0,
        },
        1: {
            "size": 185,
            "order_amount_brutto": 350.0,
            "n_items": 1.5,
            "avg_item_price": 250.0,
        },
    }


@pytest.fixture
def single_cluster_summary() -> dict[int, dict[str, float]]:
    """Edge case: only one cluster."""
    return {
        0: {
            "size": 500,
            "order_amount_brutto": 200.0,
            "n_items": 2.0,
            "avg_item_price": 100.0,
        },
    }


# ===================================================================
# Rank helpers
# ===================================================================


class TestRankClusters:
    """Tests for _rank_clusters()."""

    def test_three_clusters_correct_ranks(
        self, three_cluster_summary: dict
    ) -> None:
        ranks = _rank_clusters(three_cluster_summary, "order_amount_brutto")
        # Cluster 0 has highest (500), cluster 1 lowest (80)
        assert ranks[0] == 1.0
        assert ranks[1] == 0.0
        assert ranks[2] == 0.5

    def test_noise_excluded_by_default(
        self, summary_with_noise: dict
    ) -> None:
        ranks = _rank_clusters(summary_with_noise, "order_amount_brutto")
        assert NOISE_LABEL not in ranks

    def test_noise_included_when_requested(
        self, summary_with_noise: dict
    ) -> None:
        ranks = _rank_clusters(
            summary_with_noise, "order_amount_brutto", exclude_noise=False
        )
        assert NOISE_LABEL in ranks

    def test_single_cluster_gets_0_5(
        self, single_cluster_summary: dict
    ) -> None:
        ranks = _rank_clusters(single_cluster_summary, "order_amount_brutto")
        assert ranks[0] == 0.5

    def test_ranks_bounded_0_to_1(
        self, three_cluster_summary: dict
    ) -> None:
        ranks = _rank_clusters(three_cluster_summary, "order_amount_brutto")
        for r in ranks.values():
            assert 0.0 <= r <= 1.0

    def test_missing_feature_returns_empty(
        self, three_cluster_summary: dict
    ) -> None:
        ranks = _rank_clusters(three_cluster_summary, "nonexistent_feature")
        assert ranks == {}


# ===================================================================
# Classify segment
# ===================================================================


class TestClassifySegment:
    """Tests for _classify_segment()."""

    def test_high_monetary_low_basket(self) -> None:
        name, desc = _classify_segment(0, monetary_rank=0.9, basket_rank=0.1,
                                        avg_price_rank=0.9, size=100)
        assert name == "Premium Single-Item"

    def test_high_monetary_high_basket(self) -> None:
        name, _ = _classify_segment(0, monetary_rank=0.9, basket_rank=0.9,
                                     avg_price_rank=0.5, size=100)
        assert name == "High-Value Multi-Item"

    def test_low_monetary_high_basket(self) -> None:
        name, _ = _classify_segment(0, monetary_rank=0.1, basket_rank=0.9,
                                     avg_price_rank=0.1, size=100)
        assert name == "Budget Multi-Item"

    def test_low_monetary_low_basket(self) -> None:
        name, _ = _classify_segment(0, monetary_rank=0.1, basket_rank=0.1,
                                     avg_price_rank=0.1, size=100)
        assert name == "Small Quick Purchases"

    def test_mid_range_is_standard(self) -> None:
        name, _ = _classify_segment(0, monetary_rank=0.5, basket_rank=0.5,
                                     avg_price_rank=0.5, size=100)
        assert name == "Standard Orders"

    def test_returns_tuple_of_strings(self) -> None:
        result = _classify_segment(0, 0.5, 0.5, 0.5, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ===================================================================
# Label clusters (integration)
# ===================================================================


class TestLabelClusters:
    """Tests for label_clusters()."""

    def test_returns_list_of_profiles(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        assert isinstance(profiles, list)
        assert all(isinstance(p, SegmentProfile) for p in profiles)

    def test_correct_number_of_profiles(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        assert len(profiles) == 3

    def test_sorted_by_cluster_id(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        ids = [p.cluster_id for p in profiles]
        assert ids == sorted(ids)

    def test_noise_labeled_correctly(
        self, summary_with_noise: dict
    ) -> None:
        profiles = label_clusters(summary_with_noise)
        noise_profile = [p for p in profiles if p.cluster_id == NOISE_LABEL]
        assert len(noise_profile) == 1
        assert noise_profile[0].name == NOISE_NAME

    def test_noise_ranks_are_negative(
        self, summary_with_noise: dict
    ) -> None:
        profiles = label_clusters(summary_with_noise)
        noise = [p for p in profiles if p.cluster_id == NOISE_LABEL][0]
        assert noise.monetary_rank == -1.0
        assert noise.basket_size_rank == -1.0

    def test_size_preserved(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        profile_map = {p.cluster_id: p for p in profiles}
        assert profile_map[0].size == 100
        assert profile_map[1].size == 300
        assert profile_map[2].size == 200

    def test_highest_monetary_gets_premium_or_high(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        # Cluster 0 has monetary_rank=1.0 → should be Premium or High-*
        profile_0 = [p for p in profiles if p.cluster_id == 0][0]
        assert "Premium" in profile_0.name or "High" in profile_0.name

    def test_all_names_non_empty(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        for p in profiles:
            assert len(p.name) > 0
            assert len(p.description) > 0

    def test_single_cluster(
        self, single_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(single_cluster_summary)
        assert len(profiles) == 1
        assert profiles[0].name  # has a name


# ===================================================================
# Build label map
# ===================================================================


class TestBuildLabelMap:
    """Tests for build_label_map()."""

    def test_returns_dict(self, three_cluster_summary: dict) -> None:
        profiles = label_clusters(three_cluster_summary)
        lmap = build_label_map(profiles)
        assert isinstance(lmap, dict)

    def test_keys_match_cluster_ids(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        lmap = build_label_map(profiles)
        assert set(lmap.keys()) == {0, 1, 2}

    def test_values_are_strings(
        self, three_cluster_summary: dict
    ) -> None:
        profiles = label_clusters(three_cluster_summary)
        lmap = build_label_map(profiles)
        for v in lmap.values():
            assert isinstance(v, str)

    def test_noise_included(self, summary_with_noise: dict) -> None:
        profiles = label_clusters(summary_with_noise)
        lmap = build_label_map(profiles)
        assert NOISE_LABEL in lmap
        assert lmap[NOISE_LABEL] == NOISE_NAME
