"""Labeling module: map cluster centroids to business segment names.

Uses a rank-based strategy on unscaled (original-space) cluster means
to assign interpretable labels like "Premium Single-Item" or "Budget
Multi-Item".  The approach is relative (percentile ranks across clusters)
rather than absolute thresholds, making it robust to different datasets
and cluster counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — key features used for labeling decisions
# ---------------------------------------------------------------------------

KEY_MONETARY = "order_amount_brutto"
KEY_N_ITEMS = "n_items"
KEY_AVG_PRICE = "avg_item_price"

NOISE_LABEL = -1
NOISE_NAME = "Outliers / Noise"


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------


def _rank_clusters(
    summary: dict[int, dict[str, float]],
    feature: str,
    exclude_noise: bool = True,
) -> dict[int, float]:
    """Rank clusters by a feature mean, returning percentile ranks [0, 1].

    A rank of 1.0 means the cluster has the highest mean for that feature;
    0.0 means the lowest.

    Args:
        summary: Output of ``compute_cluster_summary()``.
        feature: Feature name to rank by.
        exclude_noise: If ``True``, skip cluster label ``-1``.

    Returns:
        Dict mapping cluster label → percentile rank.
    """
    filtered = {
        k: v[feature]
        for k, v in summary.items()
        if not (exclude_noise and k == NOISE_LABEL) and feature in v
    }

    if not filtered:
        return {}

    sorted_labels = sorted(filtered, key=lambda k: filtered[k])
    n = len(sorted_labels)

    if n == 1:
        return {sorted_labels[0]: 0.5}

    return {
        label: i / (n - 1)
        for i, label in enumerate(sorted_labels)
    }


# ---------------------------------------------------------------------------
# Labeling rules
# ---------------------------------------------------------------------------


@dataclass
class SegmentProfile:
    """Business profile for a single cluster.

    Attributes:
        cluster_id: Numeric cluster label.
        name: Human-readable segment name.
        description: Short business interpretation.
        size: Number of observations in this cluster.
        monetary_rank: Percentile rank by monetary value [0, 1].
        basket_size_rank: Percentile rank by number of items [0, 1].
    """

    cluster_id: int
    name: str
    description: str
    size: int
    monetary_rank: float
    basket_size_rank: float


def _classify_segment(
    cluster_id: int,
    monetary_rank: float,
    basket_rank: float,
    avg_price_rank: float,
    size: int,
) -> tuple[str, str]:
    """Assign a name and description based on rank combination.

    The logic uses a simple decision tree on percentile ranks:

    - High monetary + low basket → Premium Single-Item
    - High monetary + high basket → High-Value Multi-Item
    - Low monetary + high basket → Budget Multi-Item
    - Low monetary + low basket → Small Quick Purchases
    - Middle range → Standard Orders

    Args:
        cluster_id: The cluster label.
        monetary_rank: Percentile rank for monetary value.
        basket_rank: Percentile rank for basket size.
        avg_price_rank: Percentile rank for avg item price.
        size: Cluster population.

    Returns:
        Tuple of (segment_name, segment_description).
    """
    HIGH = 0.7
    LOW = 0.3

    if monetary_rank >= HIGH and basket_rank < LOW:
        return (
            "Premium Single-Item",
            "High-value orders with few expensive products. "
            "Premium segment — focus on upselling accessories.",
        )

    if monetary_rank >= HIGH and basket_rank >= HIGH:
        return (
            "High-Value Multi-Item",
            "Large orders with many items at above-average prices. "
            "Wholesale or gift-buyers — offer bundle discounts.",
        )

    if monetary_rank >= HIGH:
        return (
            "High Spenders",
            "Above-average order values. "
            "Key revenue drivers — prioritize retention.",
        )

    if monetary_rank < LOW and basket_rank >= HIGH:
        return (
            "Budget Multi-Item",
            "Many items per order but low total value. "
            "Price-sensitive bulk buyers — promote clearance items.",
        )

    if monetary_rank < LOW and basket_rank < LOW:
        return (
            "Small Quick Purchases",
            "Low-value, single-item orders. "
            "Impulse buyers — optimize checkout speed.",
        )

    if monetary_rank < LOW:
        return (
            "Budget Shoppers",
            "Below-average order values. "
            "Price-sensitive segment — target with promotions.",
        )

    # Middle range
    return (
        "Standard Orders",
        "Average-value orders representing the mainstream. "
        "Core customer base — maintain with loyalty programs.",
    )


def label_clusters(
    summary: dict[int, dict[str, float]],
) -> list[SegmentProfile]:
    """Assign business labels to all clusters.

    Takes the output of ``compute_cluster_summary()`` (computed on
    **unscaled** features for interpretability) and returns a
    ``SegmentProfile`` for each cluster.

    Args:
        summary: Dict mapping cluster_id → {feature: mean, "size": n}.

    Returns:
        List of ``SegmentProfile`` sorted by cluster_id.
    """
    monetary_ranks = _rank_clusters(summary, KEY_MONETARY)
    basket_ranks = _rank_clusters(summary, KEY_N_ITEMS)
    price_ranks = _rank_clusters(summary, KEY_AVG_PRICE)

    profiles: list[SegmentProfile] = []

    for cluster_id in sorted(summary.keys()):
        size = int(summary[cluster_id].get("size", 0))

        if cluster_id == NOISE_LABEL:
            profiles.append(SegmentProfile(
                cluster_id=cluster_id,
                name=NOISE_NAME,
                description="Transactions identified as outliers by DBSCAN. "
                            "Investigate for fraud, errors, or unusual patterns.",
                size=size,
                monetary_rank=-1.0,
                basket_size_rank=-1.0,
            ))
            continue

        m_rank = monetary_ranks.get(cluster_id, 0.5)
        b_rank = basket_ranks.get(cluster_id, 0.5)
        p_rank = price_ranks.get(cluster_id, 0.5)

        name, description = _classify_segment(
            cluster_id, m_rank, b_rank, p_rank, size
        )

        profiles.append(SegmentProfile(
            cluster_id=cluster_id,
            name=name,
            description=description,
            size=size,
            monetary_rank=m_rank,
            basket_size_rank=b_rank,
        ))

        logger.info(
            "Cluster %d → '%s' (size=%d, monetary_rank=%.2f, basket_rank=%.2f)",
            cluster_id, name, size, m_rank, b_rank,
        )

    return profiles


def build_label_map(profiles: list[SegmentProfile]) -> dict[int, str]:
    """Convert a list of SegmentProfiles to a simple {cluster_id: name} map.

    Useful for mapping cluster labels back onto a DataFrame column.

    Args:
        profiles: Output of ``label_clusters()``.

    Returns:
        Dict mapping cluster_id → segment name.
    """
    return {p.cluster_id: p.name for p in profiles}
