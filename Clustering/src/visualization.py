"""Visualization module: Plotly charts for cluster analysis.

Provides ready-to-use figure factories for the Streamlit dashboard
and Jupyter notebooks.  All functions return ``plotly.graph_objects.Figure``
objects that can be displayed with ``fig.show()`` or embedded via
``st.plotly_chart(fig)``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.labeling import SegmentProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette — consistent segment colors
# ---------------------------------------------------------------------------

SEGMENT_COLORS: list[str] = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
    "#FF6692",  # pink
    "#B6E880",  # lime
    "#FF97FF",  # magenta
    "#FECB52",  # yellow
]

NOISE_COLOR = "#7F7F7F"  # grey


def _color_for_cluster(cluster_id: int, label_map: dict[int, str]) -> str:
    """Return a consistent color for a cluster label."""
    if cluster_id == -1:
        return NOISE_COLOR
    ids = sorted(k for k in label_map if k != -1)
    idx = ids.index(cluster_id) if cluster_id in ids else 0
    return SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]


# ---------------------------------------------------------------------------
# 1. Elbow & Silhouette plot
# ---------------------------------------------------------------------------


def plot_elbow(
    k_values: list[int],
    inertias: list[float],
    silhouette_scores: list[float],
    best_k: int | None = None,
) -> go.Figure:
    """Create a dual-axis elbow + silhouette plot.

    Left y-axis shows inertia (elbow method), right y-axis shows
    silhouette score.  A vertical dashed line marks the best k.

    Args:
        k_values: List of tested k values.
        inertias: Inertia per k.
        silhouette_scores: Silhouette score per k.
        best_k: Optimal k to highlight (optional).

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=k_values, y=inertias,
            mode="lines+markers",
            name="Inertia",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=k_values, y=silhouette_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#EF553B", width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )

    if best_k is not None:
        fig.add_vline(
            x=best_k, line_dash="dash", line_color="green",
            annotation_text=f"Best k={best_k}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Elbow Method & Silhouette Analysis",
        xaxis_title="Number of Clusters (k)",
        template="plotly_white",
        legend=dict(x=0.5, y=1.15, xanchor="center", orientation="h"),
    )
    fig.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)

    return fig


# ---------------------------------------------------------------------------
# 2. 3D PCA scatter
# ---------------------------------------------------------------------------


def plot_pca_3d(
    embedding: np.ndarray,
    labels: np.ndarray,
    label_map: dict[int, str],
) -> go.Figure:
    """Create an interactive 3D scatter plot of PCA-reduced data.

    Points are colored by cluster assignment with segment names in the
    legend.

    Args:
        embedding: PCA embedding of shape ``(n, 3)``.
        labels: Cluster labels array.
        label_map: ``{cluster_id: segment_name}`` mapping.

    Returns:
        Plotly Figure.
    """
    df = pd.DataFrame(embedding, columns=["PC1", "PC2", "PC3"])
    df["cluster"] = labels
    df["segment"] = df["cluster"].map(label_map).fillna("Unknown")

    fig = go.Figure()

    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        color = _color_for_cluster(cid, label_map)
        name = label_map.get(cid, f"Cluster {cid}")

        fig.add_trace(go.Scatter3d(
            x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],
            mode="markers",
            name=name,
            marker=dict(size=3, color=color, opacity=0.7),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="3D PCA — Cluster Visualization",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        template="plotly_white",
        legend=dict(x=1.0, y=0.9),
    )

    return fig


# ---------------------------------------------------------------------------
# 3. 2D scatter (PCA or t-SNE)
# ---------------------------------------------------------------------------


def plot_scatter_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    label_map: dict[int, str],
    title: str = "2D Cluster Visualization",
    axis_prefix: str = "Dim",
) -> go.Figure:
    """Create a 2D scatter plot colored by cluster.

    Works with any 2D embedding (PCA 2-component or t-SNE).

    Args:
        embedding: 2D embedding ``(n, 2)``.
        labels: Cluster labels.
        label_map: ``{cluster_id: segment_name}``.
        title: Plot title.
        axis_prefix: Axis label prefix (e.g. "PC" or "t-SNE").

    Returns:
        Plotly Figure.
    """
    df = pd.DataFrame(
        embedding, columns=[f"{axis_prefix}_1", f"{axis_prefix}_2"]
    )
    df["cluster"] = labels
    df["segment"] = df["cluster"].map(label_map).fillna("Unknown")

    fig = go.Figure()

    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        color = _color_for_cluster(cid, label_map)
        name = label_map.get(cid, f"Cluster {cid}")

        fig.add_trace(go.Scatter(
            x=subset.iloc[:, 0], y=subset.iloc[:, 1],
            mode="markers",
            name=name,
            marker=dict(size=5, color=color, opacity=0.7),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{axis_prefix}_1",
        yaxis_title=f"{axis_prefix}_2",
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Radar chart per segment
# ---------------------------------------------------------------------------


def plot_radar(
    profiles: list[SegmentProfile],
    summary: dict[int, dict[str, float]],
    features: list[str],
) -> go.Figure:
    """Create a radar (spider) chart comparing feature means across segments.

    Each segment is a polygon on the radar.  Feature values are
    min-max normalized across clusters for fair visual comparison.

    Args:
        profiles: List of ``SegmentProfile`` objects.
        summary: Cluster summary with raw feature means.
        features: Feature names to include on the radar axes.

    Returns:
        Plotly Figure.
    """
    # Collect values per feature across non-noise clusters
    non_noise = {k: v for k, v in summary.items() if k != -1}
    if not non_noise:
        return go.Figure().update_layout(title="No clusters to display")

    # Min-max normalize each feature across clusters
    normalized: dict[int, list[float]] = {}
    for cid in non_noise:
        normalized[cid] = []

    for feat in features:
        values = [non_noise[cid].get(feat, 0.0) for cid in non_noise]
        vmin, vmax = min(values), max(values)
        span = vmax - vmin if vmax != vmin else 1.0
        for i, cid in enumerate(non_noise):
            normalized[cid].append((values[i] - vmin) / span)

    label_map = {p.cluster_id: p.name for p in profiles}

    fig = go.Figure()

    for cid in sorted(non_noise.keys()):
        name = label_map.get(cid, f"Cluster {cid}")
        color = _color_for_cluster(cid, label_map)
        vals = normalized[cid] + [normalized[cid][0]]  # close the polygon
        cats = features + [features[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            name=name,
            line=dict(color=color),
            opacity=0.6,
        ))

    fig.update_layout(
        title="Segment Profiles — Radar Chart",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# 5. Box plots per cluster
# ---------------------------------------------------------------------------


def plot_feature_boxplots(
    df: pd.DataFrame,
    feature: str,
    labels: np.ndarray,
    label_map: dict[int, str],
) -> go.Figure:
    """Create box plots of a feature, grouped by cluster.

    Args:
        df: Original (unscaled) DataFrame with the feature column.
        feature: Column name to plot.
        labels: Cluster labels.
        label_map: ``{cluster_id: segment_name}``.

    Returns:
        Plotly Figure.
    """
    plot_df = df[[feature]].copy()
    plot_df["segment"] = [label_map.get(lbl, f"Cluster {lbl}") for lbl in labels]

    fig = px.box(
        plot_df,
        x="segment",
        y=feature,
        color="segment",
        title=f"Distribution of {feature} by Segment",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, xaxis_title="Segment")

    return fig


# ---------------------------------------------------------------------------
# 6. Segment size bar chart
# ---------------------------------------------------------------------------


def plot_segment_sizes(
    profiles: list[SegmentProfile],
) -> go.Figure:
    """Create a horizontal bar chart of segment population sizes.

    Args:
        profiles: List of ``SegmentProfile`` objects.

    Returns:
        Plotly Figure.
    """
    label_map = {p.cluster_id: p.name for p in profiles}
    names = [p.name for p in profiles]
    sizes = [p.size for p in profiles]
    colors = [_color_for_cluster(p.cluster_id, label_map) for p in profiles]

    fig = go.Figure(go.Bar(
        x=sizes,
        y=names,
        orientation="h",
        marker_color=colors,
        text=sizes,
        textposition="auto",
    ))

    fig.update_layout(
        title="Segment Sizes",
        xaxis_title="Number of Transactions",
        yaxis_title="",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )

    return fig


# ---------------------------------------------------------------------------
# 7. Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
) -> go.Figure:
    """Create a correlation heatmap for selected features.

    Args:
        df: DataFrame containing the features (unscaled).
        features: Column names to include.

    Returns:
        Plotly Figure.
    """
    available = [f for f in features if f in df.columns]
    corr = df[available].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_white",
        width=600,
        height=600,
    )

    return fig
