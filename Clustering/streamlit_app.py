"""Customer Segmentation Engine — Streamlit Dashboard.

Launch with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
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
from src.visualization import (
    plot_correlation_heatmap,
    plot_elbow,
    plot_feature_boxplots,
    plot_pca_3d,
    plot_radar,
    plot_scatter_2d,
    plot_segment_sizes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
SEGMENTED_PATH = Path("data/processed/orders_segmented.parquet")
RAW_PATH = Path("data/processed/orders_features.parquet")

NUMERICAL_FEATURES: list[str] = []  # loaded from config.yaml at startup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data
def load_config() -> dict:
    """Load YAML configuration (cached)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_segmented_data() -> pd.DataFrame | None:
    """Load pre-computed segmented data if available."""
    if SEGMENTED_PATH.exists():
        return pd.read_parquet(SEGMENTED_PATH)
    return None


@st.cache_data
def run_full_pipeline(
    k_override: int | None,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> tuple[pd.DataFrame, dict, list, dict]:
    """Run the full pipeline with custom parameters (cached).

    Returns:
        Tuple of (enriched_df, elbow_dict, profiles, cluster_summary).
    """
    config = load_config()
    data_cfg = config["data"]
    feat_cfg = config["features"]

    # ETL
    if RAW_PATH.exists():
        df = pd.read_parquet(RAW_PATH)
    else:
        df = run_etl(
            raw_dir=data_cfg["raw_dir"],
            output_path=data_cfg["output_parquet"],
            pattern=data_cfg.get("xml_glob", "*.xml"),
        )

    # Feature Engineering
    pipeline = FeaturePipeline.from_config(feat_cfg)
    X_scaled = pipeline.fit_transform(df)

    # Elbow analysis
    km_cfg = config["clustering"]["kmeans"]
    k_min, k_max = km_cfg["k_range"]
    elbow = evaluate_kmeans_range(
        X_scaled, k_min=k_min, k_max=k_max,
        n_init=km_cfg.get("n_init", 10),
        random_state=km_cfg.get("random_state", 42),
    )

    # K-Means
    chosen_k = k_override if k_override else elbow.best_k_silhouette
    _, km_labels = fit_kmeans(
        X_scaled, n_clusters=chosen_k,
        n_init=km_cfg.get("n_init", 10),
        random_state=km_cfg.get("random_state", 42),
    )

    # DBSCAN
    dbscan_result = fit_dbscan(X_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples)

    # PCA
    pca_cfg = config["dimensionality"]["pca"]
    pca_result = fit_pca(X_scaled, n_components=pca_cfg.get("n_components", 3))

    # Labeling
    available_raw = [c for c in NUMERICAL_FEATURES if c in df.columns]
    raw_matrix = df[available_raw].values
    cluster_summary = compute_cluster_summary(raw_matrix, km_labels, available_raw)
    profiles = label_clusters(cluster_summary)
    label_map = build_label_map(profiles)

    # Enrich
    df = df.copy()
    df["cluster_id"] = km_labels
    df["segment"] = df["cluster_id"].map(label_map)
    df["dbscan_label"] = dbscan_result.labels
    df["is_noise"] = dbscan_result.labels == -1
    df["pca_1"] = pca_result.embedding[:, 0]
    df["pca_2"] = pca_result.embedding[:, 1]
    df["pca_3"] = pca_result.embedding[:, 2]

    elbow_dict = {
        "k_values": elbow.k_values,
        "inertias": elbow.inertias,
        "silhouette_scores": elbow.silhouette_scores,
        "best_k": elbow.best_k_silhouette,
    }

    return df, elbow_dict, profiles, cluster_summary


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Engine",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Segmentation Engine")
st.markdown("Transaction-level clustering of e-commerce orders.")

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Pipeline Parameters")

config = load_config()
NUMERICAL_FEATURES = config["features"]["numerical"]
km_cfg = config["clustering"]["kmeans"]
db_cfg = config["clustering"]["dbscan"]
k_min, k_max = km_cfg["k_range"]

use_auto_k = st.sidebar.checkbox("Auto-select k (best silhouette)", value=True)
k_override = None
if not use_auto_k:
    k_override = st.sidebar.slider("Number of clusters (k)", k_min, k_max, value=4)

dbscan_eps = st.sidebar.slider(
    "DBSCAN eps", 0.1, 5.0, value=float(db_cfg.get("eps", 0.5)), step=0.1
)
dbscan_min_samples = st.sidebar.slider(
    "DBSCAN min_samples", 2, 20, value=int(db_cfg.get("min_samples", 5))
)

run_btn = st.sidebar.button("Run Pipeline", type="primary")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Try loading pre-computed data first
df = load_segmented_data()
elbow_dict = None
profiles = None
cluster_summary = None

if run_btn:
    with st.spinner("Running segmentation pipeline..."):
        df, elbow_dict, profiles, cluster_summary = run_full_pipeline(
            k_override=k_override,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
    st.success("Pipeline complete!")

if df is None:
    st.info(
        "No segmented data found. Click **Run Pipeline** in the sidebar, "
        "or run `python main.py` first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI metrics
# ---------------------------------------------------------------------------

st.header("Overview")

n_segments = df["segment"].nunique()
n_orders = len(df)
n_noise = int(df["is_noise"].sum()) if "is_noise" in df.columns else 0
avg_order = df["order_amount_brutto"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", f"{n_orders:,}")
col2.metric("Segments", n_segments)
col3.metric("DBSCAN Noise", f"{n_noise:,} ({100*n_noise/n_orders:.1f}%)")
col4.metric("Avg Order Value", f"{avg_order:.2f} PLN")

# ---------------------------------------------------------------------------
# Segment sizes
# ---------------------------------------------------------------------------

st.header("Segment Distribution")

if profiles:
    st.plotly_chart(plot_segment_sizes(profiles), width="stretch")
else:
    # Build profiles from existing data
    seg_counts = df["segment"].value_counts()
    st.bar_chart(seg_counts)

# ---------------------------------------------------------------------------
# Segment details table
# ---------------------------------------------------------------------------

if profiles:
    st.subheader("Segment Profiles")
    profile_data = []
    for p in profiles:
        profile_data.append({
            "Segment": p.name,
            "Size": p.size,
            "% of Total": f"{100 * p.size / n_orders:.1f}%",
            "Monetary Rank": f"{p.monetary_rank:.2f}" if p.monetary_rank >= 0 else "N/A",
            "Basket Rank": f"{p.basket_size_rank:.2f}" if p.basket_size_rank >= 0 else "N/A",
            "Description": p.description,
        })
    st.dataframe(pd.DataFrame(profile_data), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Elbow + Silhouette
# ---------------------------------------------------------------------------

if elbow_dict:
    st.header("Elbow & Silhouette Analysis")
    fig_elbow = plot_elbow(
        elbow_dict["k_values"],
        elbow_dict["inertias"],
        elbow_dict["silhouette_scores"],
        best_k=elbow_dict["best_k"],
    )
    st.plotly_chart(fig_elbow, width="stretch")

# ---------------------------------------------------------------------------
# 3D PCA Scatter
# ---------------------------------------------------------------------------

st.header("3D Cluster Visualization (PCA)")

if all(col in df.columns for col in ["pca_1", "pca_2", "pca_3"]):
    label_map = dict(zip(df["cluster_id"], df["segment"]))
    embedding_3d = df[["pca_1", "pca_2", "pca_3"]].values
    labels = df["cluster_id"].values

    fig_3d = plot_pca_3d(embedding_3d, labels, label_map)
    st.plotly_chart(fig_3d, width="stretch")

    # 2D version
    with st.expander("2D PCA View"):
        fig_2d = plot_scatter_2d(
            embedding_3d[:, :2], labels, label_map,
            title="2D PCA — Cluster Visualization",
            axis_prefix="PC",
        )
        st.plotly_chart(fig_2d, width="stretch")

# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

if profiles and cluster_summary:
    st.header("Segment Profiles — Radar Chart")
    fig_radar = plot_radar(profiles, cluster_summary, NUMERICAL_FEATURES)
    st.plotly_chart(fig_radar, width="stretch")

# ---------------------------------------------------------------------------
# Feature distributions
# ---------------------------------------------------------------------------

st.header("Feature Distributions by Segment")

label_map = dict(zip(df["cluster_id"], df["segment"]))
selected_feature = st.selectbox(
    "Select feature:",
    NUMERICAL_FEATURES,
    index=0,
)

fig_box = plot_feature_boxplots(df, selected_feature, df["cluster_id"].values, label_map)
st.plotly_chart(fig_box, width="stretch")

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

with st.expander("Feature Correlation Matrix"):
    available_feats = [f for f in NUMERICAL_FEATURES if f in df.columns]
    fig_corr = plot_correlation_heatmap(df, available_feats)
    st.plotly_chart(fig_corr, width="stretch")

# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------

st.header("Data Explorer")

segment_filter = st.multiselect(
    "Filter by segment:",
    options=sorted(df["segment"].dropna().unique()),
    default=sorted(df["segment"].dropna().unique()),
)

show_noise_only = st.checkbox("Show DBSCAN noise only", value=False)

filtered = df[df["segment"].isin(segment_filter)]
if show_noise_only and "is_noise" in filtered.columns:
    filtered = filtered[filtered["is_noise"]]

display_cols = [
    "order_id", "date_add", "segment", "order_amount_brutto",
    "n_items", "avg_item_price", "client_city", "delivery_type",
    "payment_name", "is_noise",
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].sort_values("order_amount_brutto", ascending=False),
    width="stretch",
    hide_index=True,
    height=400,
)

st.caption(f"Showing {len(filtered):,} of {n_orders:,} transactions.")
