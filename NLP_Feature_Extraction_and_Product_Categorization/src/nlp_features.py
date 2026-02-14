"""NLP feature engineering for product categorization.

Builds numerical feature representations from product text data:
TF-IDF vectors, text statistics, and one-hot encoded extracted features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF-IDF features
# ---------------------------------------------------------------------------


def compute_tfidf_features(
    texts: pd.Series,
    max_features: int = 500,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on product names.

    Args:
        texts: Series of preprocessed product name strings.
        max_features: Maximum number of TF-IDF features.
        ngram_range: N-gram range for TF-IDF.
        min_df: Minimum document frequency.

    Returns:
        Tuple of (tfidf_matrix as dense array, fitted TfidfVectorizer).
    """
    texts_clean = texts.fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    tfidf_matrix = vectorizer.fit_transform(texts_clean)

    if issparse(tfidf_matrix):
        tfidf_dense = tfidf_matrix.toarray()
    else:
        tfidf_dense = np.asarray(tfidf_matrix)

    logger.info(
        "TF-IDF computed: %d documents × %d features.",
        tfidf_dense.shape[0],
        tfidf_dense.shape[1],
    )
    return tfidf_dense, vectorizer


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------


def compute_text_statistics(
    df: pd.DataFrame, name_col: str = "name"
) -> pd.DataFrame:
    """Add text-level statistical features to DataFrame.

    Adds columns: name_length, name_word_count, name_has_size,
    name_uppercase_ratio, name_digit_count.

    Args:
        df: Input DataFrame.
        name_col: Column with raw product names.

    Returns:
        DataFrame with added statistical columns.
    """
    df = df.copy()
    names = df[name_col].fillna("").astype(str)

    df["name_length"] = names.str.len()
    df["name_word_count"] = names.str.split().str.len().fillna(0).astype(int)
    df["name_has_size"] = names.str.contains(
        r"\b(?:3[5-9]|4[0-9]|50)\b", regex=True
    ).astype(int)
    df["name_uppercase_ratio"] = names.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df["name_digit_count"] = names.str.count(r"\d")

    logger.info("Text statistics computed for %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# One-hot encoding of extracted features
# ---------------------------------------------------------------------------


def encode_extracted_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode the rule-based extracted features.

    Args:
        df: DataFrame with extracted feature columns.
        feature_cols: Columns to encode. Defaults to standard extracted features.

    Returns:
        DataFrame with one-hot encoded columns replacing originals.
    """
    if feature_cols is None:
        feature_cols = ["color", "material", "product_type", "brand", "season"]

    df = df.copy()
    cols_to_encode = [c for c in feature_cols if c in df.columns]

    if not cols_to_encode:
        return df

    for col in cols_to_encode:
        df[col] = df[col].fillna("unknown")

    encoded = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode)
    df = pd.concat([df.drop(columns=cols_to_encode), encoded], axis=1)

    logger.info(
        "One-hot encoded %d columns → %d binary features.",
        len(cols_to_encode),
        len(encoded.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    tfidf_matrix: np.ndarray,
    vectorizer: TfidfVectorizer,
    include_stats: bool = True,
    include_encoded: bool = True,
) -> pd.DataFrame:
    """Combine TF-IDF, text stats, and encoded features into one matrix.

    Args:
        df: DataFrame with text stats and extracted features.
        tfidf_matrix: Dense TF-IDF array.
        vectorizer: Fitted TfidfVectorizer (for feature names).
        include_stats: Whether to include text statistics.
        include_encoded: Whether to include one-hot encoded features.

    Returns:
        Clean feature DataFrame ready for classification.
    """
    tfidf_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_names, index=df.index)

    parts = [tfidf_df]

    if include_stats:
        stat_cols = [
            c for c in ["name_length", "name_word_count", "name_has_size",
                        "name_uppercase_ratio", "name_digit_count"]
            if c in df.columns
        ]
        if stat_cols:
            parts.append(df[stat_cols].reset_index(drop=True))

    if include_encoded:
        encoded_cols = [c for c in df.columns if any(
            c.startswith(p) for p in
            ["color_", "material_", "product_type_", "brand_", "season_"]
        )]
        if encoded_cols:
            parts.append(df[encoded_cols].reset_index(drop=True))

    feature_matrix = pd.concat(parts, axis=1)
    feature_matrix = feature_matrix.fillna(0)

    logger.info("Feature matrix built: %d × %d", *feature_matrix.shape)
    return feature_matrix
