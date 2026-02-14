"""Tests for NLP feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.nlp_features import (
    build_feature_matrix,
    compute_text_statistics,
    compute_tfidf_features,
    encode_extracted_features,
)


# ---------------------------------------------------------------------------
# compute_tfidf_features
# ---------------------------------------------------------------------------


class TestTfidfFeatures:
    def test_correct_shape(self) -> None:
        texts = pd.Series(["CZARNE BOTKI", "BIAŁE SZPILKI", "PORTFEL SKÓRZANY"])
        matrix, vec = compute_tfidf_features(texts, max_features=10, min_df=1)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 10

    def test_non_negative_values(self) -> None:
        texts = pd.Series(["CZARNE BOTKI", "BIAŁE SZPILKI"])
        matrix, _ = compute_tfidf_features(texts, max_features=10, min_df=1)
        assert (matrix >= 0).all()

    def test_returns_dense_array(self) -> None:
        texts = pd.Series(["CZARNE BOTKI", "BIAŁE SZPILKI"])
        matrix, _ = compute_tfidf_features(texts, max_features=10, min_df=1)
        assert isinstance(matrix, np.ndarray)

    def test_vectorizer_fitted(self) -> None:
        texts = pd.Series(["CZARNE BOTKI", "BIAŁE SZPILKI"])
        _, vec = compute_tfidf_features(texts, max_features=10, min_df=1)
        assert len(vec.get_feature_names_out()) > 0

    def test_handles_empty_texts(self) -> None:
        texts = pd.Series(["BOTKI", "", "SZPILKI"])
        matrix, _ = compute_tfidf_features(texts, max_features=10, min_df=1)
        assert matrix.shape[0] == 3

    def test_handles_nan(self) -> None:
        texts = pd.Series(["BOTKI", None, "SZPILKI"])
        matrix, _ = compute_tfidf_features(texts, max_features=10, min_df=1)
        assert matrix.shape[0] == 3


# ---------------------------------------------------------------------------
# compute_text_statistics
# ---------------------------------------------------------------------------


class TestTextStatistics:
    def test_adds_columns(self) -> None:
        df = pd.DataFrame({"name": ["CZARNE BOTKI 38"]})
        result = compute_text_statistics(df)
        for col in ["name_length", "name_word_count", "name_has_size",
                     "name_uppercase_ratio", "name_digit_count"]:
            assert col in result.columns

    def test_correct_length(self) -> None:
        df = pd.DataFrame({"name": ["ABC"]})
        result = compute_text_statistics(df)
        assert result["name_length"].iloc[0] == 3

    def test_correct_word_count(self) -> None:
        df = pd.DataFrame({"name": ["CZARNE BOTKI SKÓRZANE"]})
        result = compute_text_statistics(df)
        assert result["name_word_count"].iloc[0] == 3

    def test_has_size_true(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI 38"]})
        result = compute_text_statistics(df)
        assert result["name_has_size"].iloc[0] == 1

    def test_has_size_false(self) -> None:
        df = pd.DataFrame({"name": ["PORTFEL SKÓRZANY"]})
        result = compute_text_statistics(df)
        assert result["name_has_size"].iloc[0] == 0

    def test_digit_count(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI 38 MODEL 123"]})
        result = compute_text_statistics(df)
        assert result["name_digit_count"].iloc[0] == 5

    def test_handles_none(self) -> None:
        df = pd.DataFrame({"name": [None]})
        result = compute_text_statistics(df)
        assert result["name_length"].iloc[0] == 0

    def test_does_not_modify_original(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI"]})
        result = compute_text_statistics(df)
        assert "name_length" not in df.columns
        assert "name_length" in result.columns


# ---------------------------------------------------------------------------
# encode_extracted_features
# ---------------------------------------------------------------------------


class TestEncodeExtractedFeatures:
    def test_produces_one_hot_columns(self) -> None:
        df = pd.DataFrame({
            "color": ["czarny", "bialy"],
            "material": ["skorzany", None],
        })
        result = encode_extracted_features(df, feature_cols=["color", "material"])
        assert any(c.startswith("color_") for c in result.columns)
        assert any(c.startswith("material_") for c in result.columns)

    def test_handles_none_as_unknown(self) -> None:
        df = pd.DataFrame({
            "color": [None, "czarny"],
        })
        result = encode_extracted_features(df, feature_cols=["color"])
        assert "color_unknown" in result.columns

    def test_removes_original_columns(self) -> None:
        df = pd.DataFrame({
            "color": ["czarny"],
            "other": [1],
        })
        result = encode_extracted_features(df, feature_cols=["color"])
        assert "color" not in result.columns
        assert "other" in result.columns

    def test_missing_feature_cols_skipped(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI"]})
        result = encode_extracted_features(df, feature_cols=["color"])
        assert "name" in result.columns


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def _make_vec(self, docs: list[str]) -> TfidfVectorizer:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        vec.fit(docs)
        return vec

    def test_correct_shape(self) -> None:
        texts = pd.Series(["BOTKI CZARNE", "SZPILKI BIAŁE"])
        tfidf, vec = compute_tfidf_features(texts, max_features=10, min_df=1)
        n_tfidf = tfidf.shape[1]
        df = pd.DataFrame({
            "name_length": [10, 20],
            "name_word_count": [3, 5],
            "name_has_size": [1, 0],
            "name_uppercase_ratio": [0.8, 0.9],
            "name_digit_count": [2, 0],
        })
        result = build_feature_matrix(df, tfidf, vec, include_encoded=False)
        assert result.shape[0] == 2
        assert result.shape[1] == n_tfidf + 5  # tfidf + stats

    def test_no_nan(self) -> None:
        df = pd.DataFrame({
            "name_length": [10],
            "name_word_count": [3],
        })
        tfidf = np.array([[0.5, 0.3]])
        vec = self._make_vec(["BOTKI CZARNE"])
        result = build_feature_matrix(df, tfidf, vec, include_encoded=False)
        assert not result.isna().any().any()

    def test_tfidf_prefix(self) -> None:
        df = pd.DataFrame({"name_length": [10]})
        tfidf = np.array([[0.5]])
        vec = self._make_vec(["BOTKI"])
        result = build_feature_matrix(df, tfidf, vec, include_encoded=False)
        assert any(c.startswith("tfidf_") for c in result.columns)
