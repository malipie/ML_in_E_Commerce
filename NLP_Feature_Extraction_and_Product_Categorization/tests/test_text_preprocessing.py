"""Tests for text preprocessing module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.text_preprocessing import (
    lemmatize_tokens,
    normalize_text,
    preprocess_dataframe,
    preprocess_name,
    remove_stopwords,
    tokenize,
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_uppercases_text(self) -> None:
        assert normalize_text("botki skórzane") == "BOTKI SKÓRZANE"

    def test_strips_whitespace(self) -> None:
        assert normalize_text("  BOTKI  ") == "BOTKI"

    def test_removes_special_chars(self) -> None:
        result = normalize_text("Sneakersy, z muzycznym motywem!")
        assert "," not in result
        assert "!" not in result

    def test_preserves_polish_diacritics(self) -> None:
        result = normalize_text("SKÓRZANE ZAMSZOWE AŻUROWE")
        assert "SKÓRZANE" in result
        assert "AŻUROWE" in result

    def test_preserves_hyphens(self) -> None:
        result = normalize_text("EKO-SKÓRA")
        assert "-" in result

    def test_preserves_digits(self) -> None:
        result = normalize_text("BOTKI 41")
        assert "41" in result

    def test_collapses_multiple_spaces(self) -> None:
        result = normalize_text("BOTKI   SKÓRZANE   41")
        assert result == "BOTKI SKÓRZANE 41"

    def test_none_returns_empty(self) -> None:
        assert normalize_text(None) == ""

    def test_empty_returns_empty(self) -> None:
        assert normalize_text("") == ""

    def test_nan_returns_empty(self) -> None:
        assert normalize_text(float("nan")) == ""


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_splits_on_whitespace(self) -> None:
        assert tokenize("BOTKI SKÓRZANE 41") == ["BOTKI", "SKÓRZANE", "41"]

    def test_min_length_filters(self) -> None:
        result = tokenize("A BOTKI W SZPIC", min_length=2)
        assert "A" not in result
        assert "W" not in result
        assert "BOTKI" in result

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_min_length_one(self) -> None:
        result = tokenize("A B CD", min_length=1)
        assert result == ["A", "B", "CD"]

    def test_default_min_length_is_two(self) -> None:
        result = tokenize("A BOTKI")
        assert result == ["BOTKI"]


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------


class TestRemoveStopwords:
    def test_removes_default_stopwords(self) -> None:
        tokens = ["BOTKI", "NA", "ZAMEK", "W", "SZPIC"]
        result = remove_stopwords(tokens)
        assert "NA" not in result
        assert "W" not in result
        assert "BOTKI" in result
        assert "ZAMEK" in result

    def test_custom_stopwords(self) -> None:
        tokens = ["BOTKI", "DAMSKIE", "CZARNE"]
        result = remove_stopwords(tokens, stopwords={"DAMSKIE"})
        assert result == ["BOTKI", "CZARNE"]

    def test_empty_tokens(self) -> None:
        assert remove_stopwords([]) == []

    def test_all_stopwords(self) -> None:
        tokens = ["NA", "W", "Z", "DO"]
        assert remove_stopwords(tokens) == []


# ---------------------------------------------------------------------------
# lemmatize_tokens
# ---------------------------------------------------------------------------


class TestLemmatizeTokens:
    def test_no_model_returns_original(self) -> None:
        tokens = ["BOTKI", "SKÓRZANE"]
        assert lemmatize_tokens(tokens, nlp=None) == tokens

    def test_empty_tokens(self) -> None:
        assert lemmatize_tokens([], nlp=None) == []

    def test_with_spacy_model(self) -> None:
        try:
            import spacy
            nlp = spacy.load("pl_core_news_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy Polish model not available")

        tokens = ["CZARNE", "BOTKI", "SKÓRZANE"]
        result = lemmatize_tokens(tokens, nlp=nlp)
        assert len(result) > 0
        assert all(isinstance(t, str) for t in result)
        assert all(t == t.upper() for t in result)


# ---------------------------------------------------------------------------
# preprocess_name
# ---------------------------------------------------------------------------


class TestPreprocessName:
    def test_full_pipeline(self) -> None:
        name = "TAUPE KLASYCZNE KOWBOJKI ZA KOSTKĘ NA ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41"
        result = preprocess_name(name)
        assert "TAUPE" in result
        assert "KOWBOJKI" in result
        assert "NA" not in result.split()
        assert "ZA" not in result.split()

    def test_none_input(self) -> None:
        assert preprocess_name(None) == ""

    def test_mixed_case_input(self) -> None:
        result = preprocess_name("Sneakersy z muzycznym motywem - 37")
        assert "SNEAKERSY" in result
        assert "37" in result

    def test_preserves_content_words(self) -> None:
        result = preprocess_name("CZARNY MAŁY PORTFEL MONNARI")
        assert "CZARNY" in result
        assert "PORTFEL" in result
        assert "MONNARI" in result


# ---------------------------------------------------------------------------
# preprocess_dataframe
# ---------------------------------------------------------------------------


class TestPreprocessDataframe:
    def test_adds_name_clean_column(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI SKÓRZANE 38"]})
        result = preprocess_dataframe(df)
        assert "name_clean" in result.columns

    def test_adds_name_tokens_column(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI SKÓRZANE 38"]})
        result = preprocess_dataframe(df)
        assert "name_tokens" in result.columns
        assert isinstance(result["name_tokens"].iloc[0], list)

    def test_does_not_modify_original(self) -> None:
        df = pd.DataFrame({"name": ["BOTKI"]})
        result = preprocess_dataframe(df)
        assert "name_clean" not in df.columns
        assert "name_clean" in result.columns

    def test_handles_none_names(self) -> None:
        df = pd.DataFrame({"name": [None, "BOTKI"]})
        result = preprocess_dataframe(df)
        assert result["name_clean"].iloc[0] == ""
        assert result["name_tokens"].iloc[0] == []

    def test_tokens_match_clean(self) -> None:
        df = pd.DataFrame({"name": ["CZARNE BOTKI NA ZAMEK 39"]})
        result = preprocess_dataframe(df)
        clean = result["name_clean"].iloc[0]
        tokens = result["name_tokens"].iloc[0]
        assert " ".join(tokens) == clean
