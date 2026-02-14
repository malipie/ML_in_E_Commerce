"""Text preprocessing module for Polish product names.

Provides normalization, tokenization, stopword removal, and optional
spaCy-based lemmatization for Polish e-commerce product descriptions.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polish stopwords (common prepositions/conjunctions in product names)
# ---------------------------------------------------------------------------

POLISH_STOPWORDS: set[str] = {
    "NA", "W", "Z", "ZE", "DO", "ZA", "I", "DLA", "PRZY", "OD",
    "PO", "BEZ", "PRZED", "MIĘDZY", "PRZEZ", "O", "POD", "ORAZ",
    "A", "AN", "THE", "OF", "AND",  # English stopwords in mixed text
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str | None) -> str:
    """Normalize text: uppercase, strip whitespace, remove special chars.

    Keeps letters (including Polish diacritics), digits, spaces, and hyphens.

    Args:
        text: Raw text string or None.

    Returns:
        Normalized uppercase string. Empty string if input is None.
    """
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).upper().strip()
    text = re.sub(r"[^A-ZĄĆĘŁŃÓŚŹŻA-Z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Split text into tokens, filtering by minimum length.

    Args:
        text: Normalized text string.
        min_length: Minimum token length to keep.

    Returns:
        List of token strings.
    """
    if not text:
        return []
    tokens = text.split()
    return [t for t in tokens if len(t) >= min_length]


# ---------------------------------------------------------------------------
# Stopword removal
# ---------------------------------------------------------------------------


def remove_stopwords(
    tokens: list[str], stopwords: set[str] | None = None
) -> list[str]:
    """Remove Polish stopwords from token list.

    Args:
        tokens: List of uppercase token strings.
        stopwords: Custom stopword set. Defaults to POLISH_STOPWORDS.

    Returns:
        Filtered token list.
    """
    if stopwords is None:
        stopwords = POLISH_STOPWORDS
    return [t for t in tokens if t not in stopwords]


# ---------------------------------------------------------------------------
# Lemmatization (optional, requires spaCy)
# ---------------------------------------------------------------------------


def _load_spacy_model(model_name: str = "pl_core_news_sm") -> Any:
    """Load spaCy Polish model. Returns None if unavailable."""
    try:
        import spacy
        return spacy.load(model_name)
    except (ImportError, OSError):
        logger.warning("spaCy model '%s' not available. Skipping lemmatization.",
                       model_name)
        return None


def lemmatize_tokens(tokens: list[str], nlp: Any = None) -> list[str]:
    """Lemmatize tokens using spaCy Polish model.

    Falls back to identity (returns original tokens) if model is unavailable.

    Args:
        tokens: List of token strings.
        nlp: Pre-loaded spaCy Language model or None.

    Returns:
        List of lemmatized tokens (uppercase).
    """
    if nlp is None or not tokens:
        return tokens

    text = " ".join(tokens)
    doc = nlp(text)
    return [token.lemma_.upper() for token in doc if len(token.lemma_) >= 2]


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_name(
    text: str | None,
    nlp: Any = None,
    use_lemma: bool = False,
    min_length: int = 2,
) -> str:
    """Full preprocessing pipeline: normalize → tokenize → stopwords → (lemma) → rejoin.

    Args:
        text: Raw product name.
        nlp: Pre-loaded spaCy model (only used if use_lemma=True).
        use_lemma: Whether to apply lemmatization.
        min_length: Minimum token length.

    Returns:
        Preprocessed text string.
    """
    normalized = normalize_text(text)
    tokens = tokenize(normalized, min_length=min_length)
    tokens = remove_stopwords(tokens)
    if use_lemma and nlp is not None:
        tokens = lemmatize_tokens(tokens, nlp)
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    name_col: str = "name",
    use_lemma: bool = False,
    min_length: int = 2,
) -> pd.DataFrame:
    """Add preprocessed text columns to DataFrame.

    Adds ``name_clean`` (preprocessed string) and ``name_tokens`` (token list).

    Args:
        df: Input DataFrame with product names.
        name_col: Column containing raw product names.
        use_lemma: Whether to apply lemmatization.
        min_length: Minimum token length.

    Returns:
        DataFrame with added columns.
    """
    df = df.copy()

    nlp = None
    if use_lemma:
        nlp = _load_spacy_model()
        if nlp is None:
            logger.warning("Lemmatization requested but model unavailable. Proceeding without.")

    df["name_clean"] = df[name_col].apply(
        lambda x: preprocess_name(x, nlp=nlp, use_lemma=use_lemma, min_length=min_length)
    )
    df["name_tokens"] = df["name_clean"].apply(lambda x: x.split() if x else [])

    logger.info("Text preprocessing complete: %d rows processed.", len(df))
    return df
