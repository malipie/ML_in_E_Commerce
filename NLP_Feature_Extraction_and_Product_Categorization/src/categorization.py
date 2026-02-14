"""ML-based product categorization using TF-IDF + classifiers.

Trains and evaluates multi-class classifiers (LinearSVC, RandomForest, XGBoost)
to predict product type from preprocessed product names.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Holds results from one model evaluation."""

    model_name: str
    model: Any
    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]
    conf_matrix: np.ndarray
    class_names: list[str]
    cv_scores: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_labeled_data(
    df: pd.DataFrame,
    label_col: str = "product_type",
    min_class_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into labeled and unlabeled portions.

    Filters out classes with fewer than min_class_size samples.

    Args:
        df: Input DataFrame with optional label column.
        label_col: Column with product type labels.
        min_class_size: Minimum samples per class to keep.

    Returns:
        Tuple of (labeled_df, unlabeled_df).
    """
    if label_col not in df.columns:
        return pd.DataFrame(), df.copy()

    labeled_mask = df[label_col].notna()
    df_labeled = df[labeled_mask].copy()
    df_unlabeled = df[~labeled_mask].copy()

    class_counts = df_labeled[label_col].value_counts()
    valid_classes = class_counts[class_counts >= min_class_size].index
    small_classes = class_counts[class_counts < min_class_size]

    if len(small_classes) > 0:
        logger.info(
            "Filtering %d classes with < %d samples: %s",
            len(small_classes),
            min_class_size,
            list(small_classes.index),
        )
        too_small_mask = df_labeled[label_col].isin(small_classes.index)
        df_unlabeled = pd.concat([df_unlabeled, df_labeled[too_small_mask]])
        df_labeled = df_labeled[df_labeled[label_col].isin(valid_classes)]

    logger.info(
        "Labeled: %d samples, %d classes. Unlabeled: %d samples.",
        len(df_labeled),
        df_labeled[label_col].nunique(),
        len(df_unlabeled),
    )
    return df_labeled, df_unlabeled


# ---------------------------------------------------------------------------
# XGBoost wrapper (handles string labels via LabelEncoder)
# ---------------------------------------------------------------------------


class _XGBWithLabelEncoder(BaseEstimator, ClassifierMixin):
    """Wraps XGBClassifier to handle string labels transparently."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42,
        eval_metric: str = "mlogloss",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.eval_metric = eval_metric

    def fit(self, X: Any, y: Any) -> _XGBWithLabelEncoder:
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y)
        self._xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
        )
        self._xgb.fit(X, y_encoded)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        y_encoded = self._xgb.predict(X)
        return self._le.inverse_transform(y_encoded)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _build_pipeline(
    model_type: str, config: dict[str, Any] | None = None
) -> Pipeline:
    """Build sklearn Pipeline(TfidfVectorizer, Classifier).

    Args:
        model_type: One of 'svm', 'random_forest', 'xgboost'.
        config: Optional model-specific config dict.

    Returns:
        Unfitted sklearn Pipeline.
    """
    tfidf_config = (config or {}).get("tfidf", {})
    model_config = (config or {}).get("models", {}).get(model_type, {})

    ngram_range = tuple(tfidf_config.get("ngram_range", [1, 2]))

    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.get("max_features", 500),
        ngram_range=ngram_range,
        min_df=tfidf_config.get("min_df", 2),
    )

    if model_type == "svm":
        classifier = LinearSVC(
            C=model_config.get("C", 1.0),
            max_iter=5000,
            dual="auto",
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 100),
            random_state=model_config.get("random_state", 42),
        )
    elif model_type == "xgboost":
        classifier = _XGBWithLabelEncoder(
            n_estimators=model_config.get("n_estimators", 100),
            max_depth=model_config.get("max_depth", 6),
            random_state=model_config.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_tfidf_classifier(
    X_texts: pd.Series,
    y_labels: pd.Series,
    model_type: str = "svm",
    config: dict[str, Any] | None = None,
    cv_folds: int = 5,
) -> tuple[Pipeline, ClassificationResult]:
    """Build, cross-validate, and fit a TF-IDF + classifier pipeline.

    Args:
        X_texts: Series of preprocessed product name strings.
        y_labels: Series of product type labels.
        model_type: One of 'svm', 'random_forest', 'xgboost'.
        config: Classification config dict.
        cv_folds: Number of cross-validation folds.

    Returns:
        Tuple of (fitted Pipeline, ClassificationResult).
    """
    pipeline = _build_pipeline(model_type, config)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_texts, y_labels, cv=cv, scoring="f1_macro"
    )

    pipeline.fit(X_texts, y_labels)
    y_pred = pipeline.predict(X_texts)

    class_names = sorted(y_labels.unique().tolist())
    f1_per_class = {}
    for cls in class_names:
        cls_mask = y_labels == cls
        if cls_mask.sum() > 0:
            f1_per_class[cls] = float(f1_score(
                y_labels == cls, y_pred == cls, zero_division=0
            ))

    result = ClassificationResult(
        model_name=model_type,
        model=pipeline,
        accuracy=float(accuracy_score(y_labels, y_pred)),
        f1_macro=float(f1_score(y_labels, y_pred, average="macro", zero_division=0)),
        f1_per_class=f1_per_class,
        conf_matrix=confusion_matrix(y_labels, y_pred, labels=class_names),
        class_names=class_names,
        cv_scores=cv_scores,
    )

    logger.info(
        "Model '%s': CV F1=%.3f (±%.3f), Train Acc=%.3f, Train F1=%.3f",
        model_type,
        cv_scores.mean(),
        cv_scores.std(),
        result.accuracy,
        result.f1_macro,
    )
    return pipeline, result


# ---------------------------------------------------------------------------
# Multi-model evaluation
# ---------------------------------------------------------------------------


def evaluate_models(
    X_texts: pd.Series,
    y_labels: pd.Series,
    config: dict[str, Any] | None = None,
) -> list[ClassificationResult]:
    """Train and evaluate SVM, RandomForest, and XGBoost.

    Args:
        X_texts: Series of preprocessed text.
        y_labels: Series of labels.
        config: Classification config dict.

    Returns:
        List of ClassificationResult sorted by CV F1-macro descending.
    """
    cv_folds = (config or {}).get("cv_folds", 5)
    results: list[ClassificationResult] = []

    for model_type in ["svm", "random_forest", "xgboost"]:
        _, result = train_tfidf_classifier(
            X_texts, y_labels, model_type=model_type,
            config=config, cv_folds=cv_folds,
        )
        results.append(result)

    results.sort(key=lambda r: r.cv_scores.mean(), reverse=True)
    logger.info("Best model: %s (CV F1=%.3f)", results[0].model_name,
                results[0].cv_scores.mean())
    return results


def select_champion(results: list[ClassificationResult]) -> ClassificationResult:
    """Select the best model by CV F1-macro score.

    Args:
        results: List of ClassificationResult.

    Returns:
        The champion model result.
    """
    return max(results, key=lambda r: r.cv_scores.mean())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_unlabeled(
    pipeline: Pipeline,
    df_unlabeled: pd.DataFrame,
    name_col: str = "name_clean",
) -> pd.DataFrame:
    """Predict product_type for unlabeled items.

    Args:
        pipeline: Fitted TF-IDF + classifier pipeline.
        df_unlabeled: DataFrame with unlabeled items.
        name_col: Column with preprocessed text.

    Returns:
        DataFrame with predicted product_type column.
    """
    df = df_unlabeled.copy()
    if df.empty:
        return df

    texts = df[name_col].fillna("").astype(str)
    df["product_type"] = pipeline.predict(texts)

    logger.info("Predicted %d unlabeled items.", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Serialize fitted pipeline to disk.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: Output file path.

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: str | Path) -> Pipeline:
    """Load a serialized pipeline from disk.

    Args:
        path: Path to pickle file.

    Returns:
        Fitted sklearn Pipeline.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301
