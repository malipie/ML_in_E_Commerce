"""Tests for ML categorization module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.categorization import (
    ClassificationResult,
    evaluate_models,
    load_model,
    predict_unlabeled,
    prepare_labeled_data,
    save_model,
    select_champion,
    train_tfidf_classifier,
)


# ---------------------------------------------------------------------------
# prepare_labeled_data
# ---------------------------------------------------------------------------


class TestPrepareLabeledData:
    def test_splits_correctly(self, sample_labeled_df: pd.DataFrame) -> None:
        labeled, unlabeled = prepare_labeled_data(sample_labeled_df, min_class_size=3)
        assert len(labeled) == 21
        assert len(unlabeled) == 0

    def test_filters_small_classes(self) -> None:
        df = pd.DataFrame({
            "name_clean": ["A", "B", "C", "D", "E"],
            "product_type": ["botki", "botki", "botki", "rare", "rare"],
        })
        labeled, unlabeled = prepare_labeled_data(df, min_class_size=3)
        assert "rare" not in labeled["product_type"].values
        assert len(unlabeled) == 2

    def test_handles_all_none_labels(self) -> None:
        df = pd.DataFrame({
            "name_clean": ["A", "B"],
            "product_type": [None, None],
        })
        labeled, unlabeled = prepare_labeled_data(df)
        assert len(labeled) == 0
        assert len(unlabeled) == 2

    def test_missing_label_col(self) -> None:
        df = pd.DataFrame({"name_clean": ["A", "B"]})
        labeled, unlabeled = prepare_labeled_data(df)
        assert len(labeled) == 0
        assert len(unlabeled) == 2


# ---------------------------------------------------------------------------
# train_tfidf_classifier
# ---------------------------------------------------------------------------


class TestTrainClassifier:
    def test_svm_fits(self, sample_labeled_df: pd.DataFrame) -> None:
        pipeline, result = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        assert result.model_name == "svm"
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1_macro <= 1

    def test_random_forest_fits(self, sample_labeled_df: pd.DataFrame) -> None:
        _, result = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="random_forest",
            cv_folds=3,
        )
        assert result.model_name == "random_forest"
        assert len(result.cv_scores) == 3

    def test_xgboost_fits(self, sample_labeled_df: pd.DataFrame) -> None:
        _, result = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="xgboost",
            cv_folds=3,
        )
        assert result.model_name == "xgboost"
        assert result.conf_matrix.shape[0] == len(result.class_names)

    def test_invalid_model_type(self, sample_labeled_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            train_tfidf_classifier(
                sample_labeled_df["name_clean"],
                sample_labeled_df["product_type"],
                model_type="invalid",
            )

    def test_cv_scores_shape(self, sample_labeled_df: pd.DataFrame) -> None:
        _, result = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        assert len(result.cv_scores) == 3

    def test_f1_per_class_has_all_classes(self, sample_labeled_df: pd.DataFrame) -> None:
        _, result = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        for cls in sample_labeled_df["product_type"].unique():
            assert cls in result.f1_per_class


# ---------------------------------------------------------------------------
# evaluate_models
# ---------------------------------------------------------------------------


class TestEvaluateModels:
    def test_returns_three_results(self, sample_labeled_df: pd.DataFrame) -> None:
        results = evaluate_models(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            config={"cv_folds": 3},
        )
        assert len(results) == 3

    def test_sorted_by_f1(self, sample_labeled_df: pd.DataFrame) -> None:
        results = evaluate_models(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            config={"cv_folds": 3},
        )
        scores = [r.cv_scores.mean() for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_model_types_present(self, sample_labeled_df: pd.DataFrame) -> None:
        results = evaluate_models(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            config={"cv_folds": 3},
        )
        names = {r.model_name for r in results}
        assert names == {"svm", "random_forest", "xgboost"}


# ---------------------------------------------------------------------------
# select_champion
# ---------------------------------------------------------------------------


class TestSelectChampion:
    def test_picks_highest_f1(self) -> None:
        results = [
            ClassificationResult("a", None, 0.8, 0.7, {}, np.array([]), [],
                                 cv_scores=np.array([0.6, 0.7])),
            ClassificationResult("b", None, 0.9, 0.85, {}, np.array([]), [],
                                 cv_scores=np.array([0.8, 0.9])),
            ClassificationResult("c", None, 0.85, 0.8, {}, np.array([]), [],
                                 cv_scores=np.array([0.7, 0.8])),
        ]
        champion = select_champion(results)
        assert champion.model_name == "b"


# ---------------------------------------------------------------------------
# predict_unlabeled
# ---------------------------------------------------------------------------


class TestPredictUnlabeled:
    def test_predicts_correct_shape(self, sample_labeled_df: pd.DataFrame) -> None:
        pipeline, _ = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        unlabeled = pd.DataFrame({
            "name_clean": ["CZARNE BOTKI 38", "BIAŁE SZPILKI 39"],
        })
        result = predict_unlabeled(pipeline, unlabeled)
        assert len(result) == 2
        assert "product_type" in result.columns

    def test_valid_labels(self, sample_labeled_df: pd.DataFrame) -> None:
        pipeline, _ = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        unlabeled = pd.DataFrame({"name_clean": ["CZARNE BOTKI"]})
        result = predict_unlabeled(pipeline, unlabeled)
        valid_classes = set(sample_labeled_df["product_type"].unique())
        assert result["product_type"].iloc[0] in valid_classes

    def test_empty_dataframe(self, sample_labeled_df: pd.DataFrame) -> None:
        pipeline, _ = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        result = predict_unlabeled(pipeline, pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


class TestModelPersistence:
    def test_save_creates_file(self, sample_labeled_df: pd.DataFrame,
                                tmp_path: Path) -> None:
        pipeline, _ = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        path = save_model(pipeline, tmp_path / "model.pkl")
        assert path.exists()

    def test_load_roundtrip(self, sample_labeled_df: pd.DataFrame,
                            tmp_path: Path) -> None:
        pipeline, _ = train_tfidf_classifier(
            sample_labeled_df["name_clean"],
            sample_labeled_df["product_type"],
            model_type="svm",
            cv_folds=3,
        )
        path = save_model(pipeline, tmp_path / "model.pkl")
        loaded = load_model(path)
        pred_orig = pipeline.predict(["CZARNE BOTKI"])
        pred_loaded = loaded.predict(["CZARNE BOTKI"])
        assert pred_orig[0] == pred_loaded[0]
