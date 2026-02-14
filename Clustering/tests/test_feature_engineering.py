"""Unit tests for src/feature_engineering.py — transformations & pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    CYCLICAL_PERIODS,
    FeaturePipeline,
    apply_log_transform,
    encode_all_cyclical,
    encode_categorical,
    encode_cyclical,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking ETL output for feature engineering."""
    return pd.DataFrame({
        "customer_id": ["a", "b", "c", "d", "e"],
        "order_id": ["1", "2", "3", "4", "5"],
        "date_add": pd.to_datetime([
            "2025-06-15 14:30:00",
            "2025-06-16 09:00:00",
            "2025-06-17 22:15:00",
            "2025-06-18 03:45:00",
            "2025-06-19 12:00:00",
        ]),
        "order_amount_brutto": [150.0, 300.0, 50.0, 1200.0, 0.0],
        "n_items": [2, 1, 3, 1, 1],
        "avg_item_price": [75.0, 300.0, 16.67, 1200.0, 0.0],
        "max_item_price": [100.0, 300.0, 25.0, 1200.0, 0.0],
        "total_quantity": [3, 1, 5, 1, 1],
        "hour_of_day": [14, 9, 22, 3, 12],
        "day_of_week": [6, 0, 2, 3, 4],
        "delivery_type": [
            "Paczkomaty InPost", "Kurier DPD", "Paczkomaty InPost",
            "Kurier DPD", "Poczta Polska",
        ],
        "payment_name": [
            "Allegro Finance", "Przelew", "Allegro Finance",
            "Allegro Finance", "Przelew",
        ],
    })


@pytest.fixture
def default_pipeline() -> FeaturePipeline:
    """Pipeline configured with defaults matching config.yaml."""
    return FeaturePipeline(
        numerical_cols=[
            "order_amount_brutto", "n_items", "avg_item_price", "max_item_price",
        ],
        log_transform_cols=["order_amount_brutto", "avg_item_price", "max_item_price"],
        cyclical_config=dict(CYCLICAL_PERIODS),
        categorical_cols=["delivery_type", "payment_name"],
    )


# ===================================================================
# Log transform
# ===================================================================


class TestApplyLogTransform:
    """Tests for apply_log_transform()."""

    def test_log1p_correctness(self) -> None:
        df = pd.DataFrame({"val": [0.0, 1.0, 100.0]})
        result = apply_log_transform(df, ["val"])
        expected = np.log1p([0.0, 1.0, 100.0])
        np.testing.assert_array_almost_equal(result["val"].values, expected)

    def test_zero_safe(self) -> None:
        df = pd.DataFrame({"val": [0.0]})
        result = apply_log_transform(df, ["val"])
        assert result["val"].iloc[0] == 0.0  # log1p(0) = 0

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"val": [100.0]})
        _ = apply_log_transform(df, ["val"])
        assert df["val"].iloc[0] == 100.0  # original unchanged

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"val": [1.0]})
        result = apply_log_transform(df, ["nonexistent"])
        assert "val" in result.columns
        assert result["val"].iloc[0] == 1.0

    def test_multiple_columns(self) -> None:
        df = pd.DataFrame({"a": [10.0], "b": [20.0], "c": [30.0]})
        result = apply_log_transform(df, ["a", "c"])
        assert result["a"].iloc[0] == pytest.approx(np.log1p(10.0))
        assert result["b"].iloc[0] == 20.0  # untouched
        assert result["c"].iloc[0] == pytest.approx(np.log1p(30.0))


# ===================================================================
# Cyclical encoding
# ===================================================================


class TestEncodeCyclical:
    """Tests for encode_cyclical() and encode_all_cyclical()."""

    def test_sin_cos_columns_created(self) -> None:
        df = pd.DataFrame({"hour_of_day": [0, 6, 12, 18]})
        result = encode_cyclical(df, "hour_of_day", 24)
        assert "hour_of_day_sin" in result.columns
        assert "hour_of_day_cos" in result.columns
        assert "hour_of_day" not in result.columns

    def test_midnight_and_noon(self) -> None:
        df = pd.DataFrame({"hour_of_day": [0, 12]})
        result = encode_cyclical(df, "hour_of_day", 24)
        # At hour 0: sin=0, cos=1
        assert result["hour_of_day_sin"].iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert result["hour_of_day_cos"].iloc[0] == pytest.approx(1.0, abs=1e-10)
        # At hour 12: sin≈0, cos=-1
        assert result["hour_of_day_sin"].iloc[1] == pytest.approx(0.0, abs=1e-10)
        assert result["hour_of_day_cos"].iloc[1] == pytest.approx(-1.0, abs=1e-10)

    def test_six_am(self) -> None:
        df = pd.DataFrame({"hour_of_day": [6]})
        result = encode_cyclical(df, "hour_of_day", 24)
        # At hour 6 (quarter period): sin=1, cos≈0
        assert result["hour_of_day_sin"].iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert result["hour_of_day_cos"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_day_of_week_period_7(self) -> None:
        df = pd.DataFrame({"day_of_week": [0]})  # Monday
        result = encode_cyclical(df, "day_of_week", 7)
        assert result["day_of_week_sin"].iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert result["day_of_week_cos"].iloc[0] == pytest.approx(1.0, abs=1e-10)

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"hour_of_day": [12]})
        _ = encode_cyclical(df, "hour_of_day", 24)
        assert "hour_of_day" in df.columns  # original untouched

    def test_missing_column_returns_unchanged(self) -> None:
        df = pd.DataFrame({"other": [1]})
        result = encode_cyclical(df, "nonexistent", 24)
        assert list(result.columns) == ["other"]

    def test_encode_all_cyclical(self) -> None:
        df = pd.DataFrame({"hour_of_day": [12], "day_of_week": [3]})
        result = encode_all_cyclical(df)
        assert "hour_of_day_sin" in result.columns
        assert "hour_of_day_cos" in result.columns
        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
        assert "hour_of_day" not in result.columns
        assert "day_of_week" not in result.columns

    def test_values_bounded_minus1_to_1(self) -> None:
        df = pd.DataFrame({"hour_of_day": list(range(24))})
        result = encode_cyclical(df, "hour_of_day", 24)
        assert result["hour_of_day_sin"].between(-1, 1).all()
        assert result["hour_of_day_cos"].between(-1, 1).all()


# ===================================================================
# Categorical encoding
# ===================================================================


class TestEncodeCategorical:
    """Tests for encode_categorical()."""

    def test_basic_one_hot(self) -> None:
        df = pd.DataFrame({"color": ["red", "blue", "red"]})
        result, cats = encode_categorical(df, ["color"])
        assert "color_blue" in result.columns
        assert "color_red" in result.columns
        assert "color" not in result.columns

    def test_categories_learned(self) -> None:
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        _, cats = encode_categorical(df, ["color"])
        assert "color" in cats
        assert len(cats["color"]) == 3

    def test_transform_mode_unseen_category(self) -> None:
        df_train = pd.DataFrame({"color": ["red", "blue"]})
        _, cats = encode_categorical(df_train, ["color"])

        df_new = pd.DataFrame({"color": ["purple"]})
        result, _ = encode_categorical(df_new, ["color"], categories_map=cats)
        # purple is unseen → all zeros for known categories
        assert result["color_blue"].iloc[0] == 0.0
        assert result["color_red"].iloc[0] == 0.0

    def test_transform_mode_preserves_known_columns(self) -> None:
        df_train = pd.DataFrame({"color": ["red", "blue", "green"]})
        _, cats = encode_categorical(df_train, ["color"])

        df_new = pd.DataFrame({"color": ["red"]})
        result, _ = encode_categorical(df_new, ["color"], categories_map=cats)
        assert set(cats["color"]).issubset(set(result.columns))

    def test_dtype_is_float(self) -> None:
        df = pd.DataFrame({"color": ["red", "blue"]})
        result, _ = encode_categorical(df, ["color"])
        for col in result.columns:
            assert result[col].dtype == float

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"other": [1]})
        result, cats = encode_categorical(df, ["nonexistent"])
        assert list(result.columns) == ["other"]
        assert cats == {}

    def test_multiple_columns(self) -> None:
        df = pd.DataFrame({
            "color": ["red", "blue"],
            "size": ["S", "L"],
        })
        result, cats = encode_categorical(df, ["color", "size"])
        assert "color_red" in result.columns
        assert "size_S" in result.columns
        assert "color" in cats
        assert "size" in cats


# ===================================================================
# FeaturePipeline
# ===================================================================


class TestFeaturePipeline:
    """Tests for the FeaturePipeline class."""

    def test_fit_transform_returns_numpy(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        result = default_pipeline.fit_transform(raw_df)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_fit_transform_shape(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        result = default_pipeline.fit_transform(raw_df)
        assert result.shape[0] == len(raw_df)
        assert result.shape[1] == len(default_pipeline.feature_names)

    def test_feature_names_populated_after_fit(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        default_pipeline.fit(raw_df)
        assert len(default_pipeline.feature_names) > 0

    def test_feature_names_contain_expected(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        default_pipeline.fit(raw_df)
        names = default_pipeline.feature_names
        # Numerical (after log)
        assert "order_amount_brutto" in names
        assert "n_items" in names
        # Cyclical sin/cos
        assert "hour_of_day_sin" in names
        assert "hour_of_day_cos" in names
        assert "day_of_week_sin" in names
        assert "day_of_week_cos" in names
        # Original cyclical columns should NOT be present
        assert "hour_of_day" not in names
        assert "day_of_week" not in names

    def test_scaler_fitted(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        default_pipeline.fit(raw_df)
        assert default_pipeline.scaler is not None
        assert hasattr(default_pipeline.scaler, "mean_")

    def test_scaled_output_zero_mean(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        result = default_pipeline.fit_transform(raw_df)
        means = result.mean(axis=0)
        np.testing.assert_array_almost_equal(means, 0.0, decimal=10)

    def test_scaled_output_unit_variance(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        result = default_pipeline.fit_transform(raw_df)
        stds = result.std(axis=0, ddof=0)
        # Columns with zero variance (constant) will have std=0 after scaling
        non_constant = stds > 1e-10
        np.testing.assert_array_almost_equal(stds[non_constant], 1.0, decimal=10)

    def test_transform_without_fit_raises(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            default_pipeline.transform(raw_df)

    def test_transform_consistency(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        result1 = default_pipeline.fit_transform(raw_df)
        result2 = default_pipeline.transform(raw_df)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_categories_map_populated(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        default_pipeline.fit(raw_df)
        assert "delivery_type" in default_pipeline.categories_map
        assert "payment_name" in default_pipeline.categories_map

    def test_get_feature_dataframe(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        scaled = default_pipeline.fit_transform(raw_df)
        df_result = default_pipeline.get_feature_dataframe(scaled)
        assert isinstance(df_result, pd.DataFrame)
        assert list(df_result.columns) == default_pipeline.feature_names
        assert len(df_result) == len(raw_df)

    def test_get_feature_dataframe_wrong_shape_raises(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        default_pipeline.fit(raw_df)
        bad_array = np.zeros((5, 2))
        with pytest.raises(ValueError, match="Expected"):
            default_pipeline.get_feature_dataframe(bad_array)

    def test_get_feature_dataframe_unfitted_raises(
        self, default_pipeline: FeaturePipeline
    ) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            default_pipeline.get_feature_dataframe(np.zeros((1, 1)))

    def test_from_config(self) -> None:
        config = {
            "numerical": ["order_amount_brutto", "n_items"],
            "cyclical": ["hour_of_day"],
            "categorical": ["delivery_type"],
            "log_transform": ["order_amount_brutto"],
        }
        pipe = FeaturePipeline.from_config(config)
        assert pipe.numerical_cols == ["order_amount_brutto", "n_items"]
        assert pipe.log_transform_cols == ["order_amount_brutto"]
        assert "hour_of_day" in pipe.cyclical_config
        assert pipe.categorical_cols == ["delivery_type"]

    def test_log_transform_applied(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        """Verify that log-transformed values differ from raw values."""
        default_pipeline.fit(raw_df)
        transformed = default_pipeline._select_and_transform(raw_df, fit=False)

        # order_amount_brutto should be log-transformed
        raw_vals = raw_df["order_amount_brutto"].values
        trans_vals = transformed["order_amount_brutto"].values
        # For non-zero values, log1p(x) < x for x > 0
        mask = raw_vals > 1.0
        assert (trans_vals[mask] < raw_vals[mask]).all()

    def test_no_nan_in_output(
        self, raw_df: pd.DataFrame, default_pipeline: FeaturePipeline
    ) -> None:
        result = default_pipeline.fit_transform(raw_df)
        assert not np.isnan(result).any()
