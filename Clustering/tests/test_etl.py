"""Unit tests for src/etl.py — XML parsing, cleaning, aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.etl import (
    _aggregate_items,
    _generate_customer_id,
    _parse_datetime,
    _safe_float,
    orders_to_dataframe,
    parse_xml_directory,
    parse_xml_file,
    save_parquet,
)


# ===================================================================
# XML Parsing
# ===================================================================


class TestParseXmlFile:
    """Tests for parse_xml_file()."""

    def test_parses_correct_number_of_orders(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        assert len(orders) == 3

    def test_order_has_required_fields(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        order = orders[0]
        assert "order_id" in order
        assert "date_add" in order
        assert "order_amount_brutto" in order
        assert "items" in order

    def test_items_parsed_correctly(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        # First order has 2 items
        assert len(orders[0]["items"]) == 2
        # Second order has 1 item
        assert len(orders[1]["items"]) == 1

    def test_item_fields_extracted(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        item = orders[0]["items"][0]
        assert item["item_price_brutto"] == "100.00"
        assert item["quantity"] == "1"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_xml_file(tmp_path / "nonexistent.xml")

    def test_empty_xml_returns_empty_list(self, empty_xml_path: Path) -> None:
        orders = parse_xml_file(empty_xml_path)
        assert orders == []

    def test_missing_fields_handled(self, missing_fields_xml_path: Path) -> None:
        orders = parse_xml_file(missing_fields_xml_path)
        assert len(orders) == 1
        order = orders[0]
        assert order["order_id"] == "999999"
        # Empty tags should return empty string or None
        assert order["client_city"] is None or order["client_city"] == ""


class TestParseXmlDirectory:
    """Tests for parse_xml_directory()."""

    def test_parses_multiple_files(self, sample_xml_dir: Path) -> None:
        orders = parse_xml_directory(sample_xml_dir)
        # 2 files × 3 orders each = 6
        assert len(orders) == 6

    def test_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_xml_directory(tmp_path / "nonexistent_dir")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            parse_xml_directory(empty_dir)

    def test_pattern_filtering(self, sample_xml_dir: Path) -> None:
        # No .csv files in directory
        with pytest.raises(FileNotFoundError):
            parse_xml_directory(sample_xml_dir, pattern="*.csv")


# ===================================================================
# Aggregation helpers
# ===================================================================


class TestAggregateItems:
    """Tests for _aggregate_items()."""

    def test_single_item(self) -> None:
        items = [{"item_price_brutto": "99.99", "quantity": "1"}]
        result = _aggregate_items(items)
        assert result["n_items"] == 1
        assert result["avg_item_price"] == pytest.approx(99.99)
        assert result["max_item_price"] == pytest.approx(99.99)
        assert result["total_quantity"] == 1

    def test_multiple_items(self) -> None:
        items = [
            {"item_price_brutto": "100.00", "quantity": "1"},
            {"item_price_brutto": "50.00", "quantity": "2"},
        ]
        result = _aggregate_items(items)
        assert result["n_items"] == 2
        assert result["avg_item_price"] == pytest.approx(75.0)
        assert result["max_item_price"] == pytest.approx(100.0)
        assert result["total_quantity"] == 3

    def test_empty_items(self) -> None:
        result = _aggregate_items([])
        assert result["n_items"] == 0
        assert result["avg_item_price"] == 0.0
        assert result["max_item_price"] == 0.0
        assert result["total_quantity"] == 0

    def test_invalid_price_defaults_to_zero(self) -> None:
        items = [{"item_price_brutto": "invalid", "quantity": "1"}]
        result = _aggregate_items(items)
        assert result["avg_item_price"] == 0.0

    def test_missing_quantity_defaults_to_zero(self) -> None:
        items = [{"item_price_brutto": "50.00", "quantity": None}]
        result = _aggregate_items(items)
        assert result["total_quantity"] == 0


# ===================================================================
# Helper functions
# ===================================================================


class TestGenerateCustomerId:
    """Tests for _generate_customer_id()."""

    def test_deterministic(self) -> None:
        id1 = _generate_customer_id("12345")
        id2 = _generate_customer_id("12345")
        assert id1 == id2

    def test_different_inputs_different_ids(self) -> None:
        id1 = _generate_customer_id("12345")
        id2 = _generate_customer_id("67890")
        assert id1 != id2

    def test_returns_12_char_hex(self) -> None:
        result = _generate_customer_id("test")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)


class TestParseDatetime:
    """Tests for _parse_datetime()."""

    def test_valid_format(self) -> None:
        result = _parse_datetime("15.06.2025 14:30:00")
        assert result == pd.Timestamp("2025-06-15 14:30:00")

    def test_none_returns_none(self) -> None:
        assert _parse_datetime(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_datetime("") is None

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_datetime("not-a-date") is None


class TestSafeFloat:
    """Tests for _safe_float()."""

    def test_valid_string(self) -> None:
        assert _safe_float("123.45") == pytest.approx(123.45)

    def test_none_returns_zero(self) -> None:
        assert _safe_float(None) == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert _safe_float("") == 0.0

    def test_invalid_returns_zero(self) -> None:
        assert _safe_float("abc") == 0.0


# ===================================================================
# DataFrame construction
# ===================================================================


class TestOrdersToDataframe:
    """Tests for orders_to_dataframe()."""

    def test_correct_shape(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        assert len(df) == 3

    def test_has_required_columns(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        required = {
            "customer_id", "order_id", "date_add",
            "order_amount_brutto", "n_items", "avg_item_price",
            "max_item_price", "hour_of_day", "day_of_week",
        }
        assert required.issubset(set(df.columns))

    def test_date_parsed_as_datetime(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        assert pd.api.types.is_datetime64_any_dtype(df["date_add"])

    def test_numeric_columns_are_float(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        for col in ["order_amount_brutto", "avg_item_price", "max_item_price"]:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"

    def test_item_aggregation_values(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        # First order: 2 items, prices 100 and 50
        row = df[df["order_id"] == "100001"].iloc[0]
        assert row["n_items"] == 2
        assert row["avg_item_price"] == pytest.approx(75.0)
        assert row["max_item_price"] == pytest.approx(100.0)
        assert row["total_quantity"] == 3

    def test_hour_and_day_extracted(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        row = df[df["order_id"] == "100001"].iloc[0]
        assert row["hour_of_day"] == 14
        assert row["day_of_week"] == 6  # Sunday

    def test_customer_id_column_exists(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        assert df["customer_id"].notna().all()
        assert df["customer_id"].nunique() == 3

    def test_empty_orders_returns_empty_df(self) -> None:
        df = orders_to_dataframe([])
        assert df.empty

    def test_leading_column_order(self, sample_xml_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        assert list(df.columns[:3]) == ["customer_id", "order_id", "date_add"]


# ===================================================================
# Persistence
# ===================================================================


class TestSaveParquet:
    """Tests for save_parquet()."""

    def test_creates_file(self, sample_xml_path: Path, tmp_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        out = tmp_path / "output" / "test.parquet"
        result_path = save_parquet(df, out)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_roundtrip_preserves_data(self, sample_xml_path: Path, tmp_path: Path) -> None:
        orders = parse_xml_file(sample_xml_path)
        df = orders_to_dataframe(orders)
        out = tmp_path / "roundtrip.parquet"
        save_parquet(df, out)
        df_loaded = pd.read_parquet(out)
        assert len(df_loaded) == len(df)
        assert set(df_loaded.columns) == set(df.columns)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = tmp_path / "deep" / "nested" / "dir" / "file.parquet"
        save_parquet(df, out)
        assert out.exists()
