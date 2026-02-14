"""Tests for ETL module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from lxml import etree

from src.etl import (
    _clean_text,
    _generate_item_id,
    _parse_datetime,
    _parse_order_element,
    _safe_float,
    items_to_dataframe,
    parse_xml_directory,
    parse_xml_file,
    save_parquet,
)
from tests.conftest import SAMPLE_XML


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_strips_whitespace(self) -> None:
        assert _clean_text("  hello  ") == "hello"

    def test_returns_none_for_empty(self) -> None:
        assert _clean_text("") is None

    def test_returns_none_for_none(self) -> None:
        assert _clean_text(None) is None

    def test_returns_none_for_null_string(self) -> None:
        assert _clean_text("NULL") is None

    def test_returns_none_for_null_lowercase(self) -> None:
        assert _clean_text("null") is None

    def test_preserves_polish_chars(self) -> None:
        assert _clean_text("SKÓRZANE ZAMSZOWE") == "SKÓRZANE ZAMSZOWE"


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_valid_float(self) -> None:
        assert _safe_float("211.59") == 211.59

    def test_valid_int_string(self) -> None:
        assert _safe_float("100") == 100.0

    def test_none_returns_zero(self) -> None:
        assert _safe_float(None) == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert _safe_float("") == 0.0

    def test_non_numeric_returns_zero(self) -> None:
        assert _safe_float("abc") == 0.0


# ---------------------------------------------------------------------------
# _parse_datetime
# ---------------------------------------------------------------------------


class TestParseDatetime:
    def test_valid_date(self) -> None:
        result = _parse_datetime("15.01.2025 10:30:00")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_none_returns_none(self) -> None:
        assert _parse_datetime(None) is None

    def test_empty_returns_none(self) -> None:
        assert _parse_datetime("") is None

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_datetime("2025-01-15") is None


# ---------------------------------------------------------------------------
# _generate_item_id
# ---------------------------------------------------------------------------


class TestGenerateItemId:
    def test_produces_16_char_hex(self) -> None:
        result = _generate_item_id("100001", "5226", "15120470896")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        id1 = _generate_item_id("100001", "5226", "15120470896")
        id2 = _generate_item_id("100001", "5226", "15120470896")
        assert id1 == id2

    def test_different_inputs_different_ids(self) -> None:
        id1 = _generate_item_id("100001", "5226", "15120470896")
        id2 = _generate_item_id("100002", "5226", "15120470896")
        assert id1 != id2

    def test_handles_none_values(self) -> None:
        result = _generate_item_id(None, None, None)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# _parse_order_element
# ---------------------------------------------------------------------------


class TestParseOrderElement:
    def _get_orders(self) -> list[etree._Element]:
        root = etree.fromstring(SAMPLE_XML.encode("utf-8"))
        return root.findall("order")

    def test_single_item_order(self) -> None:
        orders = self._get_orders()
        items = _parse_order_element(orders[0])
        assert len(items) == 1
        assert items[0]["name"] == "TAUPE KLASYCZNE KOWBOJKI ZA KOSTKĘ NA ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41"
        assert items[0]["order_id"] == "100001"

    def test_multi_item_order(self) -> None:
        orders = self._get_orders()
        items = _parse_order_element(orders[1])
        assert len(items) == 2
        assert items[0]["name"] == "CZARNY MAŁY PORTFEL MONNARI PORTMONETKA SKÓRZANY LAKIEROWANY"
        assert items[1]["name"] == "CZARNE NUBUKOWE WYSOKIE BOTKI NA KLOCKU NA PLATFORMIE SERGIO LEONE 40"

    def test_order_fields_on_each_item(self) -> None:
        orders = self._get_orders()
        items = _parse_order_element(orders[1])
        for item in items:
            assert item["order_id"] == "100002"
            assert item["payment_name"] == "Przelewy24"
            assert item["delivery_type"] == "Allegro Kurier DPD"

    def test_item_fields_extracted(self) -> None:
        orders = self._get_orders()
        items = _parse_order_element(orders[0])
        item = items[0]
        assert item["products_id"] == "5226"
        assert item["products_sku"] == "OPT-611-OC-CAP-41"
        assert item["item_price_brutto"] == "211.59"
        assert item["quantity"] == "1"
        assert item["vat_rate"] == "23.00"

    def test_seller_comments_extracted(self) -> None:
        orders = self._get_orders()
        items = _parse_order_element(orders[0])
        assert items[0]["seller_comments"] == "reklamacja rozpruwa sie szew"

    def test_empty_comments_are_none(self) -> None:
        orders = self._get_orders()
        items = _parse_order_element(orders[1])
        assert items[0]["buyer_comments"] is None or items[0]["buyer_comments"] == ""


# ---------------------------------------------------------------------------
# parse_xml_file
# ---------------------------------------------------------------------------


class TestParseXmlFile:
    def test_parses_correct_item_count(self, sample_xml_file: Path) -> None:
        items = parse_xml_file(sample_xml_file)
        assert len(items) == 4  # 1 + 2 + 1

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_xml_file(tmp_path / "nonexistent.xml")

    def test_malformed_xml_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("<orders><order>", encoding="utf-8")
        with pytest.raises(etree.XMLSyntaxError):
            parse_xml_file(bad_file)

    def test_returns_list_of_dicts(self, sample_xml_file: Path) -> None:
        items = parse_xml_file(sample_xml_file)
        assert isinstance(items, list)
        assert all(isinstance(item, dict) for item in items)


# ---------------------------------------------------------------------------
# parse_xml_directory
# ---------------------------------------------------------------------------


class TestParseXmlDirectory:
    def test_parses_multiple_files(self, sample_xml_dir: Path) -> None:
        items = parse_xml_directory(sample_xml_dir)
        assert len(items) == 8  # 4 items * 2 files

    def test_directory_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_xml_directory(tmp_path / "nonexistent")

    def test_no_matching_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_xml_directory(tmp_path, pattern="*.csv")

    def test_pattern_filtering(self, sample_xml_dir: Path) -> None:
        items = parse_xml_directory(sample_xml_dir, pattern="1.xml")
        assert len(items) == 4


# ---------------------------------------------------------------------------
# items_to_dataframe
# ---------------------------------------------------------------------------


class TestItemsToDataframe:
    def _get_items(self, sample_xml_file: Path) -> list:
        return parse_xml_file(sample_xml_file)

    def test_correct_row_count(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert len(df) == 4

    def test_has_item_id_column(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert "item_id" in df.columns

    def test_item_id_unique(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert df["item_id"].nunique() == len(df)

    def test_date_parsed(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert pd.api.types.is_datetime64_any_dtype(df["date_add"])

    def test_numeric_columns(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert df["item_price_brutto"].dtype == float
        assert df["amount_brutto"].dtype == float
        assert df["vat_rate"].dtype == float

    def test_name_cleaned(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert df["name"].iloc[0] is not None
        assert isinstance(df["name"].iloc[0], str)

    def test_empty_comments_become_none(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        # Order 100002 has empty seller_comments
        order2_items = df[df["order_id"] == "100002"]
        assert order2_items["seller_comments"].isna().all()

    def test_non_empty_comments_preserved(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        order1 = df[df["order_id"] == "100001"]
        assert order1["seller_comments"].iloc[0] == "reklamacja rozpruwa sie szew"

    def test_column_ordering(self, sample_xml_file: Path) -> None:
        items = self._get_items(sample_xml_file)
        df = items_to_dataframe(items)
        assert list(df.columns[:4]) == ["item_id", "order_id", "name", "products_sku"]

    def test_empty_input_returns_empty_df(self) -> None:
        df = items_to_dataframe([])
        assert df.empty


# ---------------------------------------------------------------------------
# save_parquet
# ---------------------------------------------------------------------------


class TestSaveParquet:
    def test_creates_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        output = save_parquet(df, tmp_path / "test.parquet")
        assert output.exists()

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        output = save_parquet(df, tmp_path / "test.parquet")
        loaded = pd.read_parquet(output)
        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = save_parquet(
            pd.DataFrame({"a": [1]}),
            tmp_path / "sub" / "dir" / "test.parquet",
        )
        assert output.exists()
