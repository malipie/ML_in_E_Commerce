"""Shared test fixtures for the Clustering test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sample XML
# ---------------------------------------------------------------------------

SAMPLE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<orders>
<order>
  <client_id></client_id>
  <order_id>100001</order_id>
  <date_add>15.06.2025 14:30:00</date_add>
  <payment_name>Allegro Finance</payment_name>
  <payment_status>1</payment_status>
  <payment_paid>150.00</payment_paid>
  <delivery_type>Allegro Paczkomaty InPost</delivery_type>
  <delivery_price>0.00</delivery_price>
  <buyer_comments></buyer_comments>
  <seller_comments></seller_comments>
  <client_city>Warszawa</client_city>
  <order_amount_brutto>150.00</order_amount_brutto>
  <rows>
    <row>
      <name>Product A</name>
      <products_id>1001</products_id>
      <products_ean></products_ean>
      <products_sku>SKU-001</products_sku>
      <auction_id>9000001</auction_id>
      <item_price_brutto>100.00</item_price_brutto>
      <quantity>1</quantity>
      <amount_brutto>100.00</amount_brutto>
      <vat_rate>23.00</vat_rate>
      <symkar>1001</symkar>
    </row>
    <row>
      <name>Product B</name>
      <products_id>1002</products_id>
      <products_ean></products_ean>
      <products_sku>SKU-002</products_sku>
      <auction_id>9000002</auction_id>
      <item_price_brutto>50.00</item_price_brutto>
      <quantity>2</quantity>
      <amount_brutto>100.00</amount_brutto>
      <vat_rate>23.00</vat_rate>
      <symkar>1002</symkar>
    </row>
  </rows>
</order>
<order>
  <client_id></client_id>
  <order_id>100002</order_id>
  <date_add>20.06.2025 09:15:22</date_add>
  <payment_name>Przelew online</payment_name>
  <payment_status>1</payment_status>
  <payment_paid>300.50</payment_paid>
  <delivery_type>Kurier DPD</delivery_type>
  <delivery_price>12.99</delivery_price>
  <buyer_comments>Proszę o fakturę</buyer_comments>
  <seller_comments></seller_comments>
  <client_city>Kraków</client_city>
  <order_amount_brutto>300.50</order_amount_brutto>
  <rows>
    <row>
      <name>Product C</name>
      <products_id>2001</products_id>
      <products_ean>5901234567890</products_ean>
      <products_sku>SKU-003</products_sku>
      <auction_id>9000003</auction_id>
      <item_price_brutto>300.50</item_price_brutto>
      <quantity>1</quantity>
      <amount_brutto>300.50</amount_brutto>
      <vat_rate>23.00</vat_rate>
      <symkar>2001</symkar>
    </row>
  </rows>
</order>
<order>
  <client_id></client_id>
  <order_id>100003</order_id>
  <date_add>25.06.2025 22:05:10</date_add>
  <payment_name>Płatność przy odbiorze</payment_name>
  <payment_status>0</payment_status>
  <payment_paid>0.00</payment_paid>
  <delivery_type>Allegro Paczkomaty InPost</delivery_type>
  <delivery_price>0.00</delivery_price>
  <buyer_comments></buyer_comments>
  <seller_comments></seller_comments>
  <client_city>Gdańsk</client_city>
  <order_amount_brutto>89.99</order_amount_brutto>
  <rows>
    <row>
      <name>Product D</name>
      <products_id>3001</products_id>
      <products_ean></products_ean>
      <products_sku>SKU-004</products_sku>
      <auction_id>9000004</auction_id>
      <item_price_brutto>89.99</item_price_brutto>
      <quantity>1</quantity>
      <amount_brutto>89.99</amount_brutto>
      <vat_rate>23.00</vat_rate>
      <symkar>3001</symkar>
    </row>
  </rows>
</order>
</orders>
"""

SAMPLE_XML_EMPTY = """\
<?xml version="1.0" encoding="utf-8"?>
<orders>
</orders>
"""

SAMPLE_XML_MISSING_FIELDS = """\
<?xml version="1.0" encoding="utf-8"?>
<orders>
<order>
  <order_id>999999</order_id>
  <date_add>01.01.2025 00:00:00</date_add>
  <payment_name></payment_name>
  <payment_status>0</payment_status>
  <payment_paid></payment_paid>
  <delivery_type></delivery_type>
  <delivery_price></delivery_price>
  <client_city></client_city>
  <order_amount_brutto></order_amount_brutto>
  <rows>
  </rows>
</order>
</orders>
"""


@pytest.fixture
def sample_xml_path(tmp_path: Path) -> Path:
    """Write sample XML to a temp file and return its path."""
    xml_file = tmp_path / "test_orders.xml"
    xml_file.write_text(SAMPLE_XML, encoding="utf-8")
    return xml_file


@pytest.fixture
def sample_xml_dir(tmp_path: Path) -> Path:
    """Create a temp directory with two XML files."""
    xml_dir = tmp_path / "xml_data"
    xml_dir.mkdir()
    (xml_dir / "part1.xml").write_text(SAMPLE_XML, encoding="utf-8")
    (xml_dir / "part2.xml").write_text(SAMPLE_XML, encoding="utf-8")
    return xml_dir


@pytest.fixture
def empty_xml_path(tmp_path: Path) -> Path:
    """Write an XML file with no orders."""
    xml_file = tmp_path / "empty.xml"
    xml_file.write_text(SAMPLE_XML_EMPTY, encoding="utf-8")
    return xml_file


@pytest.fixture
def missing_fields_xml_path(tmp_path: Path) -> Path:
    """Write an XML file with missing/empty fields."""
    xml_file = tmp_path / "missing.xml"
    xml_file.write_text(SAMPLE_XML_MISSING_FIELDS, encoding="utf-8")
    return xml_file


@pytest.fixture
def sample_orders_df() -> pd.DataFrame:
    """Pre-built DataFrame matching the SAMPLE_XML fixture (3 orders)."""
    return pd.DataFrame({
        "customer_id": ["cust_001", "cust_002", "cust_003"],
        "order_id": ["100001", "100002", "100003"],
        "date_add": pd.to_datetime(
            ["2025-06-15 14:30:00", "2025-06-20 09:15:22", "2025-06-25 22:05:10"]
        ),
        "order_amount_brutto": [150.00, 300.50, 89.99],
        "n_items": [2, 1, 1],
        "avg_item_price": [75.00, 300.50, 89.99],
        "max_item_price": [100.00, 300.50, 89.99],
        "total_quantity": [3, 1, 1],
        "hour_of_day": [14, 9, 22],
        "day_of_week": [6, 4, 2],
        "delivery_type": [
            "Allegro Paczkomaty InPost",
            "Kurier DPD",
            "Allegro Paczkomaty InPost",
        ],
        "payment_name": [
            "Allegro Finance",
            "Przelew online",
            "Płatność przy odbiorze",
        ],
    })
