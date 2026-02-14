"""Shared test fixtures for NLP Feature Extraction project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Sample XML data
# ---------------------------------------------------------------------------

SAMPLE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<orders>
  <order>
    <order_id>100001</order_id>
    <date_add>15.01.2025 10:30:00</date_add>
    <payment_name>Allegro Finance</payment_name>
    <delivery_type>Allegro Paczkomaty InPost</delivery_type>
    <client_city>Warszawa</client_city>
    <order_amount_brutto>211.59</order_amount_brutto>
    <seller_comments>reklamacja rozpruwa sie szew</seller_comments>
    <buyer_comments></buyer_comments>
    <rows>
      <row>
        <name>TAUPE KLASYCZNE KOWBOJKI ZA KOSTKĘ NA ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41</name>
        <products_id>5226</products_id>
        <products_ean>5904528942505</products_ean>
        <products_sku>OPT-611-OC-CAP-41</products_sku>
        <auction_id>15120470896</auction_id>
        <item_price_brutto>211.59</item_price_brutto>
        <quantity>1</quantity>
        <amount_brutto>211.59</amount_brutto>
        <vat_rate>23.00</vat_rate>
        <symkar>5226</symkar>
      </row>
    </rows>
  </order>
  <order>
    <order_id>100002</order_id>
    <date_add>16.01.2025 14:20:00</date_add>
    <payment_name>Przelewy24</payment_name>
    <delivery_type>Allegro Kurier DPD</delivery_type>
    <client_city>Kraków</client_city>
    <order_amount_brutto>499.00</order_amount_brutto>
    <seller_comments></seller_comments>
    <buyer_comments></buyer_comments>
    <rows>
      <row>
        <name>CZARNY MAŁY PORTFEL MONNARI PORTMONETKA SKÓRZANY LAKIEROWANY</name>
        <products_id>8801</products_id>
        <products_ean></products_ean>
        <products_sku>MON-PUR0162-020-24</products_sku>
        <auction_id>15330001234</auction_id>
        <item_price_brutto>89.00</item_price_brutto>
        <quantity>1</quantity>
        <amount_brutto>89.00</amount_brutto>
        <vat_rate>23.00</vat_rate>
        <symkar>8801</symkar>
      </row>
      <row>
        <name>CZARNE NUBUKOWE WYSOKIE BOTKI NA KLOCKU NA PLATFORMIE SERGIO LEONE 40</name>
        <products_id>7102</products_id>
        <products_ean>5904528955123</products_ean>
        <products_sku>SL-TR928-CZARNY-NUB-40</products_sku>
        <auction_id>15330001235</auction_id>
        <item_price_brutto>410.00</item_price_brutto>
        <quantity>1</quantity>
        <amount_brutto>410.00</amount_brutto>
        <vat_rate>23.00</vat_rate>
        <symkar>7102</symkar>
      </row>
    </rows>
  </order>
  <order>
    <order_id>100003</order_id>
    <date_add>17.01.2025 09:15:30</date_add>
    <payment_name>PayU</payment_name>
    <delivery_type>Allegro One Box</delivery_type>
    <client_city>Gdańsk</client_city>
    <order_amount_brutto>159.99</order_amount_brutto>
    <seller_comments>stan zm.</seller_comments>
    <buyer_comments>proszę o szybką wysyłkę</buyer_comments>
    <rows>
      <row>
        <name>Sneakersy z muzycznym motywem - 37</name>
        <products_id>9903</products_id>
        <products_ean></products_ean>
        <products_sku>SNK23-060-37</products_sku>
        <auction_id>15440002345</auction_id>
        <item_price_brutto>159.99</item_price_brutto>
        <quantity>1</quantity>
        <amount_brutto>159.99</amount_brutto>
        <vat_rate>23.00</vat_rate>
        <symkar>9903</symkar>
      </row>
    </rows>
  </order>
</orders>
"""

SAMPLE_PRODUCT_NAMES: list[str] = [
    "TAUPE KLASYCZNE KOWBOJKI ZA KOSTKĘ NA ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41",
    "CZARNY MAŁY PORTFEL MONNARI PORTMONETKA SKÓRZANY LAKIEROWANY",
    "KAMELOWE BRĄZOWE BOTKI PÓŁBUTY WĘŻOWE CHOLEWKI 36",
    "Sneakersy z muzycznym motywem - 37",
    "Trapery, Glany w Czachy Czaszki - 38",
    "SREBRNE SZPILKI W SZPIC NISKIE SEASTAR POŁYSK 38",
    "CZARNE NUBUKOWE WYSOKIE BOTKI NA KLOCKU NA PLATFORMIE SERGIO LEONE 40",
    "STYLOWA TOREBKA DAMSKA CROSSBODY MONNARI POLIESTER",
    "TOREBKA DAMSKA MONNARI LISTONOSZKA MAŁA KLASYCZNA EKO SKÓRA",
    "PŁASKIE BOTKI KOWBOJKI DAMSKIE SKÓRZANE ZAMSZOWE AŻUROWE BRĄZOWE 40",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_xml_file(tmp_path: Path) -> Path:
    """Write SAMPLE_XML to a temp file and return its path."""
    xml_file = tmp_path / "test_orders.xml"
    xml_file.write_text(SAMPLE_XML, encoding="utf-8")
    return xml_file


@pytest.fixture
def sample_xml_dir(tmp_path: Path) -> Path:
    """Write SAMPLE_XML to two files in a temp directory."""
    for i in range(1, 3):
        xml_file = tmp_path / f"{i}.xml"
        xml_file.write_text(SAMPLE_XML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_items_df() -> pd.DataFrame:
    """Pre-built item-level DataFrame for testing."""
    return pd.DataFrame({
        "item_id": ["aaa111", "bbb222", "ccc333", "ddd444"],
        "order_id": ["100001", "100002", "100002", "100003"],
        "name": [
            "TAUPE KLASYCZNE KOWBOJKI ZA KOSTKĘ NA ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41",
            "CZARNY MAŁY PORTFEL MONNARI PORTMONETKA SKÓRZANY LAKIEROWANY",
            "CZARNE NUBUKOWE WYSOKIE BOTKI NA KLOCKU NA PLATFORMIE SERGIO LEONE 40",
            "Sneakersy z muzycznym motywem - 37",
        ],
        "products_sku": [
            "OPT-611-OC-CAP-41",
            "MON-PUR0162-020-24",
            "SL-TR928-CZARNY-NUB-40",
            "SNK23-060-37",
        ],
        "item_price_brutto": [211.59, 89.00, 410.00, 159.99],
        "quantity": [1, 1, 1, 1],
        "seller_comments": ["reklamacja rozpruwa sie szew", None, None, "stan zm."],
        "buyer_comments": [None, None, None, "proszę o szybką wysyłkę"],
    })


@pytest.fixture
def sample_extracted_df(sample_items_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with extracted features already applied."""
    df = sample_items_df.copy()
    df["name_clean"] = [
        "TAUPE KLASYCZNE KOWBOJKI KOSTKĘ ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41",
        "CZARNY MAŁY PORTFEL MONNARI PORTMONETKA SKÓRZANY LAKIEROWANY",
        "CZARNE NUBUKOWE WYSOKIE BOTKI KLOCKU PLATFORMIE SERGIO LEONE 40",
        "SNEAKERSY MUZYCZNYM MOTYWEM 37",
    ]
    df["color"] = ["taupe", "czarny", "czarny", None]
    df["material"] = ["zamszowy", "lakierowany", "nubukowy", None]
    df["size"] = [41, None, 40, 37]
    df["product_type"] = ["kowbojki", "portfel", "botki", "sneakersy"]
    df["brand"] = ["Optimo", "Monnari", "Sergio Leone", "Art"]
    df["season"] = ["warm", None, None, None]
    return df


@pytest.fixture
def sample_labeled_df() -> pd.DataFrame:
    """DataFrame suitable for classification (with product_type labels)."""
    names = [
        "CZARNE BOTKI SKÓRZANE 38",
        "BRĄZOWE BOTKI ZAMSZOWE 39",
        "CZARNE BOTKI NUBUKOWE 40",
        "CZERWONE SZPILKI LAKIEROWANE 37",
        "ZŁOTE SZPILKI ZAMSZOWE 38",
        "CZARNE SZPILKI SKÓRZANE 39",
        "BRĄZOWY PORTFEL SKÓRZANY",
        "CZARNY PORTFEL LAKIEROWANY",
        "SZARY PORTFEL ZAMSZOWY",
        "BIAŁE SNEAKERSY TEKSTYLNE 40",
        "CZARNE SNEAKERSY SKÓRZANE 41",
        "SZARE SNEAKERSY NUBUKOWE 42",
        "CZARNE KOZAKI SKÓRZANE 38",
        "BRĄZOWE KOZAKI ZAMSZOWE 39",
        "CZARNE KOZAKI NUBUKOWE 40",
        "BIAŁE TRAMPKI TEKSTYLNE 37",
        "CZARNE TRAMPKI SKÓRZANE 38",
        "SZARE TRAMPKI ZAMSZOWE 39",
        "BEŻOWE SANDAŁY SKÓRZANE 38",
        "ZŁOTE SANDAŁY ZAMSZOWE 37",
        "BRĄZOWE SANDAŁY NUBUKOWE 39",
    ]
    labels = [
        "botki", "botki", "botki",
        "szpilki", "szpilki", "szpilki",
        "portfel", "portfel", "portfel",
        "sneakersy", "sneakersy", "sneakersy",
        "kozaki", "kozaki", "kozaki",
        "trampki", "trampki", "trampki",
        "sandaly", "sandaly", "sandaly",
    ]
    return pd.DataFrame({
        "name": names,
        "name_clean": [n.replace("  ", " ") for n in names],
        "product_type": labels,
    })
