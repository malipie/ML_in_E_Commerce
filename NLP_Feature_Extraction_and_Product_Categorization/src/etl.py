"""ETL module: XML order data → item-level pandas DataFrame.

Parses multiple XML files containing e-commerce orders, extracts item-level
product data (name, SKU, price) with order context (order_id, date, comments),
cleans data types, and persists the result as Parquet.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORDER_FIELDS: list[str] = [
    "order_id",
    "date_add",
    "payment_name",
    "delivery_type",
    "client_city",
    "order_amount_brutto",
    "seller_comments",
    "buyer_comments",
]

ITEM_FIELDS: list[str] = [
    "name",
    "products_id",
    "products_ean",
    "products_sku",
    "auction_id",
    "item_price_brutto",
    "quantity",
    "amount_brutto",
    "vat_rate",
    "symkar",
]


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------


def _parse_order_element(order_el: etree._Element) -> list[dict[str, Any]]:
    """Extract item-level dicts from one <order>, each carrying order fields.

    Args:
        order_el: An lxml Element representing one ``<order>``.

    Returns:
        List of item dicts. Each dict contains order-level + item-level fields.
    """
    order_data: dict[str, Any] = {}
    for field in ORDER_FIELDS:
        node = order_el.find(field)
        order_data[field] = node.text if node is not None else None

    items: list[dict[str, Any]] = []
    rows_el = order_el.find("rows")
    if rows_el is not None:
        for row_el in rows_el.findall("row"):
            item: dict[str, Any] = dict(order_data)
            for field in ITEM_FIELDS:
                node = row_el.find(field)
                item[field] = node.text if node is not None else None
            items.append(item)

    if not items:
        item = dict(order_data)
        for field in ITEM_FIELDS:
            item[field] = None
        items.append(item)

    return items


def parse_xml_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Parse a single XML file and return a list of item-level dicts.

    Args:
        file_path: Path to the XML file.

    Returns:
        List of item dictionaries (one per product row).

    Raises:
        FileNotFoundError: If the file does not exist.
        etree.XMLSyntaxError: If the XML is malformed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"XML file not found: {file_path}")

    tree = etree.parse(str(file_path))  # noqa: S320
    root = tree.getroot()

    items: list[dict[str, Any]] = []
    for order_el in root.findall("order"):
        items.extend(_parse_order_element(order_el))

    logger.info("Parsed %d items from %s", len(items), file_path.name)
    return items


def parse_xml_directory(
    directory: str | Path, pattern: str = "*.xml"
) -> list[dict[str, Any]]:
    """Parse all XML files matching *pattern* in a directory.

    Args:
        directory: Path to directory containing XML files.
        pattern: Glob pattern for XML files.

    Returns:
        Concatenated list of item dictionaries from all matching files.

    Raises:
        FileNotFoundError: If the directory does not exist or has no matches.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    all_items: list[dict[str, Any]] = []
    for f in files:
        all_items.extend(parse_xml_file(f))

    logger.info("Total items parsed: %d from %d files", len(all_items), len(files))
    return all_items


# ---------------------------------------------------------------------------
# Cleaning & Type Casting
# ---------------------------------------------------------------------------


def _generate_item_id(order_id: str | None, products_id: str | None,
                      auction_id: str | None) -> str:
    """Generate a deterministic item_id from composite key.

    Args:
        order_id: The order identifier.
        products_id: The product identifier.
        auction_id: The auction identifier.

    Returns:
        SHA-256 hex digest (first 16 characters).
    """
    key = f"{order_id or ''}|{products_id or ''}|{auction_id or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _parse_datetime(date_str: str | None) -> pd.Timestamp | None:
    """Parse date string in ``DD.MM.YYYY HH:MM:SS`` format.

    Args:
        date_str: Date string or None.

    Returns:
        Parsed Timestamp or None if parsing fails.
    """
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str, format="%d.%m.%Y %H:%M:%S")
    except (ValueError, TypeError):
        logger.warning("Failed to parse date: %s", date_str)
        return None


def _safe_float(value: Any) -> float:
    """Convert value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _clean_text(value: Any) -> str | None:
    """Strip whitespace and return None for empty/NULL/NaN values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value)
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "NULL":
        return None
    return cleaned


def items_to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw item dicts to a clean, typed DataFrame.

    Performs:
    - Date parsing
    - Numeric type casting (prices, quantity, vat_rate)
    - item_id generation
    - Text cleaning (seller/buyer comments)
    - Column ordering

    Args:
        items: List of item dicts as returned by ``parse_xml_file``.

    Returns:
        DataFrame with one row per product item, cleaned and typed.
    """
    df = pd.DataFrame(items)

    if df.empty:
        return df

    # --- Text cleaning ---
    df["name"] = df["name"].apply(_clean_text)
    df["seller_comments"] = df["seller_comments"].apply(_clean_text)
    df["buyer_comments"] = df["buyer_comments"].apply(_clean_text)
    df["products_sku"] = df["products_sku"].apply(_clean_text)

    # --- Type conversions ---
    df["date_add"] = df["date_add"].apply(_parse_datetime)
    df["order_amount_brutto"] = df["order_amount_brutto"].apply(_safe_float)
    df["item_price_brutto"] = df["item_price_brutto"].apply(_safe_float)
    df["amount_brutto"] = df["amount_brutto"].apply(_safe_float)
    df["vat_rate"] = df["vat_rate"].apply(_safe_float)
    df["quantity"] = df["quantity"].apply(
        lambda x: int(_safe_float(x)) if x is not None and not (isinstance(x, float) and pd.isna(x)) else 0
    )

    # --- Derived fields ---
    df["item_id"] = df.apply(
        lambda row: _generate_item_id(
            row.get("order_id"), row.get("products_id"), row.get("auction_id")
        ),
        axis=1,
    )

    # --- Column ordering ---
    leading_cols = ["item_id", "order_id", "name", "products_sku"]
    other_cols = [c for c in df.columns if c not in leading_cols]
    df = df[leading_cols + other_cols]

    logger.info("DataFrame created: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Save DataFrame to Parquet format.

    Args:
        df: DataFrame to persist.
        path: Output file path.

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_etl(
    raw_dir: str | Path, output_path: str | Path, pattern: str = "*.xml"
) -> pd.DataFrame:
    """Execute the full ETL pipeline: parse → clean → save.

    Args:
        raw_dir: Directory containing source XML files.
        output_path: Destination path for the Parquet output.
        pattern: Glob pattern to match XML files.

    Returns:
        The cleaned DataFrame.
    """
    logger.info("Starting ETL pipeline: %s/%s → %s", raw_dir, pattern, output_path)
    items = parse_xml_directory(raw_dir, pattern)
    df = items_to_dataframe(items)
    save_parquet(df, output_path)
    logger.info("ETL pipeline complete.")
    return df
