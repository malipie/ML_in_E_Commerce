"""ETL module: XML order data → clean pandas DataFrame.

Parses multiple XML files containing e-commerce orders, extracts order-level
and item-level fields, aggregates items per order, cleans data types,
and persists the result as Parquet.
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
    "payment_status",
    "payment_paid",
    "delivery_type",
    "delivery_price",
    "client_city",
    "order_amount_brutto",
]

ITEM_FIELDS: list[str] = [
    "item_price_brutto",
    "quantity",
    "amount_brutto",
]


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------


def _parse_order_element(order_el: etree._Element) -> dict[str, Any]:
    """Extract a flat dict from a single <order> element.

    Args:
        order_el: An lxml Element representing one ``<order>``.

    Returns:
        Dictionary with order-level scalars and an ``items`` list of dicts.
    """
    record: dict[str, Any] = {}

    for field in ORDER_FIELDS:
        node = order_el.find(field)
        record[field] = node.text if node is not None else None

    items: list[dict[str, Any]] = []
    rows_el = order_el.find("rows")
    if rows_el is not None:
        for row_el in rows_el.findall("row"):
            item: dict[str, Any] = {}
            for field in ITEM_FIELDS:
                node = row_el.find(field)
                item[field] = node.text if node is not None else None
            items.append(item)

    record["items"] = items
    return record


def parse_xml_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Parse a single XML file and return a list of order dicts.

    Args:
        file_path: Path to the XML file.

    Returns:
        List of order dictionaries (one per ``<order>`` element).

    Raises:
        FileNotFoundError: If the file does not exist.
        etree.XMLSyntaxError: If the XML is malformed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"XML file not found: {file_path}")

    tree = etree.parse(str(file_path))  # noqa: S320
    root = tree.getroot()

    orders: list[dict[str, Any]] = []
    for order_el in root.findall("order"):
        orders.append(_parse_order_element(order_el))

    logger.info("Parsed %d orders from %s", len(orders), file_path.name)
    return orders


def parse_xml_directory(directory: str | Path, pattern: str = "*.xml") -> list[dict[str, Any]]:
    """Parse all XML files matching *pattern* in a directory.

    Args:
        directory: Path to directory containing XML files.
        pattern: Glob pattern for XML files.

    Returns:
        Concatenated list of order dictionaries from all matching files.

    Raises:
        FileNotFoundError: If the directory does not exist or has no matches.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    all_orders: list[dict[str, Any]] = []
    for f in files:
        all_orders.extend(parse_xml_file(f))

    logger.info("Total orders parsed: %d from %d files", len(all_orders), len(files))
    return all_orders


# ---------------------------------------------------------------------------
# Aggregation & Cleaning
# ---------------------------------------------------------------------------


def _aggregate_items(items: list[dict[str, Any]]) -> dict[str, float]:
    """Compute order-level aggregates from item rows.

    Args:
        items: List of item dicts with ``item_price_brutto`` and ``quantity``.

    Returns:
        Dict with ``n_items``, ``avg_item_price``, ``max_item_price``,
        ``total_quantity``.
    """
    if not items:
        return {
            "n_items": 0,
            "avg_item_price": 0.0,
            "max_item_price": 0.0,
            "total_quantity": 0,
        }

    prices: list[float] = []
    total_qty = 0
    for item in items:
        try:
            price = float(item.get("item_price_brutto") or 0)
        except (ValueError, TypeError):
            price = 0.0
        prices.append(price)

        try:
            qty = int(float(item.get("quantity") or 0))
        except (ValueError, TypeError):
            qty = 0
        total_qty += qty

    return {
        "n_items": len(prices),
        "avg_item_price": sum(prices) / len(prices) if prices else 0.0,
        "max_item_price": max(prices) if prices else 0.0,
        "total_quantity": total_qty,
    }


def _generate_customer_id(order_id: str) -> str:
    """Generate a deterministic customer_id from order_id.

    Since each order is treated as a unique customer in this dataset,
    we hash the order_id to produce a consistent identifier.

    Args:
        order_id: The order identifier.

    Returns:
        SHA-256 hex digest (first 12 characters).
    """
    return hashlib.sha256(str(order_id).encode()).hexdigest()[:12]


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


def orders_to_dataframe(orders: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw order dicts to a clean, typed DataFrame.

    Performs:
    - Item aggregation (n_items, avg/max price, total_quantity)
    - Date parsing
    - Numeric type casting
    - customer_id generation

    Args:
        orders: List of order dicts as returned by ``parse_xml_file``.

    Returns:
        DataFrame with one row per order, cleaned and typed.
    """
    records: list[dict[str, Any]] = []

    for order in orders:
        items = order.pop("items", [])
        agg = _aggregate_items(items)

        record = {**order, **agg}
        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # --- Type conversions ---
    df["date_add"] = df["date_add"].apply(_parse_datetime)
    df["order_amount_brutto"] = df["order_amount_brutto"].apply(_safe_float)
    df["payment_paid"] = df["payment_paid"].apply(_safe_float)
    df["delivery_price"] = df["delivery_price"].apply(_safe_float)
    df["payment_status"] = df["payment_status"].apply(
        lambda x: int(float(x)) if x is not None else 0
    )

    # --- Derived fields ---
    df["customer_id"] = df["order_id"].apply(_generate_customer_id)
    df["hour_of_day"] = df["date_add"].dt.hour
    df["day_of_week"] = df["date_add"].dt.dayofweek  # 0=Monday

    # --- Column ordering ---
    leading_cols = ["customer_id", "order_id", "date_add"]
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


def run_etl(raw_dir: str | Path, output_path: str | Path, pattern: str = "*.xml") -> pd.DataFrame:
    """Execute the full ETL pipeline: parse → clean → save.

    Args:
        raw_dir: Directory containing source XML files.
        output_path: Destination path for the Parquet output.
        pattern: Glob pattern to match XML files.

    Returns:
        The cleaned DataFrame.
    """
    logger.info("Starting ETL pipeline: %s/%s → %s", raw_dir, pattern, output_path)
    orders = parse_xml_directory(raw_dir, pattern)
    df = orders_to_dataframe(orders)
    save_parquet(df, output_path)
    logger.info("ETL pipeline complete.")
    return df
