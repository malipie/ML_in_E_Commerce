import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_xml_to_df(xml_path: str) -> pd.DataFrame:
    """
    Parses the XML file and extracts relevant fields: date_add, order_amount_brutto.
    """
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    logging.info(f"Parsing XML file: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []

    # Iterate over each order
    for order in root.findall('order'):
        date_add_str = order.find('date_add').text
        amount_text = order.find('order_amount_brutto').text

        if date_add_str and amount_text:
            data.append({
                'date_add': date_add_str,
                'order_amount_brutto': float(amount_text)
            })

    df = pd.DataFrame(data)
    logging.info(f"Parsed {len(df)} orders.")
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts dates, aggregates by day, and fills missing dates.
    """
    logging.info("Processing data...")

    # Convert date_add to datetime (DD.MM.YYYY HH:MM:SS)
    df['date_add'] = pd.to_datetime(df['date_add'], format='%d.%m.%Y %H:%M:%S')

    # Normalize to date only (remove time) to group by day
    df['date'] = df['date_add'].dt.normalize()

    # Group by date and sum sales
    daily_sales = df.groupby('date')['order_amount_brutto'].sum().reset_index()
    daily_sales.columns = ['date', 'sales']

    # Handle missing dates (fill with 0)
    if not daily_sales.empty:
        full_range = pd.date_range(start=daily_sales['date'].min(), end=daily_sales['date'].max(), freq='D')
        daily_sales = daily_sales.set_index('date').reindex(full_range, fill_value=0).rename_axis('date').reset_index()

    logging.info(f"Data aggregated. Time range: {daily_sales['date'].min()} to {daily_sales['date'].max()}. Total days: {len(daily_sales)}")
    return daily_sales

def load_and_preprocess(xml_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Orchestrates ingestion: Load -> Parse -> Process -> (Optional) Save.
    """
    df_raw = parse_xml_to_df(xml_path)
    df_daily = process_data(df_raw)

    if output_path:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_daily.to_parquet(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")

    return df_daily
