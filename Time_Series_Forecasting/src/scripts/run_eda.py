import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def run_eda():
    print("--- EDA REPORT ---")
    data_dir = Path("Data")
    xml_files = list(data_dir.glob("*.xml"))
    if not xml_files:
        print("No XML files found in Data/")
        return

    xml_path = xml_files[0]
    print(f"Loading: {xml_path}")

    # Parse
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for order in root.findall('order'):
        date_add = order.find('date_add').text
        amount = order.find('order_amount_brutto').text
        if date_add and amount:
            data.append({'date_add': date_add, 'order_amount_brutto': float(amount)})

    df = pd.DataFrame(data)
    df['date_add'] = pd.to_datetime(df['date_add'], format='%d.%m.%Y %H:%M:%S')

    print(f"Total Rows: {len(df)}")
    print(f"Date Range: {df['date_add'].min()} to {df['date_add'].max()}")

    # Daily aggregation
    df['date'] = df['date_add'].dt.normalize()
    daily = df.groupby('date')['order_amount_brutto'].sum().reset_index()
    daily.columns = ['date', 'sales']

    print("\n--- Daily Stats ---")
    print(daily['sales'].describe())

    # Missing dates
    full_idx = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
    missing_dates = full_idx.difference(daily['date'])
    print(f"\nMissing Dates Count: {len(missing_dates)}")
    if len(missing_dates) > 0:
        print(f"First 5 missing: {missing_dates[:5]}")

    print("\n--- Seasonality Check (Simple) ---")
    daily.set_index('date', inplace=True)
    daily = daily.reindex(full_idx, fill_value=0)

    # Weekly pattern ?
    daily['weekday'] = daily.index.dayofweek
    weekly_mean = daily.groupby('weekday')['sales'].mean()
    print("Average Sales by Day of Week (0=Mon, 6=Sun):")
    print(weekly_mean)

if __name__ == "__main__":
    run_eda()
