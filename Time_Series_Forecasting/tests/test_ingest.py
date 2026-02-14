import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
from src.data.ingest import load_and_preprocess, parse_xml_to_df, process_data


def create_sample_xml(file_path):
    """Create a sample XML for testing."""
    root = ET.Element("orders")
    orders = [
        {"date_add": "01.01.2023 10:00:00", "order_amount_brutto": "100.0"},
        {"date_add": "01.01.2023 11:00:00", "order_amount_brutto": "200.0"},
        {"date_add": "02.01.2023 12:00:00", "order_amount_brutto": "150.0"},
    ]
    for order in orders:
        order_elem = ET.SubElement(root, "order")
        ET.SubElement(order_elem, "date_add").text = order["date_add"]
        ET.SubElement(order_elem, "order_amount_brutto").text = order["order_amount_brutto"]

    tree = ET.ElementTree(root)
    tree.write(file_path)

def test_parse_xml_to_df():
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        create_sample_xml(tmp.name)
        df = parse_xml_to_df(tmp.name)

        assert len(df) == 3
        assert 'date_add' in df.columns
        assert 'order_amount_brutto' in df.columns
        assert df['order_amount_brutto'].sum() == 450.0

def test_process_data():
    # Sample raw data
    data = {
        'date_add': ['01.01.2023 10:00:00', '01.01.2023 11:00:00', '02.01.2023 12:00:00'],
        'order_amount_brutto': [100.0, 200.0, 150.0]
    }
    df_raw = pd.DataFrame(data)
    df_processed = process_data(df_raw)

    assert len(df_processed) == 2  # Two days
    assert 'date' in df_processed.columns
    assert 'sales' in df_processed.columns
    assert df_processed['sales'].iloc[0] == 300.0  # Sum for day 1
    assert df_processed['sales'].iloc[1] == 150.0  # Day 2

def test_load_and_preprocess():
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        create_sample_xml(tmp.name)
        df = load_and_preprocess(tmp.name)

        assert len(df) == 2
        assert df['sales'].sum() == 450.0
