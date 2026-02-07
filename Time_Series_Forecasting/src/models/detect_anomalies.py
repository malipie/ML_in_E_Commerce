import logging

import pandas as pd
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_anomalies(input_path: str, output_path: str | None = None, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detects anomalies using Isolation Forest.
    """
    logging.info(f"Loading data from {input_path} for anomaly detection...")
    df = pd.read_parquet(input_path)

    # We can use sales + maybe rolling mean as features for anomaly detection
    # If we just use sales, it detects global outliers.
    # If we use sales + diff, it detects suddent changes.

    # Simple approach: Sales only
    X = df[['sales']]

    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = model.fit_predict(X)

    # -1 is anomaly, 1 is normal
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)

    anomalies = df[df['is_anomaly']]
    logging.info(f"Detected {len(anomalies)} anomalies out of {len(df)} records.")

    if output_path:
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved anomaly results to {output_path}")

    return df

if __name__ == "__main__":
    detect_anomalies("data/processed/features.parquet", "data/processed/anomalies.parquet")
