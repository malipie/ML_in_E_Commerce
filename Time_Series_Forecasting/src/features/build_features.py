import logging

import pandas as pd

# Try to import holidays, else use a fallback
try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar features: day_of_week, month, quarter, year, is_weekend, is_holiday.
    """
    logging.info("Adding calendar features...")
    df = df.copy()

    if 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'date' column.")

    # Basic calendar features
    df['day_of_week'] = df['date'].dt.dayofweek # 0=Monday, 6=Sunday
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    # is_weekend (Saturday=5, Sunday=6)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # is_holiday
    if HAS_HOLIDAYS:
        # Assuming Poland as per common e-commerce context in prompt (implied by "Poland holiday library")
        pl_holidays = holidays.Poland()
        df['is_holiday'] = df['date'].apply(lambda x: 1 if x in pl_holidays else 0)
    else:
        logging.warning("Holidays library not found. 'is_holiday' will be all 0.")
        df['is_holiday'] = 0

    return df

def add_payday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds specific payday flags.
    Payday often happens around 1st and 10th.
    """
    logging.info("Adding payday features...")
    df = df.copy()

    # 'Payday effect': close to 1st (1-3) and 10th (10-12)
    # Adjust window as needed
    df['is_payday_1st'] = df['day_of_month'].isin([1, 2, 3]).astype(int)
    df['is_payday_10th'] = df['day_of_month'].isin([10, 11, 12]).astype(int)

    # Combined flag
    df['is_payday'] = (df['is_payday_1st'] | df['is_payday_10th']).astype(int)

    return df

def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """
    Adds lag features.
    """
    if lags is None:
        lags = [1, 7]
    logging.info(f"Adding lag features: {lags}...")
    df = df.copy()

    # Ensure data is sorted by date before lagging
    df = df.sort_values('date')

    for lag in lags:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

    return df

def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Adds rolling mean features.
    Warning: Rolling features needs to be Shifted to avoid data leakage!
    e.g. rolling mean of last 7 days valid for TODAY should be calculated based on T-1 backwards.
    """
    logging.info(f"Adding rolling features (window={window})...")
    df = df.copy()

    # Shift by 1 to ensure we don't use today's target for today's feature
    shifted_sales = df['sales'].shift(1)

    df[f'rolling_mean_{window}'] = shifted_sales.rolling(window=window).mean()

    return df

def create_features(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Orchestrates feature engineering.
    """
    if not input_path:
         raise ValueError("Input path required.")

    logging.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
         # Fallback to csv if parquet fails/not found (though ingest saves parquet)
         logging.warning(f"Could not read parquet: {e}. Trying CSV or re-raising.")
         raise e

    df = add_calendar_features(df)
    df = add_payday_features(df)
    df = add_lag_features(df, lags=[1, 7])
    df = add_rolling_features(df, window=7)

    # Drop rows with NaN created by lags/rolling (or handle them if instructed, but usually strict ML models don't like NaNs)
    # For now, I'll keep them but might need imputation or dropping in training.
    # Actually, let's just drop the initial rows where lags are undefined to keep it clean for training
    df_clean = df.dropna()
    logging.info(f"Dropped {len(df) - len(df_clean)} rows due to NaN lags/rolling.")

    if output_path:
        df_clean.to_parquet(output_path, index=False)
        logging.info(f"Saved features to {output_path}")

    return df_clean

if __name__ == "__main__":

    input_file = "data/processed/daily_sales.parquet"
    output_file = "data/processed/features.parquet"

    # Simple check if holidays installed
    if not HAS_HOLIDAYS:
        print("Install 'holidays' library for holiday features: pip install holidays")

    try:
        create_features(input_file, output_file)
        print("Feature engineering successful.")
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
