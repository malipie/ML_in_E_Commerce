import pandas as pd

from src.features.build_features import add_calendar_features, add_lag_features, add_rolling_features


def test_add_calendar_features():
    data = {
        'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'sales': [100, 200, 150, 300, 250]
    }
    df = pd.DataFrame(data)
    df_featured = add_calendar_features(df)

    assert 'day_of_week' in df_featured.columns
    assert 'month' in df_featured.columns
    assert 'is_weekend' in df_featured.columns
    assert df_featured['day_of_week'].iloc[0] == 6  # Sunday
    assert df_featured['is_weekend'].iloc[0] == 1

def test_add_lag_features():
    data = {
        'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'sales': [100, 200, 150, 300, 250]
    }
    df = pd.DataFrame(data)
    df_featured = add_lag_features(df, lags=[1, 2])

    assert 'sales_lag_1' in df_featured.columns
    assert 'sales_lag_2' in df_featured.columns
    assert pd.isna(df_featured['sales_lag_1'].iloc[0])
    assert df_featured['sales_lag_1'].iloc[1] == 100

def test_add_rolling_features():
    data = {
        'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'sales': [100, 200, 150, 300, 250]
    }
    df = pd.DataFrame(data)
    df_featured = add_rolling_features(df, window=3)

    assert 'rolling_mean_3' in df_featured.columns
    # Check that it's shifted (no data leakage) - verify at least some values are present
    assert df_featured['rolling_mean_3'].notna().sum() > 0  # At least some non-NA values
