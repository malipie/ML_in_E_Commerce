"""Tests for DataQualityMonitor."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_quality import DataQualityMonitor


@pytest.fixture
def reference_data():
    """Create a reference DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sales": np.random.normal(1000, 200, n),
        "day_of_week": np.tile(range(7), n // 7 + 1)[:n],
        "month": dates.month,
        "lag_1": np.random.normal(1000, 200, n),
    })


def test_valid_data_passes(reference_data):
    """Reference-like data should pass validation."""
    monitor = DataQualityMonitor(reference_data)
    report = monitor.validate(reference_data)

    assert report.is_valid is True
    assert len(report.errors) == 0


def test_missing_columns_fails(reference_data):
    """Data with missing required columns should fail validation."""
    monitor = DataQualityMonitor(reference_data)
    incomplete_data = reference_data.drop(columns=["sales", "lag_1"])
    report = monitor.validate(incomplete_data)

    assert report.is_valid is False
    assert any("Missing columns" in e for e in report.errors)


def test_extreme_outliers_warn(reference_data):
    """Values far outside reference range should produce warnings."""
    monitor = DataQualityMonitor(reference_data)

    bad_data = reference_data.copy()
    bad_data.loc[0, "sales"] = -100000  # Extremely low value

    report = monitor.validate(bad_data)

    # Should produce a warning about out-of-range values
    assert len(report.warnings) > 0 or len(report.errors) > 0


def test_duplicate_dates_fail(reference_data):
    """Duplicate dates should cause validation error."""
    monitor = DataQualityMonitor(reference_data)

    dup_data = pd.concat([reference_data, reference_data.iloc[:5]], ignore_index=True)
    report = monitor.validate(dup_data)

    assert report.is_valid is False
    assert any("duplicate" in e.lower() for e in report.errors)
