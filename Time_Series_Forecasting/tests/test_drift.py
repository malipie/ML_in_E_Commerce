"""Tests for DriftDetector and PSI calculation."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector, calculate_psi


@pytest.fixture
def reference_data():
    """Create a simple reference DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "sales": np.random.normal(1000, 200, n),
        "day_of_week": np.random.randint(0, 7, n),
        "month": np.random.randint(1, 13, n),
        "lag_1": np.random.normal(1000, 200, n),
        "rolling_mean_7": np.random.normal(1000, 150, n),
    })


def test_no_drift_same_data(reference_data):
    """Same data as reference and current should produce no dataset-level drift."""
    detector = DriftDetector(reference_data, target_col="sales")
    report = detector.check_drift(reference_data)

    assert report["dataset_drift"] is False


def test_drift_detected_shifted(reference_data):
    """Heavily shifted data should trigger drift detection."""
    np.random.seed(99)
    n = 200
    shifted_data = pd.DataFrame({
        "sales": np.random.normal(5000, 200, n),
        "day_of_week": np.random.randint(0, 7, n),
        "month": np.random.randint(1, 13, n),
        "lag_1": np.random.normal(5000, 200, n),
        "rolling_mean_7": np.random.normal(5000, 150, n),
    })

    detector = DriftDetector(reference_data, target_col="sales")
    report = detector.check_drift(shifted_data)

    assert report["dataset_drift"] is True
    assert report["number_of_drifted_columns"] > 0


def test_psi_identical_is_near_zero():
    """PSI of identical distributions should be near zero."""
    np.random.seed(42)
    data = np.random.normal(100, 20, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.01


def test_psi_shifted_is_high():
    """PSI of significantly shifted distribution should be >= 0.2."""
    np.random.seed(42)
    reference = np.random.normal(100, 20, 1000)
    shifted = np.random.normal(200, 20, 1000)
    psi = calculate_psi(reference, shifted)
    assert psi >= 0.2
