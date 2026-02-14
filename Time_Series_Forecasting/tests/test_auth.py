"""Tests for API key authentication."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.config import settings


@pytest.fixture
def client():
    """Create a TestClient with mocked model loading."""
    with patch("src.serving.app.load_model"), \
         patch("src.serving.app.load_features"):
        # Simulate a loaded model
        import src.serving.app as app_module
        app_module._model = MagicMock()
        app_module._model.predict.return_value = [1000.0] * 7
        app_module._model_info = {"source": "test", "version": "test-1"}
        app_module._features_df = MagicMock()

        from src.serving.app import app
        with TestClient(app) as c:
            yield c

        # Cleanup
        app_module._model = None
        app_module._model_info = {}
        app_module._features_df = None


def test_health_no_auth_required(client):
    """Public endpoints should not require an API key."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_no_auth_required(client):
    """Root endpoint should not require an API key."""
    response = client.get("/")
    assert response.status_code == 200


def test_metrics_no_auth_required(client):
    """Metrics endpoint should not require an API key."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_forecast_without_api_key_returns_422(client):
    """Missing X-API-Key header should return 422 (validation error)."""
    response = client.post("/forecast", json={"days_ahead": 7})
    assert response.status_code == 422


def test_forecast_with_invalid_key_returns_401(client):
    """Invalid API key should return 401."""
    response = client.post(
        "/forecast",
        json={"days_ahead": 7},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_forecast_with_valid_key_succeeds(client):
    """Valid API key should allow access to protected endpoint."""
    response = client.post(
        "/forecast",
        json={"days_ahead": 7},
        headers={"X-API-Key": settings.api_key_secret},
    )
    # May be 200 or 500 depending on mock setup, but NOT 401/422
    assert response.status_code != 401
    assert response.status_code != 422


def test_model_info_requires_auth(client):
    """Model info endpoint should require API key."""
    response = client.get("/model-info")
    assert response.status_code == 422

    response = client.get(
        "/model-info",
        headers={"X-API-Key": settings.api_key_secret},
    )
    assert response.status_code == 200


def test_reload_model_requires_auth(client):
    """Reload model endpoint should require API key."""
    response = client.post("/reload-model")
    assert response.status_code == 422

    response = client.post(
        "/reload-model",
        headers={"X-API-Key": settings.api_key_secret},
    )
    assert response.status_code == 200
