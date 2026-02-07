"""Tests for ModelRegistry (mocked MLflow)."""

from unittest.mock import MagicMock, patch

import pytest

# Import the module first so patches can resolve
import src.models.registry as registry_module


@pytest.fixture
def mock_mlflow():
    """Patch MLflow so ModelRegistry can be instantiated without a server."""
    with patch.object(registry_module, "mlflow") as mock_ml, \
         patch.object(registry_module, "MlflowClient") as mock_client_cls:

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate model already registered
        mock_client.get_registered_model.return_value = MagicMock()

        yield mock_ml, mock_client


def test_register_champion_creates_run(mock_mlflow):
    """Verify mlflow.start_run is called and model version is returned."""
    mock_ml, mock_client = mock_mlflow

    # Mock the run context
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    mock_ml.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_ml.start_run.return_value.__exit__ = MagicMock(return_value=False)

    # Mock version search
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_client.search_model_versions.return_value = [mock_version]

    registry = registry_module.ModelRegistry()
    version = registry.register_champion(
        model=MagicMock(),
        metrics={"mae": 100.0, "rmse": 150.0},
        params={"model_type": "XGBoost"},
        features=["feat1", "feat2"],
        train_date_range={"start": "2023-01-01", "end": "2024-01-01"},
        test_date_range={"start": "cv", "end": "cv"},
    )

    assert version == "1"
    mock_ml.start_run.assert_called_once()
    mock_ml.sklearn.log_model.assert_called_once()


def test_promote_archives_current(mock_mlflow):
    """Verify old production model is archived before promoting new one."""
    mock_ml, mock_client = mock_mlflow

    old_prod = MagicMock()
    old_prod.version = "1"
    mock_client.get_latest_versions.return_value = [old_prod]

    registry = registry_module.ModelRegistry()
    registry.promote_to_production("2")

    # Should archive v1 and promote v2
    calls = mock_client.transition_model_version_stage.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["version"] == "1"
    assert calls[0].kwargs["stage"] == "Archived"
    assert calls[1].kwargs["version"] == "2"
    assert calls[1].kwargs["stage"] == "Production"


def test_rollback_restores_archived(mock_mlflow):
    """Verify rollback finds and promotes the latest archived version."""
    mock_ml, mock_client = mock_mlflow

    archived_v = MagicMock()
    archived_v.version = "3"

    # get_latest_versions called: first for archived, then in promote_to_production for current prod
    mock_client.get_latest_versions.side_effect = [
        [archived_v],  # archived versions
        [],            # current production (none, for promote)
    ]

    registry = registry_module.ModelRegistry()
    result = registry.rollback()

    assert result == "3"
