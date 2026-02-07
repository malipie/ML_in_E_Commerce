"""MLflow Model Registry integration for model versioning and lifecycle management."""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model registration, stage transitions, and rollback via MLflow."""

    STAGE_STAGING = "Staging"
    STAGE_PRODUCTION = "Production"
    STAGE_ARCHIVED = "Archived"

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str | None = None,
    ):
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.model_name = model_name or settings.mlflow_model_name
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self._ensure_registered_model_exists()

    def _ensure_registered_model_exists(self) -> None:
        """Create the registered model if it does not exist yet."""
        try:
            self.client.get_registered_model(self.model_name)
        except MlflowException:
            self.client.create_registered_model(
                self.model_name,
                description="Time Series Forecasting champion model",
            )
            logger.info("Created registered model: %s", self.model_name)

    def register_champion(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
        features: list[str],
        train_date_range: dict[str, str],
        test_date_range: dict[str, str],
    ) -> str:
        """
        Log a model run to MLflow and register it in the Model Registry.

        Returns the new model version string.
        """
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="champion_registration") as run:
            mlflow.log_params({
                k: str(v)[:250] for k, v in params.items()
            })
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("train_start", train_date_range.get("start", ""))
            mlflow.log_param("train_end", train_date_range.get("end", ""))

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.model_name,
            )

            metadata = {
                "features": features,
                "metrics": metrics,
                "params": {k: str(v) for k, v in params.items()},
                "train_date_range": train_date_range,
                "test_date_range": test_date_range,
            }
            metadata_path = Path(settings.models_dir) / "_mlflow_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
            mlflow.log_artifact(str(metadata_path))

            run_id = run.info.run_id

        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        version_str = latest_version.version

        logger.info(
            "Registered model version %s (run_id=%s, MAE=%s)",
            version_str,
            run_id,
            metrics.get("mae", "N/A"),
        )
        return version_str

    def transition_to_staging(self, version: str) -> None:
        """Move a model version to Staging."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=self.STAGE_STAGING,
        )
        logger.info("Model v%s transitioned to %s", version, self.STAGE_STAGING)

    def promote_to_production(self, version: str) -> None:
        """Promote a model version to Production, archiving the current one."""
        prod_versions = self.client.get_latest_versions(
            self.model_name, stages=[self.STAGE_PRODUCTION]
        )
        for pv in prod_versions:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=pv.version,
                stage=self.STAGE_ARCHIVED,
            )
            logger.info("Archived previous production model v%s", pv.version)

        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=self.STAGE_PRODUCTION,
        )
        logger.info("Model v%s promoted to %s", version, self.STAGE_PRODUCTION)

    def rollback(self) -> str | None:
        """Roll back to the most recent Archived model version."""
        archived = self.client.get_latest_versions(
            self.model_name, stages=[self.STAGE_ARCHIVED]
        )
        if not archived:
            logger.warning("No archived versions available for rollback")
            return None

        rollback_version = max(archived, key=lambda v: int(v.version))
        self.promote_to_production(rollback_version.version)
        logger.info("Rolled back to model v%s", rollback_version.version)
        return rollback_version.version

    def load_production_model(self):
        """Load the current Production model from the registry."""
        model_uri = f"models:/{self.model_name}/{self.STAGE_PRODUCTION}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Loaded production model from %s", model_uri)
            return model
        except MlflowException as e:
            logger.error("Failed to load production model: %s", e)
            return None

    def load_staging_model(self):
        """Load the current Staging model from the registry."""
        model_uri = f"models:/{self.model_name}/{self.STAGE_STAGING}"
        try:
            return mlflow.sklearn.load_model(model_uri)
        except MlflowException:
            return None

    def get_production_version_info(self) -> dict | None:
        """Get metadata about the current production model version."""
        prod_versions = self.client.get_latest_versions(
            self.model_name, stages=[self.STAGE_PRODUCTION]
        )
        if not prod_versions:
            return None
        v = prod_versions[0]
        run = self.client.get_run(v.run_id)
        return {
            "version": v.version,
            "run_id": v.run_id,
            "stage": v.current_stage,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "created_at": str(v.creation_timestamp),
        }
