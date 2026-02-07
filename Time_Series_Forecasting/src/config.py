"""Centralized configuration for the Time Series Forecasting project."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with sensible defaults."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "Data"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    results_dir: Path = project_root / "results"

    # Data files
    raw_xml_pattern: str = "*.xml"
    daily_sales_filename: str = "daily_sales_clean.parquet"
    features_filename: str = "features.parquet"
    anomalies_filename: str = "anomalies_detected.parquet"

    # Model files
    champion_model_filename: str = "champion_model.pkl"
    champion_metadata_filename: str = "champion_model_metadata.json"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "Time_Series_Forecasting"
    mlflow_model_name: str = "ts-forecasting-champion"

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Monitoring thresholds
    drift_psi_threshold: float = 0.2
    performance_mae_threshold: float = 0.10  # 10% degradation triggers alert
    retraining_schedule_hours: int = 168  # Weekly (7 * 24)

    # Retraining
    min_new_records_for_retrain: int = 30
    cv_folds: int = 3

    # Security
    api_key_secret: str = "change-me-in-production"
    api_key_enabled: bool = True
    rate_limit_per_minute: int = 60
    cors_origins: str = "*"

    model_config = {"env_prefix": "TSF_"}

    def get_champion_model_path(self) -> Path:
        return self.models_dir / self.champion_model_filename

    def get_champion_metadata_path(self) -> Path:
        return self.models_dir / self.champion_metadata_filename

    def get_features_path(self) -> Path:
        return self.processed_dir / self.features_filename

    def get_daily_sales_path(self) -> Path:
        return self.processed_dir / self.daily_sales_filename

    def get_anomalies_path(self) -> Path:
        return self.processed_dir / self.anomalies_filename


settings = Settings()
