"""Automated retraining pipeline with trigger evaluation and model promotion."""

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Evaluates retraining triggers and orchestrates the train-evaluate-promote cycle.

    Triggers:
        - time_based: Hours since last training exceeded threshold
        - performance_based: MAE degradation above threshold
        - data_based: Drift detected
    """

    def __init__(self):
        self.last_retrain_path = settings.models_dir / "last_retrain.json"

    def should_retrain(
        self,
        drift_report: dict | None = None,
        performance_report: dict | None = None,
        new_data_count: int = 0,
    ) -> dict:
        """
        Evaluate all retraining triggers.

        Returns dict with should_retrain (bool), reasons (list), trigger_details (dict).
        """
        reasons = []
        trigger_details = {}

        # 1. Time-based trigger
        hours_since_last = self._hours_since_last_retrain()
        trigger_details["hours_since_last_retrain"] = round(hours_since_last, 1)
        if hours_since_last > settings.retraining_schedule_hours:
            reasons.append(
                f"Time-based: {hours_since_last:.0f}h since last retrain "
                f"(threshold: {settings.retraining_schedule_hours}h)"
            )

        # 2. Performance-based trigger
        if performance_report and performance_report.get("alert"):
            degradation = performance_report["degradation_pct"]
            reasons.append(f"Performance-based: MAE degraded by {degradation:.1f}%")
            trigger_details["performance_degradation"] = degradation

        # 3. Data-based trigger (drift)
        if drift_report and drift_report.get("dataset_drift"):
            reasons.append(
                f"Data-based: Drift detected in "
                f"{drift_report['number_of_drifted_columns']} features"
            )
            trigger_details["drift_share"] = drift_report["drift_share"]

        # 4. Data volume
        trigger_details["new_data_count"] = new_data_count
        trigger_details["sufficient_new_data"] = (
            new_data_count >= settings.min_new_records_for_retrain
        )

        should = len(reasons) > 0

        result = {
            "should_retrain": should,
            "reasons": reasons,
            "trigger_details": trigger_details,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        if should:
            logger.info("Retraining TRIGGERED: %s", reasons)
        else:
            logger.info("No retraining triggers activated")

        return result

    def retrain(self) -> dict:
        """
        Execute the full retraining pipeline:
        1. Load latest data
        2. Build features
        3. Train models (ChampionChallenger)
        4. Compare with current production metrics
        5. Register new model in Staging if improved
        """
        logger.info("Starting retraining pipeline...")

        from src.data.ingest import load_and_preprocess
        from src.features.build_features import create_features
        from src.models.train_model import ChampionChallenger

        # Step 1: Reload and preprocess data
        data_dir = settings.data_dir
        xml_files = sorted(data_dir.glob(settings.raw_xml_pattern))
        if not xml_files:
            return {"status": "error", "message": "No XML data files found"}

        daily_sales_path = str(settings.get_daily_sales_path())
        load_and_preprocess(str(xml_files[0]), daily_sales_path)

        # Step 2: Build features
        features_path = str(settings.get_features_path())
        create_features(daily_sales_path, features_path)

        # Step 3: Train using ChampionChallenger
        df = pd.read_parquet(features_path)
        arena = ChampionChallenger(df)
        results_df = arena.run_cv(n_splits=settings.cv_folds)

        avg_results = results_df.groupby("model")["mae"].mean().sort_values()
        best_model_name = avg_results.index[0]
        best_mae = float(avg_results.iloc[0])

        logger.info("Best model from retraining: %s (MAE=%.2f)", best_model_name, best_mae)

        # Step 4: Train final model
        best_model = arena.train_final_best_model(best_model_name)

        # Step 5: Compare with current production metrics
        current_metrics = self._load_current_production_metrics()
        improved = True
        if current_metrics and "mae" in current_metrics:
            current_mae = current_metrics["mae"]
            if best_mae >= current_mae:
                improved = False
                logger.info(
                    "New model (MAE=%.2f) did NOT improve over production (MAE=%.2f). "
                    "Skipping registration.",
                    best_mae,
                    current_mae,
                )

        # Step 6: Register if improved
        action = "no_improvement"
        new_version = None
        if improved:
            try:
                from src.models.registry import ModelRegistry

                registry = ModelRegistry()
                feature_cols = [
                    c for c in df.columns
                    if c not in [arena.target_col, arena.date_col, "date_add"]
                ]
                avg_metrics = (
                    results_df[results_df["model"] == best_model_name]
                    .mean(numeric_only=True)
                    .to_dict()
                )
                new_version = registry.register_champion(
                    model=best_model,
                    metrics=avg_metrics,
                    params={"model_type": best_model_name},
                    features=feature_cols,
                    train_date_range={
                        "start": str(df["date"].min()),
                        "end": str(df["date"].max()),
                    },
                    test_date_range={"start": "cv", "end": "cv"},
                )
                registry.transition_to_staging(new_version)
                action = "registered_in_staging"
                logger.info("New model registered as v%s in Staging", new_version)
            except Exception as e:
                logger.error("Failed to register model: %s", e)
                action = "registration_failed"

        self._record_retrain(best_model_name, best_mae)

        return {
            "status": "completed",
            "best_model": best_model_name,
            "best_mae": round(best_mae, 2),
            "improved_over_production": improved,
            "action": action,
            "new_version": new_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _hours_since_last_retrain(self) -> float:
        """Read last retrain timestamp and compute hours elapsed."""
        if not self.last_retrain_path.exists():
            return float("inf")

        with open(self.last_retrain_path) as f:
            data = json.load(f)

        last_ts = datetime.fromisoformat(data["timestamp"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
        return elapsed

    def _record_retrain(self, model_name: str, mae: float) -> None:
        """Save retrain timestamp and results."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "mae": mae,
        }
        self.last_retrain_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.last_retrain_path, "w") as f:
            json.dump(record, f, indent=2)

    def _load_current_production_metrics(self) -> dict | None:
        """Load metrics of the current production model."""
        try:
            from src.models.registry import ModelRegistry

            registry = ModelRegistry()
            info = registry.get_production_version_info()
            if info:
                return info.get("metrics", {})
        except Exception:
            pass

        metadata_path = settings.get_champion_metadata_path()
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get("metrics", {})

        return None
