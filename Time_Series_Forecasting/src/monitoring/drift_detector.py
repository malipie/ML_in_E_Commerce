"""Data and prediction drift detection using Evidently and PSI."""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except ImportError:
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report

from src.config import settings

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects feature distribution drift and prediction error trends.

    Usage:
        detector = DriftDetector(reference_data=train_df)
        report = detector.check_drift(current_data=new_incoming_df)
        if report["dataset_drift"]:
            trigger_alert(...)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_col: str = "sales",
        prediction_col: str = "prediction",
        psi_threshold: float | None = None,
    ):
        self.reference_data = reference_data
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.psi_threshold = psi_threshold or settings.drift_psi_threshold

        exclude_cols = {
            target_col, prediction_col, "date", "ds",
            "anomaly_score", "is_anomaly", "anomaly_any",
            "anomaly_isoforest", "anomaly_zscore", "anomaly_iqr",
            "z_score", "is_outlier",
        }
        self.feature_cols = [
            c for c in reference_data.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

    def check_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Run Evidently DatasetDriftMetric on current vs reference data.

        Returns a dictionary with:
            - dataset_drift: bool
            - drift_share: float (fraction of drifted features)
            - drifted_features: list[str]
            - timestamp: str
        """
        ref = self.reference_data[self.feature_cols].copy()
        cur = current_data[[c for c in self.feature_cols if c in current_data.columns]].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        result_dict = report.as_dict()
        metrics_list = result_dict["metrics"]

        # Metric 0 = DatasetDriftMetric (summary), Metric 1 = DataDriftTable (per-column)
        summary = metrics_list[0]["result"]

        drifted_features = []
        if len(metrics_list) > 1:
            table_result = metrics_list[1]["result"]
            drift_by_columns = table_result.get("drift_by_columns", {})
            for col_name, col_info in drift_by_columns.items():
                if col_info.get("drift_detected", False):
                    drifted_features.append(col_name)

        drift_report = {
            "dataset_drift": summary.get("dataset_drift", False),
            "drift_share": summary.get("share_of_drifted_columns",
                           summary.get("drift_share", 0.0)),
            "number_of_drifted_columns": summary.get(
                "number_of_drifted_columns", 0
            ),
            "drifted_features": drifted_features,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if drift_report["dataset_drift"]:
            logger.warning(
                "DATA DRIFT DETECTED: %.1f%% of features drifted: %s",
                drift_report["drift_share"] * 100,
                drifted_features,
            )
        else:
            logger.info(
                "No significant drift. Drift share: %.1f%%",
                drift_report["drift_share"] * 100,
            )

        return drift_report

    def check_prediction_performance(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        baseline_mae: float,
    ) -> dict:
        """
        Check if prediction performance has degraded beyond threshold.

        Returns:
            - current_mae: float
            - baseline_mae: float
            - degradation_pct: float (percentage)
            - alert: bool
        """
        current_mae = mean_absolute_error(actuals, predictions)
        degradation_pct = (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0.0

        alert = degradation_pct > settings.performance_mae_threshold

        result = {
            "current_mae": round(current_mae, 2),
            "baseline_mae": round(baseline_mae, 2),
            "degradation_pct": round(degradation_pct * 100, 2),
            "alert": alert,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if alert:
            logger.warning(
                "PERFORMANCE DEGRADATION: MAE increased by %.1f%% "
                "(current=%.0f, baseline=%.0f)",
                degradation_pct * 100,
                current_mae,
                baseline_mae,
            )

        return result


def calculate_psi(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    PSI < 0.1: no significant change
    0.1 <= PSI < 0.2: moderate change
    PSI >= 0.2: significant change
    """
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    eps = 1e-6
    ref_pct = ref_hist / len(reference) + eps
    cur_pct = cur_hist / len(current) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
