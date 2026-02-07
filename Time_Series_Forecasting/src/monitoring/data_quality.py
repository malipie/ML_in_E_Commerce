"""Data quality validation for incoming data."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Result of a data quality check."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class DataQualityMonitor:
    """
    Validates incoming data against expected schema and statistical profiles.

    Initialized with a reference dataset to learn expected ranges and types.
    """

    def __init__(self, reference_data: pd.DataFrame, target_col: str = "sales"):
        self.target_col = target_col
        self.expected_columns = list(reference_data.columns)

        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        self.stats_bounds: dict[str, dict[str, float]] = {}
        for col in numeric_cols:
            values = reference_data[col].dropna()
            if len(values) == 0:
                continue
            self.stats_bounds[col] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "q01": float(values.quantile(0.01)),
                "q99": float(values.quantile(0.99)),
            }

    def validate(self, data: pd.DataFrame) -> QualityReport:
        """Run all quality checks and return a structured report."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Check for required columns
        missing_cols = set(self.expected_columns) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # 2. Check for null values
        null_counts = data.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            null_pct = (cols_with_nulls / len(data) * 100).round(1)
            for col, pct in null_pct.items():
                if pct > 50:
                    errors.append(f"Column '{col}' has {pct}% nulls")
                elif pct > 5:
                    warnings.append(f"Column '{col}' has {pct}% nulls")

        # 3. Check for out-of-range values
        for col, bounds in self.stats_bounds.items():
            if col not in data.columns:
                continue
            col_values = data[col].dropna()
            if len(col_values) == 0:
                continue
            col_min = float(col_values.min())
            col_max = float(col_values.max())
            historical_range = bounds["max"] - bounds["min"]
            if historical_range > 0:
                if col_min < bounds["min"] - 3 * historical_range:
                    warnings.append(
                        f"Column '{col}' min={col_min:.1f} is far below "
                        f"reference range [{bounds['min']:.1f}, {bounds['max']:.1f}]"
                    )
                if col_max > bounds["max"] + 3 * historical_range:
                    warnings.append(
                        f"Column '{col}' max={col_max:.1f} is far above "
                        f"reference range [{bounds['min']:.1f}, {bounds['max']:.1f}]"
                    )

        # 4. Check for duplicate dates
        if "date" in data.columns:
            dup_dates = data["date"].duplicated().sum()
            if dup_dates > 0:
                errors.append(f"Found {dup_dates} duplicate dates")

        # 5. Check minimum data size
        if len(data) < 7:
            warnings.append(
                f"Dataset has only {len(data)} rows, "
                f"may be insufficient for reliable analysis"
            )

        is_valid = len(errors) == 0

        stats = {
            "n_rows": len(data),
            "n_cols": len(data.columns),
            "null_rate": round(float(data.isnull().mean().mean()), 4),
        }

        report = QualityReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

        if not is_valid:
            logger.error("Data quality FAILED: %s", errors)
        elif warnings:
            logger.warning("Data quality PASSED with warnings: %s", warnings)
        else:
            logger.info("Data quality check passed")

        return report
