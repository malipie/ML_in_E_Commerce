"""Feature engineering module for transaction-level clustering.

Transforms a raw ETL DataFrame into a scaled feature matrix suitable for
clustering algorithms. The pipeline applies:
1. Log1p transformation on skewed numerical features.
2. Sin/cos encoding for cyclical features (hour, day of week).
3. One-hot encoding for categorical features.
4. StandardScaler normalization on the final feature set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — cyclical periods
# ---------------------------------------------------------------------------

CYCLICAL_PERIODS: dict[str, int] = {
    "hour_of_day": 24,
    "day_of_week": 7,
}


# ---------------------------------------------------------------------------
# Pure transformation functions
# ---------------------------------------------------------------------------


def apply_log_transform(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Apply ``log1p`` transformation to specified columns.

    Uses ``np.log1p(x)`` which equals ``ln(1 + x)``.  This is safe for
    zero values (log1p(0) = 0) and handles the right-skew typical of
    monetary features.

    Args:
        df: Input DataFrame (not mutated).
        columns: Column names to transform.

    Returns:
        DataFrame with transformed columns (copies made, originals intact).
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found — skipping log transform.", col)
            continue
        df[col] = np.log1p(df[col].astype(float))
        logger.debug("Applied log1p to '%s'.", col)
    return df


def encode_cyclical(
    df: pd.DataFrame, column: str, period: int
) -> pd.DataFrame:
    """Encode a cyclical feature as sin/cos pair.

    Converts a periodic integer feature (e.g. hour 0–23) into two
    continuous dimensions that preserve the cyclical distance:

        sin_col = sin(2π · x / period)
        cos_col = cos(2π · x / period)

    The original column is dropped after encoding.

    Args:
        df: Input DataFrame (not mutated).
        column: Name of the cyclical column.
        period: Full period of the cycle (24 for hours, 7 for days).

    Returns:
        DataFrame with ``{column}_sin`` and ``{column}_cos`` columns added,
        original column removed.
    """
    df = df.copy()
    if column not in df.columns:
        logger.warning("Column '%s' not found — skipping cyclical encoding.", column)
        return df

    values = df[column].astype(float)
    angle = 2 * np.pi * values / period
    df[f"{column}_sin"] = np.sin(angle)
    df[f"{column}_cos"] = np.cos(angle)
    df = df.drop(columns=[column])
    logger.debug("Encoded '%s' as sin/cos (period=%d).", column, period)
    return df


def encode_all_cyclical(
    df: pd.DataFrame,
    cyclical_config: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Encode all cyclical features using sin/cos pairs.

    Args:
        df: Input DataFrame.
        cyclical_config: Mapping ``{column_name: period}``.
            Defaults to ``CYCLICAL_PERIODS``.

    Returns:
        DataFrame with cyclical columns replaced by sin/cos pairs.
    """
    if cyclical_config is None:
        cyclical_config = CYCLICAL_PERIODS
    for col, period in cyclical_config.items():
        df = encode_cyclical(df, col, period)
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: list[str],
    categories_map: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode categorical columns.

    When ``categories_map`` is ``None`` (fit mode), categories are learned
    from the data.  When provided (transform mode), only the known
    categories are used — unseen values produce all-zero rows.

    Args:
        df: Input DataFrame (not mutated).
        columns: Categorical column names to encode.
        categories_map: Pre-learned categories per column (for transform
            consistency). ``None`` means fit mode.

    Returns:
        Tuple of (encoded DataFrame, learned categories_map).
    """
    df = df.copy()
    learned: dict[str, list[str]] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found — skipping one-hot.", col)
            continue

        if categories_map and col in categories_map:
            # Transform mode — use known categories
            known_cats = categories_map[col]
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            # Ensure all known columns exist
            for cat_col in known_cats:
                if cat_col not in dummies.columns:
                    dummies[cat_col] = 0.0
            # Keep only known columns in consistent order
            dummies = dummies[[c for c in known_cats if c in dummies.columns]]
            learned[col] = known_cats
        else:
            # Fit mode — learn categories
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            learned[col] = sorted(dummies.columns.tolist())
            dummies = dummies[learned[col]]

        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        logger.debug("One-hot encoded '%s' → %d columns.", col, len(dummies.columns))

    return df, learned


# ---------------------------------------------------------------------------
# Feature pipeline — stateful (fit/transform)
# ---------------------------------------------------------------------------


@dataclass
class FeaturePipeline:
    """Stateful feature engineering pipeline.

    Encapsulates the full transformation chain (log → cyclical → one-hot →
    scale) and stores fitted state for consistent re-application.

    Attributes:
        numerical_cols: Columns to keep as-is (after optional log).
        log_transform_cols: Subset of numerical to log1p-transform.
        cyclical_config: ``{column: period}`` for sin/cos encoding.
        categorical_cols: Columns for one-hot encoding.
        scaler: Fitted ``StandardScaler`` (populated after ``fit``).
        categories_map: Learned one-hot categories (populated after ``fit``).
        feature_names: Final feature column order (populated after ``fit``).
    """

    numerical_cols: list[str] = field(default_factory=list)
    log_transform_cols: list[str] = field(default_factory=list)
    cyclical_config: dict[str, int] = field(default_factory=lambda: dict(CYCLICAL_PERIODS))
    categorical_cols: list[str] = field(default_factory=list)

    # --- Fitted state (populated by fit / fit_transform) ---
    scaler: StandardScaler | None = field(default=None, repr=False)
    categories_map: dict[str, list[str]] = field(default_factory=dict, repr=False)
    feature_names: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeaturePipeline:
        """Construct a pipeline from a parsed ``config.yaml`` features block.

        Args:
            config: The ``features`` section of the YAML config.

        Returns:
            Configured (unfitted) ``FeaturePipeline``.
        """
        cyclical_cols = config.get("cyclical", [])
        cyclical_config = {
            col: CYCLICAL_PERIODS.get(col, 24) for col in cyclical_cols
        }
        return cls(
            numerical_cols=config.get("numerical", []),
            log_transform_cols=config.get("log_transform", []),
            cyclical_config=cyclical_config,
            categorical_cols=config.get("categorical", []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_and_transform(
        self,
        df: pd.DataFrame,
        fit: bool,
    ) -> pd.DataFrame:
        """Apply all transformations except scaling.

        Args:
            df: Raw ETL DataFrame.
            fit: If ``True``, learn categories from data.

        Returns:
            Transformed DataFrame (pre-scaling).
        """
        # 1. Select numerical columns and apply log1p
        available_num = [c for c in self.numerical_cols if c in df.columns]
        result = df[available_num].copy()

        log_cols = [c for c in self.log_transform_cols if c in result.columns]
        if log_cols:
            result = apply_log_transform(result, log_cols)

        # 2. Cyclical encoding
        cyclical_cols_present = {
            c: p for c, p in self.cyclical_config.items() if c in df.columns
        }
        if cyclical_cols_present:
            cyclical_df = df[list(cyclical_cols_present.keys())].copy()
            cyclical_df = encode_all_cyclical(cyclical_df, cyclical_cols_present)
            result = pd.concat([result, cyclical_df], axis=1)

        # 3. Categorical encoding
        cat_cols_present = [c for c in self.categorical_cols if c in df.columns]
        if cat_cols_present:
            cat_df = df[cat_cols_present].copy()
            cat_df, learned = encode_categorical(
                cat_df,
                cat_cols_present,
                categories_map=None if fit else self.categories_map,
            )
            if fit:
                self.categories_map = learned
            result = pd.concat([result, cat_df], axis=1)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        """Learn transformation parameters from training data.

        Fits the scaler and learns one-hot categories without returning
        the transformed result.

        Args:
            df: Raw ETL DataFrame.

        Returns:
            ``self`` (for method chaining).
        """
        transformed = self._select_and_transform(df, fit=True)
        self.feature_names = transformed.columns.tolist()

        self.scaler = StandardScaler()
        self.scaler.fit(transformed.values)

        logger.info(
            "FeaturePipeline fitted: %d features learned.", len(self.feature_names)
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline and return a scaled numpy array.

        Args:
            df: Raw ETL DataFrame.

        Returns:
            2-D numpy array of shape ``(n_samples, n_features)``, scaled.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self.scaler is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        transformed = self._select_and_transform(df, fit=False)

        # Ensure column order matches fit-time
        for col in self.feature_names:
            if col not in transformed.columns:
                transformed[col] = 0.0
        transformed = transformed[self.feature_names]

        scaled: np.ndarray = self.scaler.transform(transformed.values)
        logger.info("Transformed %d samples × %d features.", *scaled.shape)
        return scaled

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the pipeline and return the scaled feature matrix.

        Convenience method combining ``fit`` and ``transform``.

        Args:
            df: Raw ETL DataFrame.

        Returns:
            2-D numpy array of shape ``(n_samples, n_features)``, scaled.
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_dataframe(
        self, scaled_array: np.ndarray
    ) -> pd.DataFrame:
        """Wrap a scaled numpy array back into a named DataFrame.

        Useful for inspection, debugging, and visualization.

        Args:
            scaled_array: Output of ``transform`` or ``fit_transform``.

        Returns:
            DataFrame with named columns matching ``feature_names``.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If array shape doesn't match feature count.
        """
        if not self.feature_names:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        if scaled_array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} columns, "
                f"got {scaled_array.shape[1]}."
            )
        return pd.DataFrame(scaled_array, columns=self.feature_names)
