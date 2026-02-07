import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChampionChallenger:
    def __init__(self, data: pd.DataFrame, target_col: str = 'sales', date_col: str = 'date'):
        self.data = data.sort_values(date_col)
        self.target_col = target_col
        self.date_col = date_col
        self.results = {}

    def evaluate_model(self, y_true, y_pred, model_name) -> dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        logging.info(f"[{model_name}] MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")
        return {'mae': mae, 'rmse': rmse, 'mape': mape}

    def train_baseline_sarima(self, train_data: pd.Series, test_steps: int) -> np.ndarray:
        # Simple SARIMA (1,1,1)(1,1,1,7) assumption for weekly seasonality
        # In a real scenario, we'd use auto_arima or grid search
        try:
            model =SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                           enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=test_steps)
            return forecast.values
        except Exception as e:
            logging.error(f"SARIMA failed: {e}")
            return np.zeros(test_steps)

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return preds

    def train_prophet(self, train_df: pd.DataFrame, test_steps: int) -> np.ndarray:
        # Prophet requires columns ['ds', 'y']
        df_prophet = train_df[[self.date_col, self.target_col]].rename(columns={self.date_col: 'ds', self.target_col: 'y'})

        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
        # Using suppress_stdout_stderr might be needed if Prophet is noisy, but logging handles it mostly
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=test_steps)
        forecast = m.predict(future)

        # Return only the last 'test_steps' predictions
        return forecast['yhat'].tail(test_steps).values

    def run_cv(self, n_splits=3) -> pd.DataFrame:
        logging.info(f"Running Cross-Validation with {n_splits} splits...")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        metrics = []

        feature_cols = [c for c in self.data.columns if c not in [self.target_col, self.date_col, 'date_add']]

        for fold, (train_index, test_index) in enumerate(tscv.split(self.data)):
            train_df = self.data.iloc[train_index]
            test_df = self.data.iloc[test_index]

            y_test = test_df[self.target_col].values
            test_steps = len(y_test)

            logging.info(f"Fold {fold+1}/{n_splits}: Train size={len(train_df)}, Test size={len(test_df)}")

            # 1. SARIMA
            # Needs only target series
            y_pred_sarima = self.train_baseline_sarima(train_df[self.target_col], test_steps)
            metrics.append({'model': 'SARIMA', 'fold': fold, **self.evaluate_model(y_test, y_pred_sarima, 'SARIMA')})

            # 2. XGBoost
            # Needs features
            X_train = train_df[feature_cols]
            y_train = train_df[self.target_col]
            X_test = test_df[feature_cols]

            # Handle potential NaNs if any legacy ones exist
            # XGBoost handles NaNs naturally
            y_pred_xgb = self.train_xgboost(X_train, y_train, X_test)
            metrics.append({'model': 'XGBoost', 'fold': fold, **self.evaluate_model(y_test, y_pred_xgb, 'XGBoost')})

            # 3. Prophet
            # Needs ds, y
            y_pred_prophet = self.train_prophet(train_df, test_steps)
            metrics.append({'model': 'Prophet', 'fold': fold, **self.evaluate_model(y_test, y_pred_prophet, 'Prophet')})

        return pd.DataFrame(metrics)

    def train_final_best_model(self, best_model_name: str, forecast_horizon: int = 14) -> Any:
        logging.info(f"Training final BEST model: {best_model_name} on ALL data...")

        feature_cols = [c for c in self.data.columns if c not in [self.target_col, self.date_col, 'date_add']]

        if best_model_name == 'SARIMA':
            model = SARIMAX(self.data[self.target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
            return model_fit # Can call .forecast(steps=N)

        elif best_model_name == 'XGBoost':
            model = xgb.XGBRegressor(n_estimators=100)
            model.fit(self.data[feature_cols], self.data[self.target_col])
            return model

        elif best_model_name == 'Prophet':
            df_prophet = self.data[[self.date_col, self.target_col]].rename(columns={self.date_col: 'ds', self.target_col: 'y'})
            m = Prophet()
            m.fit(df_prophet)
            return m

        return None

def main(input_path: str, output_model_dir: str, register_with_mlflow: bool = True):
    df = pd.read_parquet(input_path)

    # Initialize comparison
    arena = ChampionChallenger(df)

    # Run CV
    results_df = arena.run_cv(n_splits=3)

    # Aggregate results
    avg_results = results_df.groupby('model')['mae'].mean().sort_values()
    logging.info("\nAverage MAE per model:\n" + str(avg_results))

    best_model_name = avg_results.index[0]
    logging.info(f"Champion Model: {best_model_name}")

    # Train final model
    best_model = arena.train_final_best_model(best_model_name)

    # Save model
    Path(output_model_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_model_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    logging.info(f"Saved best model to {output_model_dir}/best_model.pkl")

    # Register with MLflow Model Registry
    if register_with_mlflow:
        try:
            import json

            from src.models.registry import ModelRegistry

            registry = ModelRegistry()

            metadata_path = Path(output_model_dir) / "champion_model_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            feature_cols = [
                c for c in df.columns
                if c not in [arena.target_col, arena.date_col, "date_add"]
            ]
            avg_metrics = (
                results_df[results_df["model"] == best_model_name]
                .mean(numeric_only=True)
                .to_dict()
            )

            version = registry.register_champion(
                model=best_model,
                metrics=avg_metrics,
                params=metadata.get("best_params", {"model_type": best_model_name}),
                features=feature_cols,
                train_date_range={
                    "start": str(df[arena.date_col].min()),
                    "end": str(df[arena.date_col].max()),
                },
                test_date_range={"start": "cv", "end": "cv"},
            )
            registry.transition_to_staging(version)
            logging.info(f"Model registered as v{version} in Staging")
        except Exception as e:
            logging.warning(f"MLflow registry unavailable, skipping: {e}")

    return best_model

if __name__ == "__main__":
    main("data/processed/features.parquet", "models")
