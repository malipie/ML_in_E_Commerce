"""FastAPI application for Time Series Forecasting with Prometheus monitoring."""

import logging
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import settings
from src.serving.auth import verify_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Rate Limiter ───────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Prometheus Metrics ──────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "forecast_requests_total",
    "Total forecast requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "forecast_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTION_VALUE = Histogram(
    "forecast_prediction_value",
    "Distribution of prediction values",
    buckets=[0, 500, 1000, 2000, 5000, 10000, 20000, 50000],
)
MODEL_LOAD_GAUGE = Gauge(
    "model_loaded",
    "Whether a model is currently loaded (1=yes, 0=no)",
)
PREDICTION_ERRORS = Counter(
    "forecast_prediction_errors_total",
    "Total prediction errors",
)

# ── Global State ────────────────────────────────────────────────────

_model = None
_model_info: dict = {}
_features_df: pd.DataFrame | None = None
_start_time = time.time()


def load_model():
    """Load model from MLflow registry, falling back to local pickle."""
    global _model, _model_info

    # Try MLflow registry first
    try:
        from src.models.registry import ModelRegistry

        registry = ModelRegistry()
        loaded = registry.load_production_model()
        if loaded is not None:
            _model = loaded
            info = registry.get_production_version_info()
            _model_info = info or {}
            _model_info["source"] = "mlflow_registry"
            MODEL_LOAD_GAUGE.set(1)
            logger.info("Model loaded from MLflow Production registry")
            return
    except Exception as e:
        logger.warning("MLflow registry unavailable: %s", e)

    # Fallback to local pickle
    model_path = settings.get_champion_model_path()
    try:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        _model_info = {"source": "local_pickle", "path": str(model_path)}
        MODEL_LOAD_GAUGE.set(1)
        logger.info("Model loaded from %s", model_path)
    except FileNotFoundError:
        # Try alternative path (best_model.pkl)
        alt_path = settings.models_dir / "best_model.pkl"
        try:
            with open(alt_path, "rb") as f:
                _model = pickle.load(f)
            _model_info = {"source": "local_pickle", "path": str(alt_path)}
            MODEL_LOAD_GAUGE.set(1)
            logger.info("Model loaded from %s", alt_path)
        except Exception as e2:
            logger.error("Failed to load model: %s", e2)
            MODEL_LOAD_GAUGE.set(0)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        MODEL_LOAD_GAUGE.set(0)


def load_features():
    """Load reference feature data for feature engineering context."""
    global _features_df
    features_path = settings.get_features_path()
    try:
        _features_df = pd.read_parquet(features_path)
        logger.info("Features data loaded from %s (%d rows)", features_path, len(_features_df))
    except Exception as e:
        logger.warning("Failed to load features data: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    load_model()
    load_features()
    yield


app = FastAPI(
    title="Time Series Forecasting API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Rate Limiter Registration ──────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=90)
    start_date: str | None = None

    @field_validator("start_date")
    @classmethod
    def validate_date(cls, v):
        if v is not None:
            try:
                pd.to_datetime(v)
            except Exception as exc:
                raise ValueError(f"Invalid date format: {v}") from exc
        return v


class ForecastItem(BaseModel):
    date: str
    forecast: float


class ForecastResponse(BaseModel):
    forecasts: list[ForecastItem]
    model_version: str
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_info: dict


# ── Middleware ───────────────────────────────────────────────────────

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    return response


# ── Public Endpoints (no auth) ─────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Time Series Forecasting API", "status": "running", "version": "2.0.0"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_source=_model_info.get("source", "none"),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics")
def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Protected Endpoints (require API key) ──────────────────────────

@app.get("/model-info", response_model=ModelInfoResponse, dependencies=[Depends(verify_api_key)])
def get_model_info():
    return ModelInfoResponse(model_info=_model_info)


@app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def forecast(request: Request, body: ForecastRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_date = (
        pd.to_datetime(body.start_date)
        if body.start_date
        else pd.Timestamp.today().normalize()
    )
    future_dates = pd.date_range(start=start_date, periods=body.days_ahead, freq="D")

    try:
        future_df = _build_serving_features(future_dates)

        # Use model's expected features if available, else infer
        if hasattr(_model, "feature_names_in_"):
            feature_cols = list(_model.feature_names_in_)
        elif hasattr(_model, "get_booster") and _model.get_booster().feature_names:
            feature_cols = _model.get_booster().feature_names
        else:
            feature_cols = [
                c for c in future_df.columns
                if c not in ["sales", "date", "anomaly_score", "is_anomaly"]
            ]

        # Fill any missing features with 0
        for col in feature_cols:
            if col not in future_df.columns:
                future_df[col] = 0.0

        predictions = _model.predict(future_df[feature_cols])

        for pred in predictions:
            PREDICTION_VALUE.observe(float(pred))

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e

    forecasts = [
        ForecastItem(date=str(d.date()), forecast=round(float(p), 2))
        for d, p in zip(future_dates, predictions, strict=False)
    ]

    return ForecastResponse(
        forecasts=forecasts,
        model_version=_model_info.get("version", "local"),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/reload-model", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def reload_model(request: Request):
    """Hot-reload the model without restarting the service."""
    load_model()
    return {"status": "reloaded", "model_info": _model_info}


def _build_serving_features(future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build feature DataFrame for future dates, using reference data for lags.

    Produces features compatible with both notebook-trained models
    (weekday_sin, lag_1, etc.) and pipeline-trained models (day_of_week,
    sales_lag_1, etc.).
    """
    import numpy as np

    if _features_df is None:
        raise ValueError("Features reference data not available")

    last_row = _features_df.iloc[-1]
    last_sales = last_row.get("sales", 0.0)
    rows = []

    for date in future_dates:
        feat = {}
        feat["date"] = date
        dow = date.dayofweek
        month = date.month
        day_of_month = date.day
        day_of_year = date.timetuple().tm_yday
        week_of_year = date.isocalendar()[1]

        # Pipeline features (build_features.py style)
        feat["day_of_week"] = dow
        feat["day_of_month"] = day_of_month
        feat["month"] = month
        feat["quarter"] = date.quarter
        feat["year"] = date.year
        feat["is_weekend"] = 1 if dow >= 5 else 0
        feat["is_holiday"] = 0
        feat["is_payday_1st"] = 1 if day_of_month in (1, 2, 3) else 0
        feat["is_payday_10th"] = 1 if day_of_month in (10, 11, 12) else 0
        feat["is_payday"] = 1 if feat["is_payday_1st"] or feat["is_payday_10th"] else 0

        # Pipeline lag features
        feat["sales_lag_1"] = last_sales
        feat["sales_lag_7"] = last_sales
        feat["rolling_mean_7"] = last_sales

        # Notebook features (advanced feature set)
        feat["sales_original"] = last_sales
        feat["weekday"] = dow
        feat["week_of_year"] = week_of_year
        feat["day_of_year"] = day_of_year

        # Cyclic encoding
        feat["weekday_sin"] = np.sin(2 * np.pi * dow / 7)
        feat["weekday_cos"] = np.cos(2 * np.pi * dow / 7)
        feat["month_sin"] = np.sin(2 * np.pi * month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * month / 12)
        feat["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
        feat["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

        # Notebook lag features (carry forward last known values)
        for lag in [1, 7, 14, 21, 28]:
            col = f"lag_{lag}"
            feat[col] = last_row.get(col, last_sales)

        # Rolling features (carry forward from reference)
        for window in [7, 14, 28]:
            for stat in ["mean", "std", "min", "max"]:
                col = f"rolling_{stat}_{window}"
                feat[col] = last_row.get(col, last_sales if stat == "mean" else 0.0)
        feat["expanding_mean"] = last_row.get("expanding_mean", last_sales)

        rows.append(feat)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
