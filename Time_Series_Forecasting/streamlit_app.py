import os

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_URL = os.getenv("TSF_API_URL", "http://localhost:8000")
API_KEY = os.getenv("TSF_API_KEY_SECRET", "change-me-in-production")

st.title("Time Series Forecasting Dashboard")

st.sidebar.header("Settings")
days_ahead = st.sidebar.slider("Forecast Days Ahead", 1, 30, 7)
api_key_input = st.sidebar.text_input("API Key", value=API_KEY, type="password")

if st.button("Generate Forecast"):
    try:
        response = requests.post(
            f"{API_URL}/forecast",
            json={"days_ahead": days_ahead},
            headers={"X-API-Key": api_key_input},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            forecasts = pd.DataFrame(data["forecasts"])
            forecasts["date"] = pd.to_datetime(forecasts["date"])

            st.subheader("Forecast Results")
            fig = px.line(forecasts, x="date", y="forecast", title="Sales Forecast")
            st.plotly_chart(fig)

            st.caption(f"Model version: {data.get('model_version', 'unknown')} | Generated: {data.get('generated_at', '')}")
        elif response.status_code == 401:
            st.error("Invalid API key. Check the key in the sidebar.")
        elif response.status_code == 429:
            st.warning("Rate limit exceeded. Please wait a moment.")
        else:
            st.error(f"API error {response.status_code}: {response.text}")
    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running on port 8000:\n\n`uvicorn src.serving.app:app --reload`")
    except Exception as e:
        st.error(f"Error: {e}")

# Health check indicator
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    if health.status_code == 200:
        info = health.json()
        st.sidebar.success(f"API: {info['status']}")
        st.sidebar.caption(f"Model: {info['model_source']} | Uptime: {info['uptime_seconds']}s")
    else:
        st.sidebar.warning("API unhealthy")
except Exception:
    st.sidebar.error("API offline")

st.markdown("---")
st.markdown("Built with Streamlit for interactive ML demos.")
