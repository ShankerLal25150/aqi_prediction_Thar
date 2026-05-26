"""
Thar AQI Prediction 72-Hour Dashboard
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from dotenv import load_dotenv

st.set_page_config(page_title="Tharparkar AQI Dashboard", page_icon="🏭", layout="wide")
load_dotenv()

AQI_LEVELS = [
    (0,   50,  "Good", "#10b981", "Satisfactory. No restrictions."),
    (51,  100, "Moderate", "#d97706", "Acceptable. Sensitive people should reduce exertion."),
    (101, 150, "Unhealthy for Sensitive", "#ea580c", "Sensitive groups: limit prolonged outdoor activity."),
    (151, 200, "Unhealthy", "#ef4444", "Avoid prolonged outdoor exertion."),
    (201, 300, "Very Unhealthy", "#8b5cf6", "Health alert: avoid all outdoor activity."),
    (301, 500, "Hazardous", "#9f1239", "Health emergency: stay indoors."),
]

def aqi_level(aqi: float):
    if aqi is None or pd.isna(aqi): return "Unknown", "#cccccc", "No data available."
    for lo, hi, label, color, desc in AQI_LEVELS:
        if lo <= aqi <= hi: return label, color, desc
    return "Hazardous", "#7e0023", "Avoid ALL outdoor activity."

@st.cache_resource(show_spinner="Connecting to Database...")
def get_mongo_client():
    return MongoClient(os.environ["MONGO_URI"])

@st.cache_resource(show_spinner="Loading AI Models...")
def load_ml_assets():
    models = {
        24: joblib.load("best_aqi_model_24h.joblib"),
        48: joblib.load("best_aqi_model_48h.joblib"),
        72: joblib.load("best_aqi_model_72h.joblib")
    }
    features = joblib.load("feature_columns.joblib")
    imputer = joblib.load("imputer.joblib")
    scaler = joblib.load("scaler.joblib")
    return models, features, imputer, scaler

def main():
    st.sidebar.title("🏭 Tharparkar AQI")
    location = st.sidebar.selectbox("Select Location", [
        "TharBlock_II_SECMC", "TharBlock_I_SSRL", "TharBlock_III", "Islamkot", "Mithi"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Live Database:** MongoDB Atlas")
    st.sidebar.markdown("**Forecast Horizon:** +72 Hours")

    st.title(f"Live 72-Hour AQI Forecast: {location}")
    
    try:
        models, expected_features, imputer, scaler = load_ml_assets()
    except FileNotFoundError:
        st.error("Model files missing. Run model_training.py first.")
        st.stop()

    client = get_mongo_client()
    db = client[os.environ.get("MONGO_DB", "thar_aqi")]
    coll = db[os.environ.get("MONGO_COLL", "aqi_features")]
    
    latest_record = coll.find_one({"location": location}, sort=[("timestamp", -1)])
    if not latest_record:
        st.warning(f"No live data found for {location}.")
        st.stop()

    df_live = pd.DataFrame([latest_record])
    current_aqi = float(df_live["aqi"].iloc[0]) if pd.notna(df_live["aqi"].iloc[0]) else 0.0

    # Preprocess
    X_raw = df_live[expected_features].values
    X_imputed = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imputed)
    X_clipped = np.clip(X_scaled, -4, 4)

    # Predict all 3 horizons
    predictions = {}
    for h in [24, 48, 72]:
        val = float(models[h].predict(X_clipped)[0])
        predictions[h] = max(0, round(val, 1))

    # UI: Metric Cards (Fixed HTML Rendering - NO WHITESPACE)
    cols = st.columns(4)
    all_vals = [("Now", current_aqi)] + [(f"+{h}h", predictions[h]) for h in [24, 48, 72]]
    
    for col, (label, val) in zip(cols, all_vals):
        with col:
            cat, color, desc = aqi_level(val)
            # Using single-line string concatenation to prevent Markdown code-block rendering
            html_card = (
                f'<div style="background-color:#f9f9f9; border-left:5px solid {color}; padding:15px; border-radius:8px; min-height: 180px;">'
                f'<p style="margin:0; font-size:13px; color:#555; font-weight:bold;">{label}</p>'
                f'<h1 style="margin:5px 0; font-size:42px;">{int(val)}</h1>'
                f'<h4 style="margin:0; color:{color}; font-size:16px;">{cat}</h4>'
                f'<p style="margin-top:8px; font-size:11px; color:#666; line-height: 1.2;">{desc}</p>'
                f'</div>'
            )
            st.markdown(html_card, unsafe_allow_html=True)

    st.markdown("---")

    # UI: Plotly Chart
    st.subheader("3-Day Trajectory")
    
    times = ["Right Now", "Tomorrow (+24h)", "Day 3 (+48h)", "Day 4 (+72h)"]
    y_vals = [current_aqi, predictions[24], predictions[48], predictions[72]]
    colors = [aqi_level(v)[1] for v in y_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=y_vals,
        mode="lines+markers+text",
        line=dict(color="#333", width=2, dash="dot"),
        marker=dict(size=18, color=colors, line=dict(width=2, color="white")),
        text=[f"AQI {int(v)}" for v in y_vals],
        textposition="top center"
    ))
    
    # Add WHO background bands (Fixed Plotly Opacity Crash)
    for lo, hi, label, color, _ in AQI_LEVELS:
        fig.add_hrect(
            y0=lo, y1=hi, 
            fillcolor=color, 
            opacity=0.1,  
            line_width=0
        )

    fig.update_layout(yaxis_title="AQI", height=400, showlegend=False, yaxis=dict(range=[0, max(y_vals) + 50]))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()