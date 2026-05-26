"""
Tharparkar AQI Predictor — Streamlit Dashboard
==============================================
Pulls live data from MongoDB, applies local preprocessing, and runs XGBoost models.
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from dotenv import load_dotenv

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tharparkar AQI Dashboard", page_icon="🏭", layout="wide")
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

AQI_LEVELS = [
    (0,   50,  "Good",                        "#00e400", "Air quality is satisfactory. No restrictions needed."),
    (51,  100, "Moderate",                    "#ffff00", "Acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "Sensitive groups: limit prolonged outdoor activity."),
    (151, 200, "Unhealthy",                   "#ff0000", "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion."),
    (201, 300, "Very Unhealthy",              "#8f3f97", "Health alert: everyone may experience serious health effects. Avoid all outdoor activity."),
    (301, 500, "Hazardous",                   "#7e0023", "Health emergency — avoid ALL outdoor activity. Stay indoors and keep windows closed."),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def aqi_level(aqi: float):
    if aqi is None or pd.isna(aqi):
        return "Unknown", "#cccccc", "No data available."
    for lo, hi, label, color, desc in AQI_LEVELS:
        if lo <= aqi <= hi:
            return label, color, desc
    return "Hazardous", "#7e0023", "Avoid ALL outdoor activity."

def aqi_alert(aqi: float, label: str):
    """Show a red alert banner if AQI is hazardous."""
    if aqi >= 301:
        st.error(f"🚨 HAZARDOUS AIR QUALITY ALERT ({label}): AQI = {int(aqi)}. "
                 f"Stay indoors, keep windows closed, wear N95 if outside.")
    elif aqi >= 201:
        st.warning(f"⚠️ VERY UNHEALTHY AIR ({label}): AQI = {int(aqi)}. "
                   f"Avoid all outdoor activity.")
    elif aqi >= 151:
        st.warning(f"⚠️ UNHEALTHY AIR ({label}): AQI = {int(aqi)}. "
                   f"Sensitive groups should stay indoors.")

# ── Asset Loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Database...")
def get_mongo_client():
    return MongoClient(os.environ["MONGO_URI"])

@st.cache_resource(show_spinner="Loading AI Models...")
def load_ml_assets():
    try:
        model_24h = joblib.load("best_aqi_model_24h.joblib")
        model_48h = joblib.load("best_aqi_model_48h.joblib")
        model_72h = joblib.load("best_aqi_model_72h.joblib")
        features  = joblib.load("feature_columns.joblib")
        imputer   = joblib.load("imputer.joblib")
        scaler    = joblib.load("scaler.joblib")
        return model_24h, model_48h, model_72h, features, imputer, scaler
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Run model_training.py first.")
        st.stop()

# ── Reusable metric card ───────────────────────────────────────────────────────

def metric_card(title: str, aqi_val: float):
    cat, col, desc = aqi_level(aqi_val)
    st.markdown(f"""
    <div style="background:{col}20; border-left:5px solid {col};
                padding:20px; border-radius:10px; height:100%;">
        <p style="margin:0; font-size:13px; color:#555;">{title}</p>
        <h1 style="margin:0; font-size:42px;">{int(aqi_val)}</h1>
        <h3 style="margin:0; color:{col};">{cat}</h3>
        <p style="margin-top:8px; font-size:12px; color:#666;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.sidebar.title("🏭 Tharparkar AQI")
    location = st.sidebar.selectbox("Select Location", [
        "TharBlock_II_SECMC", "TharBlock_I_SSRL", "TharBlock_III", "Islamkot", "Mithi"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Live Database:** MongoDB Atlas")
    st.sidebar.markdown("**Model:** XGBoost")
    st.sidebar.markdown("**Forecast Horizon:** +72 Hours (3 Days)")

    st.title(f"🌍 Live AQI Forecast: {location}")
    st.caption(f"Last refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load Assets ───────────────────────────────────────────────────────────
    client = get_mongo_client()
    db   = client[os.environ.get("MONGO_DB",   "thar_aqi")]
    coll = db[os.environ.get("MONGO_COLL", "aqi_features")]

    model_24h, model_48h, model_72h, expected_features, imputer, scaler = load_ml_assets()

    # ── Fetch Latest Live Data ────────────────────────────────────────────────
    latest_record = coll.find_one({"location": location}, sort=[("timestamp", -1)])

    if not latest_record:
        st.warning(f"No live data found for {location}. Run feature_pipeline.py first.")
        st.stop()

    df_live     = pd.DataFrame([latest_record])
    current_aqi = float(df_live["aqi"].iloc[0]) if pd.notna(df_live["aqi"].iloc[0]) else 0.0

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_raw     = df_live[expected_features].values
    X_imputed = imputer.transform(X_raw)
    X_scaled  = scaler.transform(X_imputed)
    X_clipped = np.clip(X_scaled, -4, 4)

    # ── Predict all 3 horizons ────────────────────────────────────────────────
    pred_24h = max(0, round(float(model_24h.predict(X_clipped)[0]), 1))
    pred_48h = max(0, round(float(model_48h.predict(X_clipped)[0]), 1))
    pred_72h = max(0, round(float(model_72h.predict(X_clipped)[0]), 1))

    # ── Hazardous Alerts ──────────────────────────────────────────────────────
    aqi_alert(current_aqi, "Right Now")
    aqi_alert(pred_24h,    "+24h Forecast")
    aqi_alert(pred_48h,    "+48h Forecast")
    aqi_alert(pred_72h,    "+72h Forecast")

    # ── Metric Cards (4 columns) ──────────────────────────────────────────────
    st.markdown("### 📊 AQI Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Current AQI (Now)",      current_aqi)
    with c2: metric_card("Predicted AQI (+24h)",   pred_24h)
    with c3: metric_card("Predicted AQI (+48h)",   pred_48h)
    with c4: metric_card("Predicted AQI (+72h)",   pred_72h)

    st.markdown("---")

    # ── Pollutant Breakdown ───────────────────────────────────────────────────
    st.subheader("🧪 Current Pollutant Breakdown")
    poll_cols = {
        "PM2.5": "pm25", "PM10": "pm10",
        "NO₂":  "no2",  "SO₂":  "so2", "Dust": "dust"
    }
    poll_data = {
        display: [f"{df_live[col].iloc[0]:.2f}"]
        for display, col in poll_cols.items()
        if col in df_live.columns
    }
    st.dataframe(pd.DataFrame(poll_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 72h Forecast Chart ────────────────────────────────────────────────────
    st.subheader("📈 72-Hour AQI Forecast Trajectory")

    labels = ["Right Now", "Tomorrow (+24h)", "Day 2 (+48h)", "Day 3 (+72h)"]
    values = [current_aqi, pred_24h, pred_48h, pred_72h]
    colors = [aqi_level(v)[1] for v in values]

    fig = go.Figure()

    # Bar chart
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[int(v) for v in values],
        textposition="auto",
        name="AQI",
        width=0.5
    ))

    # Trend line on top
    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        mode="lines+markers",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=8, color="white"),
        name="Trend"
    ))

    # AQI threshold lines
    fig.add_hline(y=100, line_dash="dash", line_color="yellow",
                  annotation_text="Moderate (100)", annotation_position="top left")
    fig.add_hline(y=150, line_dash="dash", line_color="orange",
                  annotation_text="Unhealthy for Sensitive (150)", annotation_position="top left")
    fig.add_hline(y=200, line_dash="dash", line_color="red",
                  annotation_text="Unhealthy (200)", annotation_position="top left")

    fig.update_layout(
        yaxis_title="AQI",
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        font=dict(color="#333")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP Feature Importance ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 What's Driving the AQI? (SHAP Feature Importance)")
    shap_path = "shap_feature_importance_24h.png"
    if not os.path.exists(shap_path):
        shap_path = "shap_feature_importance.png"   # fallback to old file
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP values — features pushing AQI up or down", use_column_width=True)
    else:
        st.info("SHAP plot not found. Run model_training.py to generate it.")

if __name__ == "__main__":
    main()