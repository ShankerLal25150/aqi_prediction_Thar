# AQI Prediction Dashboard - displays live AQI data and predicts AQI levels for the next 72 hours

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import boto3

from pymongo import MongoClient
from dotenv import load_dotenv

st.set_page_config(
    page_title="Tharparkar AQI Dashboard",
    page_icon="🏭",
    layout="wide"
)

load_dotenv()

AQI_LEVELS = [
    (0, 50, "Good", "#10b981", "Satisfactory. No restrictions."),
    (51, 100, "Moderate", "#d97706", "Acceptable. Sensitive people should reduce exertion."),
    (101, 150, "Unhealthy for Sensitive", "#ea580c", "Sensitive groups should limit outdoor activity."),
    (151, 200, "Unhealthy", "#ef4444", "Avoid prolonged outdoor exertion."),
    (201, 300, "Very Unhealthy", "#8b5cf6", "Health alert. Avoid outdoor activity."),
    (301, 500, "Hazardous", "#9f1239", "Health emergency. Stay indoors.")
]

def aqi_level(aqi):
    if aqi is None or pd.isna(aqi):
        return "Unknown", "#cccccc", "No data available."

    for low, high, label, color, description in AQI_LEVELS:
        if low <= aqi <= high:
            return label, color, description

    return "Hazardous", "#7e0023", "Avoid outdoor activity."

@st.cache_resource(show_spinner="Connecting to database...")
def get_mongo_client():
    return MongoClient(os.environ["MONGO_URI"])

@st.cache_resource(show_spinner="Downloading models from AWS S3 Registry...")
def load_ml_assets():
    # 1. Connect to AWS via Streamlit Secrets
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    
    bucket_name = 'aqi-data-thar-2026'
    
    files_to_download = [
        "best_aqi_model_24h.joblib",
        "best_aqi_model_48h.joblib",
        "best_aqi_model_72h.joblib",
        "feature_columns.joblib",
        "imputer.joblib",
        "scaler.joblib"
    ]
    
    # 2. Download from AWS to Streamlit's temporary folder
    for file_name in files_to_download:
        s3_path = f"production_models/{file_name}"
        local_path = f"/tmp/{file_name}"
        
        # Only download if it isn't already cached locally
        if not os.path.exists(local_path):
            s3.download_file(bucket_name, s3_path, local_path)

    # 3. Load the artifacts into memory from the temp folder
    models = {
        24: joblib.load("/tmp/best_aqi_model_24h.joblib"),
        48: joblib.load("/tmp/best_aqi_model_48h.joblib"),
        72: joblib.load("/tmp/best_aqi_model_72h.joblib")
    }

    feature_columns = joblib.load("/tmp/feature_columns.joblib")
    imputer = joblib.load("/tmp/imputer.joblib")
    scaler = joblib.load("/tmp/scaler.joblib")

    return models, feature_columns, imputer, scaler

def main():
    st.sidebar.title("Tharparkar AQI")

    location = st.sidebar.selectbox(
        "Select Location",
        [
            "TharBlock_II_SECMC",
            "TharBlock_I_SSRL",
            "TharBlock_III",
            "Islamkot",
            "Mithi"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Database:** MongoDB Atlas")
    st.sidebar.markdown("**Forecast Horizon:** 72 Hours")

    st.sidebar.markdown("---")

    st.sidebar.subheader("System Architecture")

    st.sidebar.markdown(
        """
        **CI/CD Pipeline:** GitHub Actions automates
        hourly data ingestion and daily model retraining.

        **Model Registry:** Model artifacts are pulled dynamically 
        from **AWS S3 Cloud Storage**.

        **Environment:**
        - Python 3.10
        - Streamlit Community Cloud
        - Dependencies managed with `requirements.txt`
        """
    )

    st.title(f"Live 72-Hour AQI Forecast - {location}")

    try:
        models, expected_features, imputer, scaler = load_ml_assets()

    except Exception as e:
        st.error(f"Failed to load cloud models: {e}")
        st.stop()

    client = get_mongo_client()
    db = client[os.environ.get("MONGO_DB", "thar_aqi")]
    collection = db[os.environ.get("MONGO_COLL", "aqi_features")]

    latest_record = collection.find_one(
        {"location": location},
        sort=[("timestamp", -1)]
    )

    if not latest_record:
        st.warning(f"No live data found for {location}.")
        st.stop()

    live_df = pd.DataFrame([latest_record])

    current_aqi = (
        float(live_df["aqi"].iloc[0])
        if pd.notna(live_df["aqi"].iloc[0])
        else 0.0
    )

    X_raw = live_df[expected_features].values
    X_imputed = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imputed)
    X_processed = np.clip(X_scaled, -4, 4)

    predictions = {}

    for horizon in [24, 48, 72]:
        prediction = models[horizon].predict(X_processed)[0]
        predictions[horizon] = max(
            0,
            round(float(prediction), 1)
        )

    metric_columns = st.columns(4)

    values = [
        ("Now", current_aqi),
        ("+24h", predictions[24]),
        ("+48h", predictions[48]),
        ("+72h", predictions[72])
    ]

    for column, (label, value) in zip(metric_columns, values):
        with column:
            category, color, description = aqi_level(value)
            
            card_html = (
                f'<div style="'
                f'background-color:#f9f9f9; '
                f'border-left:5px solid {color}; '
                f'padding:15px; '
                f'border-radius:8px; '
                f'min-height:180px;">'
                
                f'<p style="'
                f'margin:0; '
                f'font-size:13px; '
                f'color:#555; '
                f'font-weight:bold;">'
                f'{label}'
                f'</p>'
                
                f'<h1 style="'
                f'margin:5px 0; '
                f'font-size:42px;">'
                f'{int(value)}'
                f'</h1>'
                
                f'<h4 style="'
                f'margin:0; '
                f'color:{color}; '
                f'font-size:16px;">'
                f'{category}'
                f'</h4>'
                
                f'<p style="'
                f'margin-top:8px; '
                f'font-size:11px; '
                f'color:#666; '
                f'line-height:1.2;">'
                f'{description}'
                f'</p>'
                
                f'</div>'
            )
            
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("3-Day AQI Forecast")

    x_values = ["Current", "+24h", "+48h", "+72h"]
    y_values = [current_aqi, predictions[24], predictions[48], predictions[72]]

    marker_colors = [aqi_level(value)[1] for value in y_values]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers+text",
            line=dict(
                color="#333",
                width=2,
                dash="dot"
            ),
            marker=dict(
                size=18,
                color=marker_colors,
                line=dict(
                    width=2,
                    color="white"
                )
            ),
            text=[f"AQI {int(value)}" for value in y_values],
            textposition="top center"
        )
    )

    for low, high, _, color, _ in AQI_LEVELS:
        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=color,
            opacity=0.1,
            line_width=0
        )

    fig.update_layout(
        yaxis_title="AQI",
        height=400,
        showlegend=False,
        yaxis=dict(
            range=[0, max(y_values) + 50]
        )
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()