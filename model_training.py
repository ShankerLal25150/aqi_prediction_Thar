"""
AQI Model Training -> trains models for
24, 48 and 72 hours for  aqi prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
import boto3
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from preprocessing import preprocess_features, check_data_leakage

# Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def time_split(X, y, val_ratio=0.15):
    n = len(X)
    split_index = int(n * (1 - val_ratio))

    X_train = X[:split_index]
    X_val = X[split_index:]

    y_train = y[:split_index]
    y_val = y[split_index:]

    return X_train, X_val, y_train, y_val


def main():
    logging.info("=" * 60)
    logging.info("Tharparkar AQI Model Training")
    logging.info("=" * 60)

    df = pd.read_parquet("thar_historical_training_data.parquet")

    horizons = [24, 48, 72]
    preprocessing_saved = False

    for horizon in horizons:
        logging.info(f"Training models for +{horizon} hour prediction")

        target_col = f"aqi_plus_{horizon}h"

        drop_cols = [
            "timestamp",
            "location",
            "aqi_plus_24h",
            "aqi_plus_48h",
            "aqi_plus_72h"
        ]

        feature_cols = [
            col for col in df.columns
            if col not in drop_cols
        ]

        X_raw = df[feature_cols].values
        y = df[target_col].values

        X_train_raw, X_val_raw, y_train, y_val = time_split(
            X_raw,
            y,
            val_ratio=0.15
        )

        X_train, X_val, imputer, scaler = preprocess_features(
            X_train_raw,
            X_val_raw
        )

        models = {
            "Ridge": Ridge(alpha=1.0),

            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            ),

            "XGBoost": xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1,
                random_state=42
            )
        }

        results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            rmse = np.sqrt(
                mean_squared_error(y_val, predictions)
            )

            results[model_name] = {
                "model": model,
                "rmse": float(rmse)
            }

            logging.info(f"{model_name} RMSE: {rmse:.2f}")

        best_model_name = min(
            results,
            key=lambda name: results[name]["rmse"]
        )

        best_model = results[best_model_name]["model"]

        logging.info(f"Best model for +{horizon}h: {best_model_name}")

        model_path = f"best_aqi_model_{horizon}h.joblib"
        joblib.dump(best_model, model_path)

        if not preprocessing_saved:
            joblib.dump(feature_cols, "feature_columns.joblib")
            joblib.dump(imputer, "imputer.joblib")
            joblib.dump(scaler, "scaler.joblib")

            preprocessing_saved = True

    logging.info("Local training complete. Initiating AWS S3 Model Registry Upload...")
    
    # AWS S3 Upload Logic
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
        bucket_name = 'aqi-data-thar-2026'

        artifacts_to_upload = [
            "best_aqi_model_24h.joblib",
            "best_aqi_model_48h.joblib",
            "best_aqi_model_72h.joblib",
            "feature_columns.joblib",
            "imputer.joblib",
            "scaler.joblib"
        ]

        for artifact in artifacts_to_upload:
            if os.path.exists(artifact):
                s3_path = f"production_models/{artifact}"
                s3.upload_file(artifact, bucket_name, s3_path)
                logging.info(f"Successfully uploaded {artifact} to AWS S3.")
            else:
                logging.warning(f"File {artifact} not found locally. Skipping upload.")

        logging.info("All artifacts successfully pushed to AWS S3 Registry.")

    except Exception as e:
        logging.error(f"Failed to upload models to AWS S3: {e}")

if __name__ == "__main__":
    main()