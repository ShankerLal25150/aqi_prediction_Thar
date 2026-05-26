"""
Tharparkar AQI — Multi-Model Training Pipeline
==============================================
Trains Ridge, RandomForest, and XGBoost locally. 
Picks the best model, computes SHAP explainability, and saves the winner.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your custom preprocessing module
from preprocessing import (
    preprocess_features, 
    check_data_leakage, 
    check_r2_validity, 
    classification_report_from_aqi
)

def time_split(X, y, val_ratio: float = 0.15):
    """Split keeping time order intact. NEVER shuffle time-series data."""
    n = len(X)
    split = int(n * (1 - val_ratio))
    return X[:split], X[split:], y[:split], y[split:]

def save_shap_plot(model, X_sample: np.ndarray, feature_names: list, out_path: str):
    """Compute SHAP values and save a summary bar chart."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (What drives pollution?)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[shap] Saved → {out_path}")
    except Exception as e:
        print(f"[shap] Could not generate plot: {e}")

def main():
    print("="*60)
    print("  Tharparkar Multi-Model Tournament (24h Forecast)")
    print("="*60)

    # 1. Load Local Parquet Data
    df = pd.read_parquet("thar_historical_training_data.parquet")

    target_col = "aqi_plus_24h"
    drop_cols = ["timestamp", "location", "aqi_plus_24h", "aqi_plus_48h", "aqi_plus_72h"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    # 2. Leakage Check
    check_data_leakage(df, target_col, feature_cols)

    X_raw = df[feature_cols].values
    y = df[target_col].values

    # 3. Time-Series Split
    X_tr_raw, X_val_raw, y_train, y_val = time_split(X_raw, y, val_ratio=0.15)
    print(f"\n[split] Train: {len(y_train)} rows | Validation: {len(y_val)} rows")

    # 4. Preprocessing
    print("[preprocess] Applying RobustScaler and ±4 IQR clipping...")
    X_train_scaled, X_val_scaled, imputer, scaler = preprocess_features(X_tr_raw, X_val_raw)

    # 5. Define Competing Models
    models = {
        "Ridge_Baseline": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
    }

    results = {}

    # 6. Train and Evaluate Each Model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))
        
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "preds": preds}
        print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.2f}")

    # 7. Crown the Winner
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    print("\n" + "="*60)
    print(f" WINNER: {best_name} (Lowest RMSE: {results[best_name]['rmse']:.2f})")
    print("="*60)

    # 8. Classification Report for the Winner
    print("\n[Classification Report for AQI Categories]")
    print(classification_report_from_aqi(y_val, results[best_name]["preds"]))

    # 9. SHAP Explainability
    if best_name in ["RandomForest", "XGBoost"]:
        print("\nComputing SHAP values for the winning model...")
        save_shap_plot(best_model, X_val_scaled[:500], feature_cols, "shap_feature_importance.png")

    # 10. Save the Winning Assets for the Dashboard
    joblib.dump(best_model, "best_aqi_model_24h.joblib")
    joblib.dump(feature_cols, "feature_columns.joblib")
    joblib.dump(imputer, "imputer.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("\n[success] Saved winning model, scaler, and imputer to disk.")

if __name__ == "__main__":
    main()