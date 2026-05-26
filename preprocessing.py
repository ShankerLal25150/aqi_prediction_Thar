import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def preprocess_features(X_train: np.ndarray, X_val: np.ndarray) -> tuple:
    # Step 1 – Imputation
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_train)
    X_v  = imputer.transform(X_val)

    # Step 2 – Robust scaling
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v  = scaler.transform(X_v)

    # Step 3 – Clip extreme outliers
    X_tr = np.clip(X_tr, -4, 4)
    X_v  = np.clip(X_v,  -4, 4)

    return X_tr, X_v, imputer, scaler

def check_data_leakage(df: pd.DataFrame, target_col: str, feature_cols: list) -> None:
    print("\n[leakage_check] Scanning for potential data leakage …")
    found = False
    for col in feature_cols:
        if col not in df.columns:
            continue
        corr = df[col].corr(df[target_col])
        if abs(corr) > 0.98:
            print(f"  ⚠️  '{col}' → correlation with '{target_col}': {corr:.4f}  ← REMOVE THIS FEATURE")
            found = True
        elif abs(corr) > 0.90:
            print(f"  ⚠️  '{col}' → correlation with '{target_col}': {corr:.4f}  ← inspect carefully")
    if not found:
        print("  ✅ No suspicious leakage detected.")

def check_r2_validity(r2: float, model_name: str) -> None:
    if r2 >= 0.99:
        print(f"  ❌ [{model_name}] R²={r2:.4f} — TOO HIGH. Check for data leakage!")
    elif r2 >= 0.90:
        print(f"  ⚠️  [{model_name}] R²={r2:.4f} — High but plausible. Double-check val split.")
    elif r2 >= 0.50:
        print(f"  ✅ [{model_name}] R²={r2:.4f} — Good. Model explains majority of variance.")
    elif r2 >= 0.20:
        print(f"  ⚠️  [{model_name}] R²={r2:.4f} — Moderate. Consider more features or better model.")
    else:
        print(f"  ❌ [{model_name}] R²={r2:.4f} — Low. Model may be underfitting.")

def get_aqi_category(aqi: float) -> str:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown"
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy_Sensitive"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very_Unhealthy"
    return "Hazardous"

def classification_report_from_aqi(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import classification_report
    cats_true = [get_aqi_category(v) for v in y_true]
    cats_pred = [get_aqi_category(v) for v in y_pred]
    return classification_report(cats_true, cats_pred, zero_division=0)