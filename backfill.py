#Historical Backfill 2+ years of data for 5 locations

import argparse
import requests
import logging
import pandas as pd
from feature_pipeline import THAR_LOCATIONS, engineer_features

# Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_historical_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,surface_pressure,cloud_cover"
        "&wind_speed_unit=ms&timezone=UTC"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame({
        "timestamp":     pd.to_datetime(h["time"]),
        "temperature":   h["temperature_2m"],
        "humidity":      h["relative_humidity_2m"],
        "wind_speed":    h["wind_speed_10m"],
        "wind_dir":      h["wind_direction_10m"],
        "precipitation": h["precipitation"],
        "pressure":      h["surface_pressure"],
        "cloud_cover":   h["cloud_cover"],
    })

def fetch_historical_air_quality(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,"
        "sulphur_dioxide,european_aqi,dust"
        "&timezone=UTC"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "aqi":   h["european_aqi"],
        "pm25":  h["pm2_5"],
        "pm10":  h["pm10"],
        "no2":   h["nitrogen_dioxide"],
        "o3":    h["ozone"],
        "co":    h["carbon_monoxide"],
        "so2":   h["sulphur_dioxide"],
        "dust":  h["dust"],
    })

def process_location(loc_name: str, info: dict, start: str, end: str) -> pd.DataFrame:
    """Fetches and engineers data for a single location safely."""
    logging.info(f"[{loc_name}] Fetching {start} to {end}...")
    
    weather_df = fetch_historical_weather(info["lat"], info["lon"], start, end)
    aq_df      = fetch_historical_air_quality(info["lat"], info["lon"], start, end)
    
    merged = weather_df.merge(aq_df, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # 1. Apply exact same feature engineering as live pipeline
    records = []
    for _, row in merged.iterrows():
        raw = row.to_dict()
        ts  = raw.pop("timestamp")
        feat = engineer_features(raw, ts, loc_name, info["is_mine"])
        records.append(feat)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Sequential Features (Calculated per location to avoid cross-city leakage)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df["aqi_rolling_3h"]  = df["aqi"].rolling(3,  min_periods=1).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(24, min_periods=1).mean()

    # 3. Add Targets (What the model will try to predict)
    horizons = [24, 48, 72]
    for h in horizons:
        df[f"aqi_plus_{h}h"] = df["aqi"].shift(-h)

    # Drop the rows at the very end of the 2 years that have no future target
    target_cols = [f"aqi_plus_{h}h" for h in horizons]
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    # Drop rows where API gave us completely blank data
    threshold = 0.3 * df.shape[1]
    df = df.dropna(thresh=int(df.shape[1] - threshold)).reset_index(drop=True)
    
    logging.info(f"Engineered {len(df)} rows for {loc_name}.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Tharparkar AQI Backfill")
    parser.add_argument("--start", default="2022-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="YYYY-MM-DD")
    args = parser.parse_args()

    logging.info("=" * 65)
    logging.info(f"STARTING THARPARKAR BACKFILL: {args.start} -> {args.end}")
    logging.info("=" * 65)

    all_location_dfs = []
    
    for loc_name, info in THAR_LOCATIONS.items():
        try:
            loc_df = process_location(loc_name, info, args.start, args.end)
            all_location_dfs.append(loc_df)
        except Exception as e:
            logging.error(f" FAILED {loc_name}: {e}")

    # Combine all 5 locations into one master dataset
    master_df = pd.concat(all_location_dfs, ignore_index=True)
    
    # Save to Parquet
    out_file = "thar_historical_training_data.parquet"
    master_df.to_parquet(out_file, index=False)
    
    logging.info("=" * 65)
    logging.info("BACKFILL COMPLETE")
    logging.info(f"Total Rows: {len(master_df)}")
    logging.info(f"Columns:    {master_df.shape[1]}")
    logging.info(f"Saved To:   {out_file}")
    logging.info("=" * 65)

if __name__ == "__main__":
    main()