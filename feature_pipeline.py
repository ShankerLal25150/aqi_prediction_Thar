"""
Feature Pipeline 
Fetches current AQI + weather from Open-Meteo and Stores to MongoDB (your existing setup)

"""

import os
import math
import argparse
import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()

# Config
MONGO_URI     = os.environ["MONGO_URI"]           # from .env
MONGO_DB      = os.environ.get("MONGO_DB",   "thar_aqi")
MONGO_COLL    = os.environ.get("MONGO_COLL", "aqi_features")

THAR_LOCATIONS = {
    "TharBlock_II_SECMC": {"lat": 24.812, "lon": 70.398, "is_mine": True},
    "TharBlock_I_SSRL":   {"lat": 24.671, "lon": 70.327, "is_mine": True},
    "TharBlock_III":      {"lat": 24.750, "lon": 70.550, "is_mine": True},
    "Islamkot":           {"lat": 24.701, "lon": 70.178, "is_mine": False},
    "Mithi":              {"lat": 24.734, "lon": 69.798, "is_mine": False},
}


# API fetching

def fetch_current_air_quality(lat: float, lon: float) -> dict:
    
    # Open-Meteo Air Quality API — CURRENT conditions.
   
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&current=pm2_5,pm10,sulphur_dioxide,nitrogen_dioxide,"
        "carbon_monoxide,ozone,european_aqi,dust"
        "&timezone=Asia/Karachi"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    c = resp.json()["current"]
    return {
        "aqi":  c.get("european_aqi"),
        "pm25": c.get("pm2_5"),
        "pm10": c.get("pm10"),
        "so2":  c.get("sulphur_dioxide"),
        "no2":  c.get("nitrogen_dioxide"),
        "co":   c.get("carbon_monoxide"),
        "o3":   c.get("ozone"),
        "dust": c.get("dust"),
    }


def fetch_current_weather(lat: float, lon: float) -> dict:
   
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,surface_pressure,cloud_cover"
        "&wind_speed_unit=ms&timezone=Asia/Karachi"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    c = resp.json()["current"]
    return {
        "temperature":   c.get("temperature_2m"),
        "humidity":      c.get("relative_humidity_2m"),
        "wind_speed":    c.get("wind_speed_10m"),
        "wind_dir":      c.get("wind_direction_10m"),
        "precipitation": c.get("precipitation"),
        "pressure":      c.get("surface_pressure"),
        "cloud_cover":   c.get("cloud_cover"),
    }


# Feature engineering
def engineer_features(raw: dict, now: datetime.datetime, location: str, is_mine: bool) -> dict:
    feat = {**raw, "location": location, "is_mine": int(is_mine)}
    feat["timestamp"] = now.replace(second=0, microsecond=0)

    # -Time features
    feat["hour"]        = now.hour
    feat["day"]         = now.day
    feat["month"]       = now.month
    feat["weekday"]     = now.weekday()
    feat["is_weekend"]  = int(now.weekday() >= 5)
    feat["day_of_year"] = now.timetuple().tm_yday
    feat["quarter"]     = (now.month - 1) // 3 + 1

    # Cyclical encoding 
    feat["hour_sin"]  = math.sin(2 * math.pi * now.hour   / 24)
    feat["hour_cos"]  = math.cos(2 * math.pi * now.hour   / 24)
    feat["month_sin"] = math.sin(2 * math.pi * now.month  / 12)
    feat["month_cos"] = math.cos(2 * math.pi * now.month  / 12)
    feat["day_sin"]   = math.sin(2 * math.pi * feat["day_of_year"] / 365)
    feat["day_cos"]   = math.cos(2 * math.pi * feat["day_of_year"] / 365)

    # Season flag 
    feat["is_dust_season"] = int(now.month in [4, 5, 6])

    #  pollution features
    pm25 = feat.get("pm25") or 0
    pm10 = feat.get("pm10") or 0
    # PM ratio (higher → more harmful fine particles)
    feat["pm_ratio"] = round(pm25 / pm10, 4) if pm10 > 0 else None
    # Heat-humidity index (hot + humid → worse AQI)
    temp = feat.get("temperature") or 0
    hum  = feat.get("humidity") or 0
    feat["heat_humidity_index"] = round(temp * hum / 100, 2)
    # low wind 
    ws = feat.get("wind_speed") or 0
    feat["stagnation_index"] = round(1.0 / (ws + 0.1), 4)
    # AQI change rate 
    feat["aqi_change_rate"]  = 0.0
    # Rolling averages
    feat["aqi_rolling_3h"]   = feat.get("aqi") or 0
    feat["aqi_rolling_24h"]  = feat.get("aqi") or 0

    return feat


#  MongoDB storage
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    coll   = db[MONGO_COLL]
    # Ensure index on location, timestamp
    coll.create_index([("location", 1), ("timestamp", 1)], unique=True)
    return coll


def push_to_mongodb(records: list) -> None:
    coll = get_mongo_collection()
    ops  = []
    for rec in records:
        # Convert timestamp to string
        rec_copy = {**rec}
        if isinstance(rec_copy.get("timestamp"), datetime.datetime):
            rec_copy["timestamp"] = rec_copy["timestamp"].isoformat()
        ops.append(UpdateOne(
            {"location": rec_copy["location"], "timestamp": rec_copy["timestamp"]},
            {"$set": rec_copy},
            upsert=True,
        ))
    if ops:
        result = coll.bulk_write(ops)
        print(f"[MongoDB] Upserted {result.upserted_count} new + "
              f"modified {result.modified_count} existing records.")


# Main 

def run_pipeline(locations: dict) -> None:
    now = datetime.datetime.utcnow()
    print(f"\n[feature_pipeline] {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Fetching {len(locations)} location(s) from Open-Meteo …\n")

    records = []
    for loc_name, info in locations.items():
        try:
            aq      = fetch_current_air_quality(info["lat"], info["lon"])
            weather = fetch_current_weather(info["lat"], info["lon"])
            raw     = {**aq, **weather}
            feat    = engineer_features(raw, now, loc_name, info["is_mine"])
            records.append(feat)
            print(f"  ✅ {loc_name:25s}  AQI={aq.get('aqi')}  "
                  f"PM2.5={aq.get('pm25')}  SO2={aq.get('so2')}")
        except Exception as e:
            print(f"  ❌ {loc_name}: {e}")

    if records:
        push_to_mongodb(records)
        print(f"\n[done] {len(records)} records pushed to MongoDB '{MONGO_DB}.{MONGO_COLL}'")


def main():
    parser = argparse.ArgumentParser(description="Tharparkar AQI feature pipeline")
    parser.add_argument("--location", default=None,
                        help="Run single location only (e.g. --location Mithi)")
    args = parser.parse_args()

    if args.location:
        if args.location not in THAR_LOCATIONS:
            print(f"Unknown location. Choose from: {list(THAR_LOCATIONS.keys())}")
            return
        locs = {args.location: THAR_LOCATIONS[args.location]}
    else:
        locs = THAR_LOCATIONS

    run_pipeline(locs)


if __name__ == "__main__":
    main()