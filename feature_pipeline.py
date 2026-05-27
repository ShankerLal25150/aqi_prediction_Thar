# Feature Pipeline fetches AQI and weather data from Open-Meteo and stores features in MongoDB.

import os
import math
import argparse
import datetime
import logging

import requests
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


MONGO_URI = os.environ["MONGO_URI"]

MONGO_DB = os.environ.get(
    "MONGO_DB",
    "thar_aqi"
)

MONGO_COLL = os.environ.get(
    "MONGO_COLL",
    "aqi_features"
)


THAR_LOCATIONS = {
    "TharBlock_II_SECMC": {
        "lat": 24.812,
        "lon": 70.398,
        "is_mine": True
    },

    "TharBlock_I_SSRL": {
        "lat": 24.671,
        "lon": 70.327,
        "is_mine": True
    },

    "TharBlock_III": {
        "lat": 24.750,
        "lon": 70.550,
        "is_mine": True
    },

    "Islamkot": {
        "lat": 24.701,
        "lon": 70.178,
        "is_mine": False
    },

    "Mithi": {
        "lat": 24.734,
        "lon": 69.798,
        "is_mine": False
    }
}


def fetch_current_air_quality(lat, lon):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&current=pm2_5,pm10,sulphur_dioxide,"
        "nitrogen_dioxide,carbon_monoxide,"
        "ozone,european_aqi,dust"
        "&timezone=Asia/Karachi"
    )

    response = requests.get(
        url,
        timeout=15
    )

    response.raise_for_status()

    current = response.json()["current"]

    return {
        "aqi": current.get("european_aqi"),
        "pm25": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "so2": current.get("sulphur_dioxide"),
        "no2": current.get("nitrogen_dioxide"),
        "co": current.get("carbon_monoxide"),
        "o3": current.get("ozone"),
        "dust": current.get("dust")
    }


def fetch_current_weather(lat, lon):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "precipitation,"
        "surface_pressure,"
        "cloud_cover"
        "&wind_speed_unit=ms"
        "&timezone=Asia/Karachi"
    )

    response = requests.get(
        url,
        timeout=15
    )

    response.raise_for_status()

    current = response.json()["current"]

    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_dir": current.get("wind_direction_10m"),
        "precipitation": current.get("precipitation"),
        "pressure": current.get("surface_pressure"),
        "cloud_cover": current.get("cloud_cover")
    }


def engineer_features(raw, now, location, is_mine):

    features = {
        **raw,
        "location": location,
        "is_mine": int(is_mine)
    }

    features["timestamp"] = now.replace(
        second=0,
        microsecond=0
    )

    features["hour"] = now.hour

    features["day"] = now.day

    features["month"] = now.month

    features["weekday"] = now.weekday()

    features["is_weekend"] = int(
        now.weekday() >= 5
    )

    features["day_of_year"] = (
        now.timetuple().tm_yday
    )

    features["quarter"] = (
        (now.month - 1) // 3 + 1
    )

    features["hour_sin"] = math.sin(
        2 * math.pi * now.hour / 24
    )

    features["hour_cos"] = math.cos(
        2 * math.pi * now.hour / 24
    )

    features["month_sin"] = math.sin(
        2 * math.pi * now.month / 12
    )

    features["month_cos"] = math.cos(
        2 * math.pi * now.month / 12
    )

    features["day_sin"] = math.sin(
        2 * math.pi * features["day_of_year"] / 365
    )

    features["day_cos"] = math.cos(
        2 * math.pi * features["day_of_year"] / 365
    )

    features["is_dust_season"] = int(
        now.month in [4, 5, 6]
    )

    pm25 = features.get("pm25") or 0

    pm10 = features.get("pm10") or 0

    features["pm_ratio"] = (
        round(pm25 / pm10, 4)
        if pm10 > 0
        else None
    )

    temperature = features.get("temperature") or 0

    humidity = features.get("humidity") or 0

    features["heat_humidity_index"] = round(
        temperature * humidity / 100,
        2
    )

    wind_speed = features.get("wind_speed") or 0

    features["stagnation_index"] = round(
        1.0 / (wind_speed + 0.1),
        4
    )

    features["aqi_change_rate"] = 0.0

    features["aqi_rolling_3h"] = (
        features.get("aqi") or 0
    )

    features["aqi_rolling_24h"] = (
        features.get("aqi") or 0
    )

    features.pop("precipitation", None)

    features.pop("cloud_cover", None)

    return features


def get_mongo_collection():

    client = MongoClient(MONGO_URI)

    database = client[MONGO_DB]

    collection = database[MONGO_COLL]

    collection.create_index(
        [
            ("location", 1),
            ("timestamp", 1)
        ],
        unique=True
    )

    return collection


def push_to_mongodb(records):

    collection = get_mongo_collection()

    operations = []

    for record in records:

        record_copy = {**record}

        if isinstance(
            record_copy.get("timestamp"),
            datetime.datetime
        ):
            record_copy["timestamp"] = (
                record_copy["timestamp"].isoformat()
            )

        operations.append(
            UpdateOne(
                {
                    "location": record_copy["location"],
                    "timestamp": record_copy["timestamp"]
                },
                {
                    "$set": record_copy
                },
                upsert=True
            )
        )

    if operations:

        result = collection.bulk_write(
            operations
        )

        logging.info(
            f"Upserted {result.upserted_count} new records "
            f"and modified {result.modified_count} existing records."
        )


def run_pipeline(locations):

    now = datetime.datetime.utcnow()

    logging.info(
        f"Pipeline started at "
        f"{now.strftime('%Y-%m-%d %H:%M')} UTC"
    )

    logging.info(
        f"Fetching data for {len(locations)} locations"
    )

    records = []

    for location_name, info in locations.items():

        try:
            air_quality = fetch_current_air_quality(
                info["lat"],
                info["lon"]
            )

            weather = fetch_current_weather(
                info["lat"],
                info["lon"]
            )

            raw_data = {
                **air_quality,
                **weather
            }

            features = engineer_features(
                raw_data,
                now,
                location_name,
                info["is_mine"]
            )

            records.append(features)

            logging.info(
                f"{location_name} | "
                f"AQI={air_quality.get('aqi')} | "
                f"PM2.5={air_quality.get('pm25')} | "
                f"SO2={air_quality.get('so2')}"
            )

        except Exception as error:

            logging.error(
                f"{location_name} failed: {error}"
            )

    if records:

        push_to_mongodb(records)

        logging.info(
            f"{len(records)} records pushed to "
            f"{MONGO_DB}.{MONGO_COLL}"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Tharparkar AQI feature pipeline"
    )

    parser.add_argument(
        "--location",
        default=None,
        help="Run pipeline for a single location"
    )

    args = parser.parse_args()

    if args.location:

        if args.location not in THAR_LOCATIONS:

            logging.error(
                f"Unknown location. Choose from: "
                f"{list(THAR_LOCATIONS.keys())}"
            )

            return

        selected_locations = {
            args.location: THAR_LOCATIONS[args.location]
        }

    else:

        selected_locations = THAR_LOCATIONS

    run_pipeline(selected_locations)


if __name__ == "__main__":
    main()