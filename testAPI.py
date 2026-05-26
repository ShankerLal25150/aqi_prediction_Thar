import requests
import pandas as pd
from datetime import datetime, timedelta

# ── The 5 Tharparkar locations ────────────────────────────────────────────────
# is_mine=True → active coal mine / is_mine=False → residential control town

THAR_LOCATIONS = {
    "TharBlock_II_SECMC":  {"lat": 24.812, "lon": 70.398, "is_mine": True,
                             "desc": "Pakistan's first active open-pit lignite mine (7.6MT/yr)"},
    "TharBlock_I_SSRL":    {"lat": 24.671, "lon": 70.327, "is_mine": True,
                             "desc": "Shanghai Electric / SSRL mine, operational 2023 (7.8MT/yr)"},
    "TharBlock_III":       {"lat": 24.750, "lon": 70.550, "is_mine": True,
                             "desc": "Newest open-pit coal mine in Thar"},
    "Islamkot":            {"lat": 24.701, "lon": 70.178, "is_mine": False,
                             "desc": "Nearest residential town to the mines"},
    "Mithi":               {"lat": 24.734, "lon": 69.798, "is_mine": False,
                             "desc": "District HQ — your home city (upwind control)"},
}


def fetch_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Open-Meteo Air Quality API — hourly air pollution data.
    """
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=pm2_5,pm10,sulphur_dioxide,nitrogen_dioxide,"
        "carbon_monoxide,ozone,european_aqi,dust"
        f"&start_date={start_date}&end_date={end_date}"
        "&timezone=Asia/Karachi"
    )
    print(f"  Calling: {url[:90]}...")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"API error {resp.status_code}: {resp.text}")
    data = resp.json()
    print(f"  Got {len(data['hourly']['time'])} hourly readings.")
    return data


def fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Open-Meteo Historical Archive — hourly weather.
    FREE. No key. Returns 2+ years of history in one call.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,surface_pressure,cloud_cover"
        "&wind_speed_unit=ms&timezone=UTC"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_air_quality(data: dict, location: str) -> pd.DataFrame:
    """Convert raw JSON to DataFrame."""
    h = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "pm25":      h["pm2_5"],
        "pm10":      h["pm10"],
        "so2":       h["sulphur_dioxide"],
        "no2":       h["nitrogen_dioxide"],
        "co":        h["carbon_monoxide"],
        "o3":        h["ozone"],
        "aqi":       h["european_aqi"],
        "dust":      h["dust"],
        "location":  location,
        "lat":       data["latitude"],
        "lon":       data["longitude"],
    })
    return df


def get_aqi_category(aqi: float) -> str:
    """US EPA / WHO AQI category labels used for classification."""
    if aqi is None or pd.isna(aqi): return "Unknown"
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy_Sensitive"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very_Unhealthy"
    return "Hazardous"


# ── Main test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  THARPARKAR AQI — API TEST")
    print("  Testing Open-Meteo for 5 Thar locations")
    print("=" * 65)

    # Test last 7 days (quick test)
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    all_dfs = []

    for loc_name, info in THAR_LOCATIONS.items():
        print(f"\n[{loc_name}]  {'MINE' if info['is_mine'] else 'TOWN'}")
        print(f"  {info['desc']}")
        try:
            raw  = fetch_air_quality(info["lat"], info["lon"], start_date, end_date)
            df   = parse_air_quality(raw, loc_name)
            df["is_mine"]    = info["is_mine"]
            df["aqi_category"] = df["aqi"].apply(get_aqi_category)
            all_dfs.append(df)

            print(f"  Avg AQI:  {df['aqi'].mean():.1f}  |  Max AQI: {df['aqi'].max():.0f}")
            print(f"  Avg PM2.5:{df['pm25'].mean():.1f}  |  Avg SO2: {df['so2'].mean():.2f}")
            print(f"  Category breakdown:")
            for cat, cnt in df["aqi_category"].value_counts().items():
                print(f"    {cat:25s} {cnt:4d} readings")

        except Exception as e:
            print(f"  ERROR: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        print("\n" + "=" * 65)
        print("  MINE vs RESIDENTIAL COMPARISON")
        print("=" * 65)
        mine_aqi = combined[combined["is_mine"]]["aqi"].mean()
        town_aqi = combined[~combined["is_mine"]]["aqi"].mean()
        print(f"  Average AQI at coal mines:  {mine_aqi:.1f}")
        print(f"  Average AQI at towns:       {town_aqi:.1f}")
        print(f"  Difference:                 {mine_aqi - town_aqi:+.1f}")

        combined.to_csv("thar_api_test.csv", index=False)
        print(f"\n  Saved test data → thar_api_test.csv")

    print("\n" + "=" * 65)
    print("  RESULT: If you see AQI numbers above, the API works!")
    print("  NEXT: python feature_pipeline.py")
    print("=" * 65)