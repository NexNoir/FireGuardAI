from pathlib import Path
import time

import pandas as pd
import requests


# ============================================================
# FireGuard — Hyrcanian Historical Weather Downloader
# Real data only — no synthetic weather values
# ============================================================

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

OUTPUT_FILE = Path(
    "data/weather/historical/hyrcanian_weather_2020_2025.csv"
)

LOCATIONS = {
    "west_01": (37.2, 49.8),
    "west_02": (37.5, 49.9),
    "central_01": (36.9, 50.6),
    "central_02": (37.3, 51.0),
    "east_01": (37.0, 51.8),
    "east_02": (37.4, 52.2),
    "east_03": (37.6, 53.0),
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
]


def download_location(location_id: str, latitude: float, longitude: float):
    print()
    print("=" * 70)
    print(f"Downloading: {location_id}")
    print(f"Coordinates: {latitude}, {longitude}")
    print("=" * 70)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Tehran",

        # Explicit units — do not silently convert later.
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    try:
        response = requests.get(
            ARCHIVE_URL,
            params=params,
            timeout=120,
        )

        print(f"HTTP status: {response.status_code}")

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        print(f"ERROR: Weather API unavailable")
        print(exc)
        return None

    except ValueError as exc:
        print("ERROR: Invalid JSON response")
        print(exc)
        return None

    if "hourly" not in data:
        print("ERROR: API response has no hourly data")
        return None

    hourly = data["hourly"]

    if "time" not in hourly:
        print("ERROR: API response has no timestamps")
        return None

    rows = {
        "location_id": location_id,
        "timestamp": hourly.get("time", []),
        "latitude": [latitude] * len(hourly["time"]),
        "longitude": [longitude] * len(hourly["time"]),

        "temperature": hourly.get(
            "temperature_2m",
            [None] * len(hourly["time"])
        ),

        "humidity": hourly.get(
            "relative_humidity_2m",
            [None] * len(hourly["time"])
        ),

        "wind_speed": hourly.get(
            "wind_speed_10m",
            [None] * len(hourly["time"])
        ),

        "wind_direction": hourly.get(
            "wind_direction_10m",
            [None] * len(hourly["time"])
        ),

        "precipitation": hourly.get(
            "precipitation",
            [None] * len(hourly["time"])
        ),

        "pressure": hourly.get(
            "pressure_msl",
            [None] * len(hourly["time"])
        ),

        "cloud_cover": hourly.get(
            "cloud_cover",
            [None] * len(hourly["time"])
        ),

        "dew_point": hourly.get(
            "dew_point_2m",
            [None] * len(hourly["time"])
        ),
    }

    df = pd.DataFrame(rows)

    print(f"Rows: {len(df)}")

    if df.empty:
        print("Status: EMPTY")
        return None

    print("Status: OK")

    # Show missingness without filling anything.
    print("Missing values:")
    print(df.isna().sum())

    return df


def validate_dataset(df: pd.DataFrame):
    required_columns = [
        "location_id",
        "timestamp",
        "latitude",
        "longitude",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "pressure",
        "cloud_cover",
        "dew_point",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise RuntimeError(
            "Invalid timestamps detected."
        )

    # Remove only exact duplicate records.
    before = len(df)

    df = df.drop_duplicates(
        subset=["location_id", "timestamp"]
    )

    removed = before - len(df)

    if removed:
        print(f"Removed duplicate rows: {removed}")

    df = df.sort_values(
        ["location_id", "timestamp"]
    ).reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame):
    print()
    print("=" * 70)
    print("DATASET CREATED")
    print("=" * 70)

    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(
        f"Locations: {df['location_id'].nunique()}"
    )

    print(
        f"Start: {df['timestamp'].min()}"
    )

    print(
        f"End: {df['timestamp'].max()}"
    )

    print()
    print("Rows by location:")
    print(
        df["location_id"].value_counts().sort_index()
    )

    print()
    print("Missing values:")
    print(df.isna().sum())

    print()
    print("Columns:")

    for column in df.columns:
        print(f" - {column}")

    print()
    print("Sample:")
    print(df.head().to_string(index=False))


def main():
    print("=" * 70)
    print("🔥 FireGuard — HYRCANIAN WEATHER DATASET")
    print("=" * 70)

    print(f"Start: {START_DATE}")
    print(f"End  : {END_DATE}")
    print(f"Locations: {len(LOCATIONS)}")
    print("Model: ERA5-Land / Open-Meteo Archive")
    print("=" * 70)

    all_frames = []

    for location_id, (latitude, longitude) in LOCATIONS.items():

        df = download_location(
            location_id,
            latitude,
            longitude,
        )

        if df is not None and not df.empty:
            all_frames.append(df)

        # Avoid hammering the API.
        time.sleep(1)

    if not all_frames:
        print()
        print("=" * 70)
        print("DATASET NOT CREATED")
        print("=" * 70)
        print("No location returned valid weather data.")
        return

    combined = pd.concat(
        all_frames,
        ignore_index=True
    )

    try:
        combined = validate_dataset(combined)
    except Exception as exc:
        print()
        print("VALIDATION ERROR:")
        print(exc)
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print_summary(combined)

    print()
    print("=" * 70)
    print("STATUS: HISTORICAL WEATHER DATA READY")
    print("=" * 70)


if __name__ == "__main__":
    main()