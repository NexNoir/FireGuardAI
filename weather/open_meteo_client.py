from pathlib import Path
from typing import Optional
import requests
import pandas as pd


BASE_URL = "https://api.open-meteo.com/v1/forecast"

SENSOR_LAT = 37.0
SENSOR_LON = 50.0

DATA_DIR = Path("data/weather")
CURRENT_DIR = DATA_DIR / "current"
FORECAST_DIR = DATA_DIR / "forecast"
HISTORICAL_DIR = DATA_DIR / "historical"

for directory in [CURRENT_DIR, FORECAST_DIR, HISTORICAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


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


def _request(params: dict) -> Optional[dict]:
    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            return None

        if "hourly" not in data:
            return None

        return data

    except requests.RequestException as exc:
        print(f"Weather API unavailable: {exc}")
        return None

    except (ValueError, TypeError) as exc:
        print(f"Invalid Weather API response: {exc}")
        return None


def _hourly_to_dataframe(data: dict) -> Optional[pd.DataFrame]:
    hourly = data.get("hourly")

    if not hourly or "time" not in hourly:
        return None

    df = pd.DataFrame(hourly)

    if df.empty:
        return None

    rename_map = {
        "time": "timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dew_point",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "precipitation": "precipitation",
        "pressure_msl": "pressure",
        "cloud_cover": "cloud_cover",
    }

    df = df.rename(columns=rename_map)

    df["latitude"] = SENSOR_LAT
    df["longitude"] = SENSOR_LON

    required = [
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

    missing = [column for column in required if column not in df.columns]

    if missing:
        print(f"Missing Weather fields: {missing}")
        return None

    return df[required]


def fetch_current_weather() -> Optional[pd.DataFrame]:
    """
    Real current weather.
    No fake/default weather values.
    """

    params = {
        "latitude": SENSOR_LAT,
        "longitude": SENSOR_LON,
        "current": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",
    }

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        current = data.get("current")

        if not current:
            return None

        row = {
            "timestamp": current.get("time"),
            "latitude": SENSOR_LAT,
            "longitude": SENSOR_LON,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "precipitation": current.get("precipitation"),
            "pressure": current.get("pressure_msl"),
            "cloud_cover": current.get("cloud_cover"),
            "dew_point": current.get("dew_point_2m"),
        }

        return pd.DataFrame([row])

    except requests.RequestException as exc:
        print(f"Current Weather unavailable: {exc}")
        return None

    except (ValueError, TypeError) as exc:
        print(f"Invalid Current Weather response: {exc}")
        return None


def fetch_forecast_weather(hours: int = 72) -> Optional[pd.DataFrame]:
    """
    Real future weather forecast.
    Used separately from current weather.
    """

    hours = max(1, min(hours, 384))

    params = {
        "latitude": SENSOR_LAT,
        "longitude": SENSOR_LON,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": hours,
        "timezone": "auto",
    }

    data = _request(params)

    if data is None:
        return None

    return _hourly_to_dataframe(data)


def fetch_historical_weather(
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Historical weather for training.

    IMPORTANT:
    This function only requests the specified historical dates.
    It does not use future weather.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": SENSOR_LAT,
        "longitude": SENSOR_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            return None

        return _hourly_to_dataframe(data)

    except requests.RequestException as exc:
        print(f"Historical Weather unavailable: {exc}")
        return None

    except (ValueError, TypeError) as exc:
        print(f"Invalid Historical Weather response: {exc}")
        return None


def save_current_weather(df: pd.DataFrame) -> Path:
    path = CURRENT_DIR / "latest.csv"
    df.to_csv(path, index=False)
    return path


def save_forecast_weather(df: pd.DataFrame) -> Path:
    path = FORECAST_DIR / "latest.csv"
    df.to_csv(path, index=False)
    return path


def save_historical_weather(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Path:
    path = HISTORICAL_DIR / f"weather_{start_date}_{end_date}.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("=" * 70)
    print("FireGuard Weather Client")
    print("=" * 70)
    print(f"SENSOR: {SENSOR_LAT}, {SENSOR_LON}")

    current = fetch_current_weather()

    if current is not None and not current.empty:
        print("Current Weather: OK")
        print(current.to_string(index=False))
        print(f"Saved: {save_current_weather(current)}")
    else:
        print("Current Weather unavailable")

    forecast = fetch_forecast_weather(72)

    if forecast is not None and not forecast.empty:
        print(f"Forecast Weather: OK ({len(forecast)} rows)")
        print(f"Saved: {save_forecast_weather(forecast)}")
    else:
        print("Forecast Weather unavailable")

    print("=" * 70)