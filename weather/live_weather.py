from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


load_dotenv()


class LiveWeatherService:

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

        self.api_key = os.getenv(
            "OPENWEATHER_API_KEY",
            "",
        ).strip()

        self.lat = os.getenv(
            "FIREGUARD_LAT",
            "37.2",
        ).strip()

        self.lon = os.getenv(
            "FIREGUARD_LON",
            "51.5",
        ).strip()

    def get_weather(self) -> Dict[str, Any]:

        if not self.api_key:
            return {
                "available": False,
                "source": "openweathermap",
                "error": "OPENWEATHER_API_KEY is not configured",
                "data": None,
            }

        try:
            lat = float(self.lat)
            lon = float(self.lon)

        except Exception:
            return {
                "available": False,
                "source": "openweathermap",
                "error": "Invalid FIREGUARD_LAT or FIREGUARD_LON",
                "data": None,
            }

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()

            raw = response.json()

        except requests.RequestException as exc:
            return {
                "available": False,
                "source": "openweathermap",
                "error": f"Weather API unavailable: {exc}",
                "data": None,
            }

        except Exception as exc:
            return {
                "available": False,
                "source": "openweathermap",
                "error": f"Weather response error: {exc}",
                "data": None,
            }

        main = raw.get("main")

        if not isinstance(main, dict):
            return {
                "available": False,
                "source": "openweathermap",
                "error": "Weather API returned no valid main weather data",
                "data": None,
            }

        wind = raw.get("wind") or {}
        weather_list = raw.get("weather") or []

        description = None

        if weather_list and isinstance(weather_list[0], dict):
            description = weather_list[0].get("description")

        data = {
            "temperature": main.get("temp"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "description": description,
            "observed_at": datetime.now(timezone.utc),
            "latitude": lat,
            "longitude": lon,
        }

        # هیچ مقدار NaN یا Infinity را معتبر نمی‌دانیم
        for key, value in data.items():

            if isinstance(value, (int, float)):

                if not math.isfinite(float(value)):
                    return {
                        "available": False,
                        "source": "openweathermap",
                        "error": (
                            f"Invalid non-finite weather value: {key}"
                        ),
                        "data": None,
                    }

        return {
            "available": True,
            "source": "openweathermap",
            "error": None,
            "data": data,
            "raw": raw,
        }