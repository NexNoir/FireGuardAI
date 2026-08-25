from __future__ import annotations

import csv
import io
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


load_dotenv()


class LiveFirmsService:

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    def __init__(self, timeout_seconds: float = 20.0):

        self.timeout_seconds = timeout_seconds

        self.map_key = os.getenv(
            "NASA_FIRMS_MAP_KEY",
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

        self.radius_km = os.getenv(
            "NASA_FIRMS_RADIUS_KM",
            "50",
        ).strip()

    def _build_area(self):

        lat = float(self.lat)
        lon = float(self.lon)
        radius_km = max(1.0, float(self.radius_km))

        # تقریبی:
        # هر درجه latitude حدود 111 کیلومتر است.
        delta_lat = radius_km / 111.0

        # برای longitude وابسته به latitude است.
        cos_lat = max(
            0.1,
            math.cos(math.radians(lat)),
        )

        delta_lon = radius_km / (111.0 * cos_lat)

        west = lon - delta_lon
        south = lat - delta_lat
        east = lon + delta_lon
        north = lat + delta_lat

        return f"{west},{south},{east},{north}"

    def fetch(self) -> Dict[str, Any]:

        if not self.map_key:

            return {
                "available": False,
                "source": "NASA FIRMS",
                "error": "NASA_FIRMS_MAP_KEY is not configured",
                "observations": [],
            }

        try:
            area = self._build_area()

        except Exception as exc:

            return {
                "available": False,
                "source": "NASA FIRMS",
                "error": f"Invalid FIRMS coordinates: {exc}",
                "observations": [],
            }

        # day range = 1
        url = (
            f"{self.BASE_URL}/"
            f"{self.map_key}/"
            f"VIIRS_SNPP_NRT/"
            f"{area}/"
            f"1"
        )

        try:

            response = requests.get(
                url,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()

            text = response.text.strip()

        except requests.RequestException as exc:

            return {
                "available": False,
                "source": "NASA FIRMS",
                "error": f"NASA FIRMS unavailable: {exc}",
                "observations": [],
            }

        except Exception as exc:

            return {
                "available": False,
                "source": "NASA FIRMS",
                "error": f"NASA FIRMS request failed: {exc}",
                "observations": [],
            }

        if not text:

            return {
                "available": True,
                "source": "NASA FIRMS",
                "error": None,
                "observations": [],
                "checked_at": datetime.now(timezone.utc),
            }

        try:

            rows = list(
                csv.DictReader(
                    io.StringIO(text)
                )
            )

        except Exception as exc:

            return {
                "available": False,
                "source": "NASA FIRMS",
                "error": f"NASA CSV parse error: {exc}",
                "observations": [],
            }

        observations: List[Dict[str, Any]] = []

        for row in rows:

            try:

                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))

            except (TypeError, ValueError):

                continue

            observation = {
                "latitude": lat,
                "longitude": lon,
                "frp": row.get("frp"),
                "confidence": row.get("confidence"),
                "brightness": row.get("bright_ti4")
                or row.get("brightness"),
                "acq_date": row.get("acq_date"),
                "acq_time": row.get("acq_time"),
                "satellite": row.get("satellite"),
                "instrument": row.get("instrument"),
                "source": "NASA FIRMS",
            }

            observations.append(observation)

        return {
            "available": True,
            "source": "NASA FIRMS",
            "error": None,
            "observations": observations,
            "checked_at": datetime.now(timezone.utc),
        }