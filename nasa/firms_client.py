from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "nasa" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT = DATA_DIR / "firms_live.csv"

load_dotenv(PROJECT_ROOT / ".env")

MAP_KEY = os.getenv("MAP_KEY", "").strip()

try:
    SENSOR_LAT = float(os.getenv("SENSOR_LAT", "37.0"))
    SENSOR_LON = float(os.getenv("SENSOR_LON", "50.0"))
except ValueError:
    SENSOR_LAT = 37.0
    SENSOR_LON = 50.0


# Hircanian study area from previous project
BBOX = "48.5,35.8,54.5,38.5"

# Keep this endpoint configurable.
FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{api_key}/VIIRS_NOAA20_NRT/{bbox}/1"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common FIRMS column names without inventing data.
    """
    rename_map = {}

    for col in df.columns:
        lower = str(col).strip().lower()

        aliases = {
            "lat": "latitude",
            "latitude": "latitude",
            "lon": "longitude",
            "longitude": "longitude",
            "acq_date": "acq_date",
            "acq_time": "acq_time",
            "confidence": "confidence",
            "frp": "frp",
            "type": "type",
        }

        if lower in aliases:
            rename_map[col] = aliases[lower]

    return df.rename(columns=rename_map)


def _add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build timestamp only from actual FIRMS date/time fields.
    No timestamp is fabricated.
    """
    df = df.copy()

    if "acq_date" in df.columns and "acq_time" in df.columns:
        date_part = df["acq_date"].astype(str).str.strip()
        time_part = (
            pd.to_numeric(df["acq_time"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )

        df["timestamp"] = pd.to_datetime(
            date_part + " " +
            time_part.str[:2] + ":" +
            time_part.str[2:],
            errors="coerce",
        )

    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )
    else:
        df["timestamp"] = pd.NaT

    return df


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Great-circle distance in kilometers.
    """
    r = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def _validate_firms_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate actual FIRMS observations.

    Invalid rows are removed.
    Missing scientific measurements are NOT fabricated.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df = _add_timestamp(df)

    required = {"latitude", "longitude"}

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    # Geographic sanity checks
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    # Study-area validation
    df = df[
        df["latitude"].between(35.8, 38.5)
        & df["longitude"].between(48.5, 54.5)
    ].copy()

    # Optional numeric fields
    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(
            df["frp"],
            errors="coerce",
        )

    # Distance is derived ONLY from actual latitude/longitude.
    df["distance_km"] = df.apply(
        lambda row: _haversine_km(
            SENSOR_LAT,
            SENSOR_LON,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# NASA FIRMS client
# ---------------------------------------------------------

def fetch_firms_data(
    timeout: int = 20,
) -> Optional[pd.DataFrame]:
    """
    Fetch real NASA FIRMS data.

    Returns:
        DataFrame with validated observations,
        or None when NASA is unavailable / invalid.

    This function never changes FireGuard model risk.
    """

    if not MAP_KEY:
        print("NASA FIRMS: API key unavailable.")
        return None

    url = FIRMS_URL.format(
        api_key=MAP_KEY,
        bbox=BBOX,
    )

    try:
        response = requests.get(
            url,
            timeout=timeout,
        )

        response.raise_for_status()

        # NASA should return CSV.
        from io import StringIO

        raw_df = pd.read_csv(
            StringIO(response.text)
        )

        if raw_df.empty:
            print("NASA FIRMS: valid response, zero detections.")
            return pd.DataFrame()

        validated = _validate_firms_data(
            raw_df
        )

        if validated.empty:
            print(
                "NASA FIRMS: response received, "
                "but no valid observations remained after validation."
            )
            return pd.DataFrame()

        return validated

    except requests.RequestException as exc:
        print(
            f"NASA FIRMS unavailable: {exc}"
        )
        return None

    except Exception as exc:
        print(
            f"NASA FIRMS processing failure: {exc}"
        )
        return None


# ---------------------------------------------------------
# Evidence summary
# ---------------------------------------------------------

def build_firms_evidence(
    firms_df: Optional[pd.DataFrame],
) -> dict:
    """
    Convert NASA observations into an evidence summary.

    This is NOT a model risk score.
    """

    if firms_df is None:
        return {
            "available": False,
            "detection_count": 0,
            "observations": [],
            "message": "NASA FIRMS unavailable",
        }

    if firms_df.empty:
        return {
            "available": True,
            "detection_count": 0,
            "observations": [],
            "message": "NASA FIRMS returned no valid detections",
        }

    observations = []

    for _, row in firms_df.iterrows():
        item = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "distance_km": float(row["distance_km"]),
        }

        if "confidence" in row.index:
            item["confidence"] = row["confidence"]

        if "frp" in row.index:
            if pd.notna(row["frp"]):
                item["frp"] = float(row["frp"])

        if "timestamp" in row.index:
            if pd.notna(row["timestamp"]):
                item["timestamp"] = (
                    row["timestamp"].isoformat()
                )

        observations.append(item)

    return {
        "available": True,
        "detection_count": len(observations),
        "observations": observations,
        "message": "NASA FIRMS evidence available",
    }


# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------

def save_firms_data(
    firms_df: Optional[pd.DataFrame],
    output_path: Path = RAW_OUTPUT,
) -> bool:
    """
    Persist validated NASA observations.

    Existing data is replaced by the latest validated snapshot.
    """
    if firms_df is None:
        return False

    try:
        firms_df.to_csv(
            output_path,
            index=False,
        )

        return True

    except Exception as exc:
        print(
            f"NASA FIRMS save failure: {exc}"
        )
        return False


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("NASA FIRMS REAL DATA TEST")
    print("=" * 70)

    print(
        f"SENSOR: {SENSOR_LAT}, {SENSOR_LON}"
    )

    if not MAP_KEY:
        print(
            "STATUS: FAIL — MAP_KEY not configured"
        )
        return 1

    df = fetch_firms_data()

    if df is None:
        print(
            "STATUS: NASA FIRMS unavailable"
        )
        print(
            "FireGuard can continue without NASA evidence."
        )
        return 0

    if df.empty:
        print(
            "STATUS: NASA reachable — "
            "zero valid detections"
        )
        return 0

    print(
        f"Detection count: {len(df)}"
    )

    print(
        "Columns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    saved = save_firms_data(df)

    print(
        f"Saved: {'YES' if saved else 'NO'}"
    )

    evidence = build_firms_evidence(df)

    print(
        f"Evidence available: "
        f"{evidence['available']}"
    )

    print(
        f"Detection count: "
        f"{evidence['detection_count']}"
    )

    print("=" * 70)
    print("NASA FIRMS TEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())