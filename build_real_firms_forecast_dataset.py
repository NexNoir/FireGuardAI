
# -*- coding: utf-8 -*-
"""
FireGuard — Real FIRMS Forecast Dataset Builder
================================================

Purpose:
    Build a REAL retraining dataset from historical FIRMS detections.

Source:
    data/retraining/historical_fire_data_2001_2025.csv

Period:
    2001-2025 only

Targets:
    fire_next_24h
    fire_next_48h
    fire_next_72h

Target definition:
    A target is 1 only if another REAL FIRMS detection occurs
    AFTER the current detection, within the requested time horizon,
    and within the configured geographic radius.

IMPORTANT:
    - No synthetic sensor data.
    - No fabricated labels.
    - No modification of source dataset.
    - Targets are derived only from real FIRMS detections.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DATASET = (
    BASE_DIR
    / "data"
    / "retraining"
    / "historical_fire_data_2001_2025.csv"
)

OUTPUT_DATASET = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)

OUTPUT_REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025_report.json"
)

EARTH_RADIUS_KM = 6371.0088

# A future detection must occur within this radius
TARGET_RADIUS_KM = 10.0

HORIZONS = {
    "24h": pd.Timedelta(hours=24),
    "48h": pd.Timedelta(hours=48),
    "72h": pd.Timedelta(hours=72),
}

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
]


# ======================================================================
# PRINT HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def status(label: str, value) -> None:
    print(f"{label:<34}: {value}")


# ======================================================================
# LOAD DATA
# ======================================================================

def load_source() -> pd.DataFrame:

    section("LOADING REAL FIRMS DATASET")

    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE_DATASET}"
        )

    df = pd.read_csv(SOURCE_DATASET)

    status("Source", SOURCE_DATASET)
    status("Rows", f"{len(df):,}")
    status("Columns", len(df.columns))

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    return df


# ======================================================================
# BUILD EXACT TIMESTAMP
# ======================================================================

def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:

    section("BUILDING DETECTION TIMESTAMPS")

    out = df.copy()

    out["acq_date"] = pd.to_datetime(
        out["acq_date"],
        errors="coerce",
    )

    # FIRMS acq_time may lose leading zero when loaded as numeric.
    acq_time = (
        out["acq_time"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    hour = pd.to_numeric(
        acq_time.str[:2],
        errors="coerce",
    )

    minute = pd.to_numeric(
        acq_time.str[2:],
        errors="coerce",
    )

    out["detection_time"] = (
        out["acq_date"]
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )

    bad_time = int(
        out["detection_time"].isna().sum()
    )

    status("Invalid timestamps", bad_time)

    if bad_time > 0:
        out = out.dropna(
            subset=["detection_time"]
        ).copy()

    out = out.sort_values(
        "detection_time"
    ).reset_index(drop=True)

    status(
        "First detection",
        out["detection_time"].min()
    )

    status(
        "Last detection",
        out["detection_time"].max()
    )

    return out


# ======================================================================
# VALIDATE COORDINATES
# ======================================================================

def validate_coordinates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("VALIDATING COORDINATES")

    out = df.copy()

    out["latitude"] = pd.to_numeric(
        out["latitude"],
        errors="coerce",
    )

    out["longitude"] = pd.to_numeric(
        out["longitude"],
        errors="coerce",
    )

    valid = (
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    )

    removed = int((~valid).sum())

    status("Invalid coordinates removed", removed)

    out = out.loc[valid].copy()

    return out.reset_index(drop=True)


# ======================================================================
# HAVERSINE DISTANCE
# ======================================================================

def haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(
        np.sqrt(a)
    )

    return EARTH_RADIUS_KM * c


# ======================================================================
# BUILD REAL FUTURE TARGETS
# ======================================================================

def build_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("BUILDING REAL 24H / 48H / 72H TARGETS")

    out = df.copy()

    timestamps = out["detection_time"].to_numpy(
        dtype="datetime64[ns]"
    )

    latitudes = out["latitude"].to_numpy(
        dtype=float
    )

    longitudes = out["longitude"].to_numpy(
        dtype=float
    )

    n = len(out)

    max_horizon = max(HORIZONS.values())

    targets = {
        name: np.full(n, np.nan)
        for name in HORIZONS
    }

    future_candidate_counts = {
        name: 0
        for name in HORIZONS
    }

    section(
        f"TARGET RADIUS: {TARGET_RADIUS_KM} KM"
    )

    for i in range(n):

        current_time = timestamps[i]

        # Future starts strictly after current detection.
        start_idx = i + 1

        if start_idx >= n:
            continue

        # Find all rows within maximum horizon.
        max_end_time = (
            current_time
            + np.timedelta64(
                int(max_horizon.total_seconds()),
                "s",
            )
        )

        end_idx = np.searchsorted(
            timestamps,
            max_end_time,
            side="right",
        )

        # No complete 72h future interval:
        # keep targets NaN rather than fabricate negatives.
        if (
            timestamps[-1]
            < max_end_time
        ):
            continue

        if end_idx <= start_idx:

            for name in HORIZONS:
                targets[name][i] = 0

            continue

        candidate_lats = latitudes[
            start_idx:end_idx
        ]

        candidate_lons = longitudes[
            start_idx:end_idx
        ]

        candidate_times = timestamps[
            start_idx:end_idx
        ]

        distances = haversine_km(
            latitudes[i],
            longitudes[i],
            candidate_lats,
            candidate_lons,
        )

        for name, horizon in HORIZONS.items():

            horizon_end = (
                current_time
                + np.timedelta64(
                    int(horizon.total_seconds()),
                    "s",
                )
            )

            time_mask = (
                candidate_times
                <= horizon_end
            )

            if not np.any(time_mask):
                targets[name][i] = 0
                continue

            within_radius = np.any(
                distances[time_mask]
                <= TARGET_RADIUS_KM
            )

            targets[name][i] = int(
                within_radius
            )

            future_candidate_counts[name] += int(
                within_radius
            )

        if (
            (i + 1) % 500 == 0
            or (i + 1) == n
        ):
            print(
                f"Processed {i + 1:,} / {n:,}"
            )

    for name, values in targets.items():

        column = f"fire_next_{name}"

        out[column] = values

        print(
            f"Created: {column}"
        )

    return out


# ======================================================================
# ADD REAL TIME FEATURES
# ======================================================================

def add_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("ADDING FEATURES FROM REAL FIRMS COLUMNS")

    out = df.copy()

    out["hour"] = (
        out["detection_time"]
        .dt.hour
    )

    out["minute"] = (
        out["detection_time"]
        .dt.minute
    )

    out["day_of_week"] = (
        out["detection_time"]
        .dt.dayofweek
    )

    out["day_of_year"] = (
        out["detection_time"]
        .dt.dayofyear
    )

    # Cyclical seasonal features from real detection date.
    out["month_sin"] = np.sin(
        2 * np.pi * out["month"] / 12.0
    )

    out["month_cos"] = np.cos(
        2 * np.pi * out["month"] / 12.0
    )

    out["hour_sin"] = np.sin(
        2 * np.pi * out["hour"] / 24.0
    )

    out["hour_cos"] = np.cos(
        2 * np.pi * out["hour"] / 24.0
    )

    print("Created real temporal features.")

    return out


# ======================================================================
# REMOVE INCOMPLETE TARGET ROWS
# ======================================================================

def remove_incomplete_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("REMOVING INCOMPLETE FORECAST WINDOWS")

    out = df.copy()

    target_columns = [
        f"fire_next_{name}"
        for name in HORIZONS
    ]

    before = len(out)

    out = out.dropna(
        subset=target_columns
    ).copy()

    removed = before - len(out)

    status("Rows before", f"{before:,}")
    status("Rows removed", f"{removed:,}")
    status("Rows with complete targets", f"{len(out):,}")

    for column in target_columns:

        out[column] = (
            out[column]
            .astype(int)
        )

    return out.reset_index(drop=True)


# ======================================================================
# REPORT
# ======================================================================

def build_report(
    df: pd.DataFrame,
) -> dict:

    section("BUILDING REPORT")

    report = {
        "source_dataset": str(SOURCE_DATASET),
        "output_dataset": str(OUTPUT_DATASET),
        "source_modified": False,
        "fabricated_data": False,
        "target_definition": (
            "Another real FIRMS detection occurs after "
            "the current detection, within the requested "
            "time horizon and within TARGET_RADIUS_KM."
        ),
        "target_radius_km": TARGET_RADIUS_KM,
        "rows": int(len(df)),
        "period_start": str(
            df["detection_time"].min()
        ),
        "period_end": str(
            df["detection_time"].max()
        ),
        "targets": {},
    }

    for name in HORIZONS:

        column = f"fire_next_{name}"

        counts = (
            df[column]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        positives = int(
            (df[column] == 1).sum()
        )

        negatives = int(
            (df[column] == 0).sum()
        )

        positive_rate = (
            positives / len(df) * 100
            if len(df) > 0
            else 0
        )

        report["targets"][name] = {
            "positive": positives,
            "negative": negatives,
            "positive_rate_percent": round(
                positive_rate,
                4,
            ),
            "counts": {
                str(key): int(value)
                for key, value in counts.items()
            },
        }

    return report


# ======================================================================
# SAVE
# ======================================================================

def save_results(
    df: pd.DataFrame,
    report: dict,
) -> None:

    section("SAVING REAL FORECAST DATASET")

    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        OUTPUT_DATASET.resolve()
        == SOURCE_DATASET.resolve()
    ):
        raise RuntimeError(
            "Safety error: output cannot overwrite source."
        )

    df.to_csv(
        OUTPUT_DATASET,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    status("Dataset", OUTPUT_DATASET)
    status("Report", OUTPUT_REPORT)
    status("Rows", f"{len(df):,}")


# ======================================================================
# FINAL SUMMARY
# ======================================================================

def final_summary(
    df: pd.DataFrame,
) -> None:

    section("FINAL RESULT")

    print("SOURCE MODIFIED       : NO")
    print("SYNTHETIC DATA ADDED  : NO")
    print("FABRICATED LABELS     : NO")
    print(
        f"TARGET RADIUS         : "
        f"{TARGET_RADIUS_KM} KM"
    )

    print()

    for name in HORIZONS:

        column = f"fire_next_{name}"

        positives = int(
            (df[column] == 1).sum()
        )

        total = len(df)

        rate = (
            positives / total * 100
            if total
            else 0
        )

        print(
            f"{name.upper():<8}"
            f"positive: {positives:,} / "
            f"{total:,} "
            f"({rate:.2f}%)"
        )

    print()
    print("STATUS: PASS")
    print(
        "Real FIRMS forecast training dataset is ready."
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    section(
        "FIREGUARD — REAL FIRMS FORECAST DATASET BUILDER"
    )

    print("PERIOD             : 2001-2025")
    print("SOURCE MODIFIED    : NO")
    print("SYNTHETIC DATA     : NO")
    print("FABRICATED LABELS  : NO")

    # 1. Load
    df = load_source()

    # 2. Build exact timestamps
    df = build_timestamp(df)

    # 3. Validate coordinates
    df = validate_coordinates(df)

    # 4. Build targets from real future detections
    df = build_targets(df)

    # 5. Add features based only on real columns
    df = add_features(df)

    # 6. Keep only complete forecast windows
    df = remove_incomplete_targets(df)

    # 7. Build report
    report = build_report(df)

    # 8. Save
    save_results(df, report)

    # 9. Final result
    final_summary(df)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print("Interrupted by user.")

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 70)
        print("FATAL ERROR")
        print("=" * 70)
        print(str(exc))

        sys.exit(1)