# -*- coding: utf-8 -*-
"""
FireGuard — V4 Feature Reconstruction Audit
============================================

Purpose:
    Reconstruct the engineered V4 features from the raw FireGuard dataset
    WITHOUT modifying the original dataset or V4 models.

Input:
    data/fireguard_forecast_60days.csv

Raw source columns required:
    timestamp
    temperature
    humidity
    smoke

Optional:
    flame
    fire_now

Output:
    data/fireguard_forecast_60days_v4_features_audit.csv

IMPORTANT:
    - No fabricated values.
    - No model modification.
    - No original dataset modification.
    - No forward filling / backward filling.
    - Initial windows remain NaN when insufficient history exists.
    - Timestamp spacing is validated.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "data" / "fireguard_forecast_60days.csv"

OUTPUT_DATASET = (
    BASE_DIR
    / "data"
    / "fireguard_forecast_60days_v4_features_audit.csv"
)

MODEL_DIR = BASE_DIR / "saved_models"


MODEL_SPECS = {
    ("sensor_only", 24):
        MODEL_DIR / "fireguard_forecast_sensor_only_24h_v4.joblib",

    ("sensor_only", 48):
        MODEL_DIR / "fireguard_forecast_sensor_only_48h_v4.joblib",

    ("sensor_only", 72):
        MODEL_DIR / "fireguard_forecast_sensor_only_72h_v4.joblib",

    ("sensor_plus_flame", 24):
        MODEL_DIR / "fireguard_forecast_sensor_plus_flame_24h_v4.joblib",

    ("sensor_plus_flame", 48):
        MODEL_DIR / "fireguard_forecast_sensor_plus_flame_48h_v4.joblib",

    ("sensor_plus_flame", 72):
        MODEL_DIR / "fireguard_forecast_sensor_plus_flame_72h_v4.joblib",
}


RAW_REQUIRED = [
    "timestamp",
    "temperature",
    "humidity",
    "smoke",
]


V4_EXPECTED_ENGINEERED = [
    "smoke_change_1m",
    "smoke_change_5m",
    "smoke_change_15m",
    "smoke_change_30m",
    "smoke_change_60m",

    "temperature_change_5m",
    "temperature_change_15m",
    "temperature_change_30m",
    "temperature_change_60m",

    "humidity_change_5m",
    "humidity_change_15m",
    "humidity_change_30m",

    "smoke_mean_5m",
    "smoke_mean_15m",
    "smoke_mean_30m",
    "smoke_mean_60m",

    "smoke_std_15m",
    "smoke_std_30m",
    "smoke_max_30m",

    "temperature_mean_15m",
    "temperature_mean_30m",

    "humidity_mean_15m",
    "humidity_mean_30m",
]


# ======================================================================
# PRINT HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def status(label: str, value: str) -> None:
    print(f"{label:<28}: {value}")


# ======================================================================
# LOAD DATASET
# ======================================================================

def load_dataset() -> pd.DataFrame:
    section("LOADING RAW DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(DATASET)

    print(f"Dataset : {DATASET}")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    missing = [
        col
        for col in RAW_REQUIRED
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required raw columns are missing:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    print()
    print("Raw source columns:")
    for col in df.columns:
        print(f" - {col}")

    return df


# ======================================================================
# VALIDATE RAW DATA
# ======================================================================

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    section("VALIDATING RAW DATA")

    df = df.copy()

    # Timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(
            f"Invalid timestamp values: {bad}"
        )

    # Numeric source columns
    numeric_cols = [
        "temperature",
        "humidity",
        "smoke",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print("Missing raw values:")
    print(
        df[RAW_REQUIRED]
        .isna()
        .sum()
        .to_string()
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Duplicate timestamps
    duplicate_count = int(
        df["timestamp"].duplicated().sum()
    )

    print()
    status(
        "Duplicate timestamps",
        str(duplicate_count)
    )

    if duplicate_count > 0:
        warnings.warn(
            "Duplicate timestamps detected. "
            "They are preserved; no synthetic values are created."
        )

    # Time differences
    deltas = (
        df["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    if len(deltas) > 0:
        print()
        print("Timestamp interval statistics:")
        print(
            deltas.describe()
            .to_string()
        )

        one_minute_ratio = float(
            (deltas == 1.0).mean()
        )

        status(
            "1-minute interval ratio",
            f"{one_minute_ratio:.4f}"
        )

    return df


# ======================================================================
# TIME-BASED FEATURE HELPERS
# ======================================================================

def calculate_change(
    series: pd.Series,
    minutes: int,
) -> pd.Series:
    """
    Change relative to the observation approximately N minutes ago.

    Uses timestamp-aware shift rather than assuming that every row
    represents exactly one minute.

    IMPORTANT:
        If the exact timestamp does not exist, result is NaN.
    """

    indexed = pd.Series(
        series.to_numpy(),
        index=series.index
    )

    return indexed - indexed.shift(
        periods=minutes
    )


def calculate_change_timestamp_aware(
    df: pd.DataFrame,
    column: str,
    minutes: int,
) -> pd.Series:
    """
    Compute value(t) - value(t-N minutes).

    Because this dataset is expected to be 1-minute data, a positional
    shift is valid only when timestamp spacing is continuous.

    If timestamp continuity is broken, the corresponding value is
    invalidated rather than fabricated.
    """

    result = (
        df[column]
        - df[column].shift(minutes)
    )

    timestamps = df["timestamp"]

    expected_previous = (
        timestamps
        - pd.to_timedelta(minutes, unit="m")
    )

    actual_previous = (
        timestamps.shift(minutes)
    )

    valid = (
        actual_previous == expected_previous
    )

    result = result.where(valid)

    return result


def rolling_time_feature(
    df: pd.DataFrame,
    column: str,
    minutes: int,
    function: str,
) -> pd.Series:
    """
    Timestamp-aware rolling feature.

    Uses a time-based window and therefore does not invent observations
    when timestamps are irregular.

    closed='left' means the current observation itself is not used,
    which avoids target/current-value contamination for historical
    change statistics.
    """

    temp = df[
        ["timestamp", column]
    ].copy()

    temp = temp.set_index("timestamp")

    window = f"{minutes}min"

    rolling = temp[column].rolling(
        window=window,
        closed="left",
        min_periods=minutes,
    )

    if function == "mean":
        result = rolling.mean()

    elif function == "std":
        result = rolling.std()

    elif function == "max":
        result = rolling.max()

    else:
        raise ValueError(
            f"Unsupported rolling function: {function}"
        )

    result.index = df.index

    return result


# ======================================================================
# RECONSTRUCT V4 FEATURES
# ======================================================================

def reconstruct_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("RECONSTRUCTING V4 FEATURES")

    out = df.copy()

    # --------------------------------------------------------------
    # SMOKE CHANGES
    # --------------------------------------------------------------

    smoke_change_windows = [
        1,
        5,
        15,
        30,
        60,
    ]

    for minutes in smoke_change_windows:

        name = f"smoke_change_{minutes}m"

        out[name] = calculate_change_timestamp_aware(
            out,
            "smoke",
            minutes,
        )

        print(f"Created: {name}")

    # --------------------------------------------------------------
    # TEMPERATURE CHANGES
    # --------------------------------------------------------------

    temperature_change_windows = [
        5,
        15,
        30,
        60,
    ]

    for minutes in temperature_change_windows:

        name = (
            f"temperature_change_{minutes}m"
        )

        out[name] = calculate_change_timestamp_aware(
            out,
            "temperature",
            minutes,
        )

        print(f"Created: {name}")

    # --------------------------------------------------------------
    # HUMIDITY CHANGES
    # --------------------------------------------------------------

    humidity_change_windows = [
        5,
        15,
        30,
    ]

    for minutes in humidity_change_windows:

        name = (
            f"humidity_change_{minutes}m"
        )

        out[name] = calculate_change_timestamp_aware(
            out,
            "humidity",
            minutes,
        )

        print(f"Created: {name}")

    # --------------------------------------------------------------
    # SMOKE ROLLING MEAN
    # --------------------------------------------------------------

    for minutes in [
        5,
        15,
        30,
        60,
    ]:

        name = f"smoke_mean_{minutes}m"

        out[name] = rolling_time_feature(
            out,
            "smoke",
            minutes,
            "mean",
        )

        print(f"Created: {name}")

    # --------------------------------------------------------------
    # SMOKE STD
    # --------------------------------------------------------------

    for minutes in [
        15,
        30,
    ]:

        name = f"smoke_std_{minutes}m"

        out[name] = rolling_time_feature(
            out,
            "smoke",
            minutes,
            "std",
        )

        print(f"Created: {name}")

    # --------------------------------------------------------------
    # SMOKE MAX
    # --------------------------------------------------------------

    out["smoke_max_30m"] = rolling_time_feature(
        out,
        "smoke",
        30,
        "max",
    )

    print("Created: smoke_max_30m")

    # --------------------------------------------------------------
    # TEMPERATURE MEAN
    # --------------------------------------------------------------

    for minutes in [
        15,
        30,
    ]:

        name = (
            f"temperature_mean_{minutes}m"
        )

        out[name] = rolling_time_feature(
            out,
            "temperature",
            minutes,
            "mean",
        )

        print(f"Created: {name}")

    # --------------------------------------------------------------
    # HUMIDITY MEAN
    # --------------------------------------------------------------

    for minutes in [
        15,
        30,
    ]:

        name = (
            f"humidity_mean_{minutes}m"
        )

        out[name] = rolling_time_feature(
            out,
            "humidity",
            minutes,
            "mean",
        )

        print(f"Created: {name}")

    return out


# ======================================================================
# CHECK FEATURE QUALITY
# ======================================================================

def audit_features(
    df: pd.DataFrame,
) -> None:

    section("FEATURE QUALITY AUDIT")

    print(
        f"{'Feature':<32}"
        f"{'Missing':>12}"
        f"{'Valid':>12}"
    )

    print("-" * 56)

    all_present = True

    for feature in V4_EXPECTED_ENGINEERED:

        if feature not in df.columns:
            print(
                f"{feature:<32}"
                f"{'MISSING':>12}"
                f"{'0':>12}"
            )

            all_present = False
            continue

        missing = int(
            df[feature].isna().sum()
        )

        valid = int(
            df[feature].notna().sum()
        )

        print(
            f"{feature:<32}"
            f"{missing:>12,}"
            f"{valid:>12,}"
        )

    print()

    if all_present:
        status(
            "Engineered feature schema",
            "PASS"
        )
    else:
        status(
            "Engineered feature schema",
            "FAIL"
        )


# ======================================================================
# LOAD MODEL SCHEMA
# ======================================================================

def get_model_features(
    artifact,
):
    """
    Extract feature names from V4 artifact.

    Supported artifact layouts:
        artifact['feature_names']
        artifact['features']
        model.feature_names_in_
    """

    if isinstance(artifact, dict):

        feature_names = artifact.get(
            "feature_names"
        )

        if feature_names is not None:
            return list(feature_names)

        features = artifact.get(
            "features"
        )

        if features is not None:
            return list(features)

        model = artifact.get("model")

        if model is not None and hasattr(
            model,
            "feature_names_in_",
        ):
            return list(
                model.feature_names_in_
            )

    elif hasattr(
        artifact,
        "feature_names_in_",
    ):
        return list(
            artifact.feature_names_in_
        )

    return None


# ======================================================================
# MODEL SCHEMA AUDIT
# ======================================================================

def audit_model_schemas(
    df: pd.DataFrame,
) -> dict:

    section("V4 MODEL SCHEMA AUDIT")

    results = {}

    for (
        experiment,
        horizon,
    ), model_path in MODEL_SPECS.items():

        label = (
            f"{experiment.upper()} / {horizon}h"
        )

        print()
        print("-" * 70)
        print(label)
        print("-" * 70)

        if not model_path.exists():

            status(
                "Model",
                "NOT FOUND"
            )

            results[
                (experiment, horizon)
            ] = {
                "status": "MODEL_NOT_FOUND"
            }

            continue

        try:
            artifact = joblib.load(
                model_path
            )
        except Exception as exc:

            status(
                "Model",
                "LOAD ERROR"
            )

            print(
                f"Error: {exc}"
            )

            results[
                (experiment, horizon)
            ] = {
                "status": "MODEL_LOAD_ERROR"
            }

            continue

        feature_names = get_model_features(
            artifact
        )

        if feature_names is None:

            status(
                "Feature schema",
                "NOT FOUND"
            )

            results[
                (experiment, horizon)
            ] = {
                "status": "FEATURE_SCHEMA_NOT_FOUND"
            }

            continue

        available = set(df.columns)
        required = set(feature_names)

        missing = sorted(
            required - available
        )

        extra = sorted(
            available - required
        )

        print(
            f"Model features : {len(feature_names)}"
        )

        print(
            f"Available cols : {len(available)}"
        )

        if missing:

            status(
                "Required features",
                "MISSING"
            )

            print()
            print("Missing features:")

            for feature in missing:
                print(
                    f" - {feature}"
                )

            schema_status = "FAIL"

        else:

            status(
                "Required features",
                "PASS"
            )

            schema_status = "PASS"

        # Exact order check
        ordered_present = [
            feature
            for feature in feature_names
            if feature in df.columns
        ]

        order_available = (
            len(ordered_present)
            == len(feature_names)
        )

        if order_available:

            status(
                "Feature order",
                "PASS"
            )

        else:

            status(
                "Feature order",
                "BLOCKED"
            )

        results[
            (experiment, horizon)
        ] = {
            "status": schema_status,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "missing": missing,
            "extra": extra,
        }

    return results


# ======================================================================
# V4 FEATURE COMPLETENESS
# ======================================================================

def print_completeness_summary(
    df: pd.DataFrame,
) -> None:

    section("V4 FEATURE COMPLETENESS SUMMARY")

    engineered_present = [
        feature
        for feature in V4_EXPECTED_ENGINEERED
        if feature in df.columns
    ]

    engineered_missing = [
        feature
        for feature in V4_EXPECTED_ENGINEERED
        if feature not in df.columns
    ]

    print(
        f"Expected engineered features : "
        f"{len(V4_EXPECTED_ENGINEERED)}"
    )

    print(
        f"Reconstructed                  : "
        f"{len(engineered_present)}"
    )

    print(
        f"Missing                         : "
        f"{len(engineered_missing)}"
    )

    if engineered_missing:

        print()
        print("Missing engineered features:")

        for feature in engineered_missing:
            print(
                f" - {feature}"
            )

    else:

        print()
        print(
            "All previously identified V4 "
            "engineered features exist."
        )


# ======================================================================
# SAVE AUDIT DATASET
# ======================================================================

def save_output(
    df: pd.DataFrame,
) -> None:

    section("SAVING AUDIT DATASET")

    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Never overwrite the original dataset.
    if OUTPUT_DATASET.resolve() == DATASET.resolve():
        raise RuntimeError(
            "Safety check failed: output equals input."
        )

    df.to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    print(
        f"Output: {OUTPUT_DATASET}"
    )

    print(
        f"Rows  : {len(df):,}"
    )

    print(
        f"Cols  : {len(df.columns)}"
    )


# ======================================================================
# FINAL CHECKPOINT
# ======================================================================

def final_checkpoint(
    df: pd.DataFrame,
    schema_results: dict,
) -> None:

    section("FINAL CHECKPOINT")

    model_count = len(MODEL_SPECS)

    found_count = sum(
        1
        for result in schema_results.values()
        if result.get("status")
        not in {
            "MODEL_NOT_FOUND",
            "MODEL_LOAD_ERROR",
            "FEATURE_SCHEMA_NOT_FOUND",
        }
    )

    schema_pass_count = sum(
        1
        for result in schema_results.values()
        if result.get("status") == "PASS"
    )

    engineered_missing = [
        feature
        for feature in V4_EXPECTED_ENGINEERED
        if feature not in df.columns
    ]

    print(
        f"Models discovered : "
        f"{found_count}/{model_count}"
    )

    print(
        f"Schema PASS       : "
        f"{schema_pass_count}/{model_count}"
    )

    print(
        f"Engineered V4 features : "
        f"{len(V4_EXPECTED_ENGINEERED) - len(engineered_missing)}"
        f"/{len(V4_EXPECTED_ENGINEERED)}"
    )

    print()
    print("Models modified : NO")
    print("Original dataset modified : NO")
    print("Fabricated values : NO")
    print("Weather values added : NO")
    print("Inference performed : NO")
    print("Confidence generated : NO")
    print("Uncertainty generated : NO")

    print()

    if not engineered_missing:
        print(
            "STATUS: 🟢 V4 FEATURE RECONSTRUCTION READY"
        )
    else:
        print(
            "STATUS: 🟡 PARTIAL — "
            "SOME V4 FEATURES ARE NOT RECONSTRUCTED"
        )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    section(
        "🔥 FireGuard — V4 Feature Reconstruction Audit"
    )

    print(
        "READ-ONLY RECONSTRUCTION"
    )

    print(
        "Models modified : NO"
    )

    print(
        "Dataset modified: NO"
    )

    print(
        "Fabricated data  : NO"
    )

    print(
        "Inference        : NO"
    )

    # --------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------------

    df = validate_raw_data(
        df
    )

    # --------------------------------------------------------------
    # RECONSTRUCT
    # --------------------------------------------------------------

    df = reconstruct_features(
        df
    )

    # --------------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------------

    audit_features(
        df
    )

    print_completeness_summary(
        df
    )

    # --------------------------------------------------------------
    # MODEL SCHEMAS
    # --------------------------------------------------------------

    schema_results = audit_model_schemas(
        df
    )

    # --------------------------------------------------------------
    # SAVE SEPARATE AUDIT DATASET
    # --------------------------------------------------------------

    save_output(
        df
    )

    # --------------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------------

    final_checkpoint(
        df,
        schema_results,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 70)
        print("❌ FATAL ERROR")
        print("=" * 70)

        print(
            str(exc)
        )

        sys.exit(1)