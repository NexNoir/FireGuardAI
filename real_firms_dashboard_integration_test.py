# -*- coding: utf-8 -*-

"""
========================================================================
🔥 FIREGUARD — REAL FIRMS DASHBOARD INTEGRATION TEST V1
========================================================================

Local dashboard integration test.

NO API
NO HTTP
NO FASTAPI
NO UVICORN
NO RETRAINING
NO MODEL MODIFICATION
NO DATASET MODIFICATION
NO SYNTHETIC DATA
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import pandas as pd


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\vista\Desktop\fireguard_v2.0"
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------
# WARNINGS
# ---------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
)


# ---------------------------------------------------------------------
# IMPORT
# ---------------------------------------------------------------------

from dashboard.real_firms_dashboard_service import (
    FireGuardDashboardService,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

DATASET = (
    PROJECT_DIR
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)


# ---------------------------------------------------------------------
# FEATURE SOURCE COLUMNS
# ---------------------------------------------------------------------

BASE_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "confidence",
    "bright_t31",
    "frp",
    "hour",
    "minute",
]


# ---------------------------------------------------------------------
# RECORD PREPARATION
# ---------------------------------------------------------------------

def prepare_record(row: pd.Series) -> dict:

    record = {}

    for column in BASE_COLUMNS:

        value = row.get(column)

        if pd.isna(value):
            record[column] = None
        else:
            record[column] = value

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Pass the categorical fields through unchanged.
    # The production service is responsible for encoding them.
    # -------------------------------------------------------------

    categorical_columns = [
        "daynight",
        "satellite",
        "instrument",
        "type",
        "season",
    ]

    for column in categorical_columns:

        if column in row.index:

            value = row.get(column)

            if pd.isna(value):
                record[column] = None
            else:
                record[column] = value

    return record


# ---------------------------------------------------------------------
# RESULT VALIDATION
# ---------------------------------------------------------------------

def validate_prediction(result: dict) -> None:

    if not isinstance(result, dict):
        raise RuntimeError(
            "Prediction result is not a dictionary."
        )

    # Accept either naming convention used by the service.
    probability_keys = [
        "prob_24h",
        "prob_48h",
        "prob_72h",
    ]

    prediction_keys = [
        "pred_24h",
        "pred_48h",
        "pred_72h",
    ]

    found_probability = [
        key for key in probability_keys
        if key in result
    ]

    found_prediction = [
        key for key in prediction_keys
        if key in result
    ]

    if not found_probability:
        raise RuntimeError(
            "No probability fields found in prediction result."
        )

    if not found_prediction:
        raise RuntimeError(
            "No prediction fields found in prediction result."
        )

    for key in found_probability:

        value = result[key]

        if value is None:
            raise RuntimeError(
                f"{key} is None."
            )

        value = float(value)

        if not 0.0 <= value <= 1.0:

            raise RuntimeError(
                f"{key} outside [0,1]: {value}"
            )

    for key in found_prediction:

        value = int(result[key])

        if value not in (0, 1):

            raise RuntimeError(
                f"{key} must be 0 or 1: {value}"
            )


# ---------------------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print("🔥 FIREGUARD — REAL FIRMS DASHBOARD INTEGRATION TEST V1")
    print("=" * 72)

    print()
    print("Architecture")
    print("-" * 72)
    print("Dashboard adapter : LOCAL")
    print("API               : NO")
    print("HTTP              : NO")
    print("FastAPI           : NO")
    print("Uvicorn           : NO")
    print("Retraining        : NO")
    print("Model modification: NO")
    print("Dataset modification: NO")
    print()

    # -----------------------------------------------------------------
    # FILE CHECK
    # -----------------------------------------------------------------

    print("=" * 72)
    print("1. CHECKING DATASET")
    print("=" * 72)

    if not DATASET.exists():

        print("dataset: FAIL")
        print(DATASET)
        sys.exit(1)

    print("dataset: PASS")
    print(DATASET)

    # -----------------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("2. LOADING REAL FIRMS DATA")
    print("=" * 72)

    df = pd.read_csv(DATASET)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    if len(df) == 0:

        print("dataset rows: FAIL")
        sys.exit(1)

    print("dataset loading: PASS")

    # -----------------------------------------------------------------
    # SERVICE
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("3. LOADING DASHBOARD SERVICE")
    print("=" * 72)

    service = FireGuardDashboardService()

    print("dashboard service: PASS")

    # -----------------------------------------------------------------
    # STATUS
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("4. SERVICE STATUS")
    print("=" * 72)

    status = service.status()

    for key, value in status.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------------------
    # SELECT REAL RECORD
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("5. SELECTING REAL RECORD")
    print("=" * 72)

    row = df.iloc[0]

    record = prepare_record(row)

    print("Source row : 0")
    print("Synthetic  : NO")
    print("Record     : PASS")

    # -----------------------------------------------------------------
    # SINGLE PREDICTION
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("6. TESTING SINGLE DASHBOARD PREDICTION")
    print("=" * 72)

    try:

        result = service.predict(record)

    except Exception as exc:

        print()
        print("single prediction: FAIL")
        print()
        print(f"Error: {exc}")
        sys.exit(1)

    validate_prediction(result)

    print("single prediction: PASS")

    print()
    print("Prediction result:")

    for key, value in result.items():

        if key.startswith("prob_") or key.startswith("pred_"):
            print(f"  {key}: {value}")

    # -----------------------------------------------------------------
    # BATCH TEST
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("7. TESTING DASHBOARD BATCH PREDICTION")
    print("=" * 72)

    sample_size = min(10, len(df))

    records = [
        prepare_record(df.iloc[index])
        for index in range(sample_size)
    ]

    try:

        batch_results = service.predict_batch(records)

    except Exception as exc:

        print()
        print("batch prediction: FAIL")
        print()
        print(f"Error: {exc}")
        sys.exit(1)

    if not isinstance(batch_results, list):

        print("batch result type: FAIL")
        sys.exit(1)

    if len(batch_results) != sample_size:

        print("batch row count: FAIL")
        print(
            f"Expected: {sample_size}"
        )
        print(
            f"Received: {len(batch_results)}"
        )
        sys.exit(1)

    for result in batch_results:
        validate_prediction(result)

    print(
        f"batch prediction: PASS ({sample_size} rows)"
    )

    # -----------------------------------------------------------------
    # FINAL
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print("Dataset loading             : PASS")
    print("Dashboard service           : PASS")
    print("Service status              : PASS")
    print("Single prediction           : PASS")
    print("Batch prediction            : PASS")
    print("Probability validation      : PASS")
    print("Prediction validation       : PASS")
    print()

    print("API                         : NO")
    print("HTTP                        : NO")
    print("Retraining                  : NO")
    print("Model modification         : NO")
    print("Dataset modification      : NO")
    print("Synthetic data              : NO")
    print()

    print(
        "STATUS: 🟢 DASHBOARD INTEGRATION TEST PASS"
    )
    print()
    print(
        "READY FOR: DASHBOARD UI CONNECTION"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()