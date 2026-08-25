# -*- coding: utf-8 -*-
"""
========================================================================
🔥 FIREGUARD — REAL FIRMS LOCAL PREDICT V1
========================================================================

Local production prediction runner.

NO API
NO HTTP
NO FastAPI
NO RETRAINING
NO MODEL MODIFICATION
NO DATASET MODIFICATION
NO SYNTHETIC DATA
NO FABRICATED LABELS

Pipeline:
    Real FIRMS CSV
        ↓
    RealFirmsProductionService
        ↓
    Batch Prediction
        ↓
    24H / 48H / 72H probabilities
        ↓
    Threshold predictions
        ↓
    CSV output
========================================================================
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Reduce unnecessary sklearn/joblib warnings.
# These warnings do NOT change model behavior.
# ---------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.utils.parallel",
)

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\vista\Desktop\fireguard_v2.0")

DATASET_PATH = (
    PROJECT_DIR
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "retraining"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "real_firms_local_prediction_results.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "real_firms_local_prediction_report.json"
)

BATCH_SIZE = 256


# ---------------------------------------------------------------------
# IMPORT PRODUCTION SERVICE
# ---------------------------------------------------------------------

try:
    from real_firms_service import RealFirmsProductionService
except Exception as exc:
    print()
    print("=" * 72)
    print("❌ FIREGUARD LOCAL PREDICT — IMPORT ERROR")
    print("=" * 72)
    print()
    print("Could not import RealFirmsProductionService.")
    print()
    print(f"Error: {exc}")
    print()
    print("Expected file:")
    print(PROJECT_DIR / "real_firms_service.py")
    print()
    sys.exit(1)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def print_header():
    print("=" * 72)
    print("🔥 FIREGUARD — REAL FIRMS LOCAL PREDICT V1")
    print("=" * 72)
    print()
    print("Local production inference.")
    print("API                  : NO")
    print("HTTP                 : NO")
    print("FastAPI              : NO")
    print("Retraining           : NO")
    print("Model modification   : NO")
    print("Dataset modification : NO")
    print("Synthetic data       : NO")
    print("Fabricated labels    : NO")
    print("=" * 72)
    print()


def load_dataset() -> pd.DataFrame:
    print("LOADING REAL FIRMS DATASET")
    print("-" * 72)
    print(f"Dataset : {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print()

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def normalize_prediction_result(result):
    """
    Normalize one service prediction result into a dictionary.

    Supports:
      - dict
      - pandas Series
      - objects exposing __dict__

    This keeps the runner compatible with the production service
    without modifying the service itself.
    """

    if isinstance(result, dict):
        return dict(result)

    if isinstance(result, pd.Series):
        return result.to_dict()

    if hasattr(result, "__dict__"):
        return dict(result.__dict__)

    raise TypeError(
        f"Unsupported prediction result type: {type(result)}"
    )


def extract_prediction_values(result_dict: dict):
    """
    Extract the canonical six production fields.

    Expected:
        prob_24h
        prob_48h
        prob_72h
        pred_24h
        pred_48h
        pred_72h
    """

    aliases = {
        "prob_24h": [
            "prob_24h",
            "probability_24h",
            "24h_probability",
            "prob24h",
        ],
        "prob_48h": [
            "prob_48h",
            "probability_48h",
            "48h_probability",
            "prob48h",
        ],
        "prob_72h": [
            "prob_72h",
            "probability_72h",
            "72h_probability",
            "prob72h",
        ],
        "pred_24h": [
            "pred_24h",
            "prediction_24h",
            "24h_prediction",
            "pred24h",
        ],
        "pred_48h": [
            "pred_48h",
            "prediction_48h",
            "48h_prediction",
            "pred48h",
        ],
        "pred_72h": [
            "pred_72h",
            "prediction_72h",
            "72h_prediction",
            "pred72h",
        ],
    }

    output = {}

    for canonical, possible_names in aliases.items():

        found = None

        for name in possible_names:
            if name in result_dict:
                found = result_dict[name]
                break

        if found is None:
            raise KeyError(
                f"Production service result missing field: {canonical}"
            )

        output[canonical] = found

    return output


def predict_batch(service, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run production inference in batches.

    The preferred production-service interface is:

        service.predict_batch(records)

    """

    all_results = []

    total = len(df)

    print("RUNNING LOCAL PRODUCTION INFERENCE")
    print("-" * 72)
    print(f"Input rows : {total:,}")
    print(f"Batch size : {BATCH_SIZE}")
    print()

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_df = df.iloc[start:end]

        records = batch_df.to_dict(
            orient="records"
        )

        try:
            results = service.predict_batch(records)
        except AttributeError as exc:
            raise RuntimeError(
                "RealFirmsProductionService must provide "
                "predict_batch(records)."
            ) from exc

        if results is None:
            raise RuntimeError(
                f"Service returned None for rows {start}:{end}."
            )

        if len(results) != len(records):
            raise RuntimeError(
                "Prediction count mismatch: "
                f"input={len(records)}, output={len(results)}"
            )

        for result in results:
            normalized = normalize_prediction_result(result)
            extracted = extract_prediction_values(normalized)
            all_results.append(extracted)

        processed = end

        print(
            f"Processed: {processed:,}/{total:,}"
            f" ({processed / total * 100:.1f}%)"
        )

    print()

    result_df = pd.DataFrame(all_results)

    if len(result_df) != len(df):
        raise RuntimeError(
            "Final output row count does not match input row count."
        )

    return result_df


def validate_output(result_df: pd.DataFrame):

    required = [
        "prob_24h",
        "prob_48h",
        "prob_72h",
        "pred_24h",
        "pred_48h",
        "pred_72h",
    ]

    print("VALIDATING PRODUCTION OUTPUT")
    print("-" * 72)

    missing = [
        column
        for column in required
        if column not in result_df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing output columns: {missing}"
        )

    print("Required columns : PASS")

    # -------------------------------------------------------------
    # Probability validation
    # -------------------------------------------------------------

    for column in [
        "prob_24h",
        "prob_48h",
        "prob_72h",
    ]:

        values = pd.to_numeric(
            result_df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise RuntimeError(
                f"{column} contains invalid probability values."
            )

        if ((values < 0) | (values > 1)).any():
            raise RuntimeError(
                f"{column} contains values outside [0, 1]."
            )

    print("Probability ranges: PASS")

    # -------------------------------------------------------------
    # Prediction validation
    # -------------------------------------------------------------

    for column in [
        "pred_24h",
        "pred_48h",
        "pred_72h",
    ]:

        values = pd.to_numeric(
            result_df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise RuntimeError(
                f"{column} contains invalid prediction values."
            )

        unique_values = set(
            values.astype(int).unique().tolist()
        )

        if not unique_values.issubset({0, 1}):
            raise RuntimeError(
                f"{column} contains values other than 0/1: "
                f"{unique_values}"
            )

    print("Prediction values : PASS")
    print()


def build_report(
    input_df: pd.DataFrame,
    result_df: pd.DataFrame,
):
    report = {
        "project": "FIREGUARD",
        "component": "real_firms_local_predict",
        "version": "V1",
        "timestamp": datetime.now().isoformat(),

        "mode": "local_production_inference",

        "api": False,
        "http": False,
        "fastapi": False,

        "retraining": False,
        "model_modified": False,
        "dataset_modified": False,
        "synthetic_data": False,
        "fabricated_labels": False,

        "dataset": str(DATASET_PATH),

        "input_rows": int(len(input_df)),
        "output_rows": int(len(result_df)),

        "batch_size": BATCH_SIZE,

        "prediction_summary": {
            "24h": {
                "mean_probability": float(
                    result_df["prob_24h"].mean()
                ),
                "positive_predictions": int(
                    result_df["pred_24h"].sum()
                ),
                "positive_rate": float(
                    result_df["pred_24h"].mean()
                ),
            },

            "48h": {
                "mean_probability": float(
                    result_df["prob_48h"].mean()
                ),
                "positive_predictions": int(
                    result_df["pred_48h"].sum()
                ),
                "positive_rate": float(
                    result_df["pred_48h"].mean()
                ),
            },

            "72h": {
                "mean_probability": float(
                    result_df["prob_72h"].mean()
                ),
                "positive_predictions": int(
                    result_df["pred_72h"].sum()
                ),
                "positive_rate": float(
                    result_df["pred_72h"].mean()
                ),
            },
        },

        "status": "PASS",
    }

    return report


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print_header()

    print("LOADING PRODUCTION SERVICE")
    print("-" * 72)

    try:
        service = RealFirmsProductionService()
    except Exception as exc:
        print()
        print("❌ Production service loading failed.")
        print(f"Error: {exc}")
        print()
        sys.exit(1)

    print("Production service loaded: PASS")
    print()

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    try:
        df = load_dataset()
    except Exception as exc:
        print()
        print("=" * 72)
        print("❌ LOCAL PREDICTION FAILED")
        print("=" * 72)
        print()
        print(exc)
        print()
        sys.exit(1)

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------

    try:
        predictions = predict_batch(
            service,
            df,
        )
    except Exception as exc:
        print()
        print("=" * 72)
        print("❌ LOCAL PRODUCTION INFERENCE FAILED")
        print("=" * 72)
        print()
        print(f"Error: {exc}")
        print()
        sys.exit(1)

    # -------------------------------------------------------------
    # Preserve original dataset
    # -------------------------------------------------------------

    output_df = df.copy()

    for column in predictions.columns:
        output_df[column] = predictions[column].values

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    try:
        validate_output(predictions)
    except Exception as exc:
        print()
        print("=" * 72)
        print("❌ OUTPUT VALIDATION FAILED")
        print("=" * 72)
        print()
        print(exc)
        print()
        sys.exit(1)

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    report = build_report(
        df,
        predictions,
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("=" * 72)
    print("FINAL LOCAL PRODUCTION RESULT")
    print("=" * 72)

    print(
        f"24H | MeanProb="
        f"{predictions['prob_24h'].mean():.4f}"
        f" | PositivePredictions="
        f"{int(predictions['pred_24h'].sum()):,}"
        f" | PositiveRate="
        f"{predictions['pred_24h'].mean() * 100:.4f}%"
    )

    print(
        f"48H | MeanProb="
        f"{predictions['prob_48h'].mean():.4f}"
        f" | PositivePredictions="
        f"{int(predictions['pred_48h'].sum()):,}"
        f" | PositiveRate="
        f"{predictions['pred_48h'].mean() * 100:.4f}%"
    )

    print(
        f"72H | MeanProb="
        f"{predictions['prob_72h'].mean():.4f}"
        f" | PositivePredictions="
        f"{int(predictions['pred_72h'].sum()):,}"
        f" | PositiveRate="
        f"{predictions['pred_72h'].mean() * 100:.4f}%"
    )

    print()
    print("=" * 72)
    print("SAVING RESULTS")
    print("=" * 72)

    print(f"Results : {OUTPUT_CSV}")
    print(f"Report  : {OUTPUT_REPORT}")

    print()
    print("API                  : NO")
    print("HTTP                 : NO")
    print("Retraining           : NO")
    print("Model modification   : NO")
    print("Dataset modification : NO")
    print("Synthetic data       : NO")
    print("Fabricated labels    : NO")

    print()
    print("STATUS: 🟢 LOCAL PRODUCTION PREDICTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()