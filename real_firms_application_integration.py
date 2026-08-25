"""
FIREGUARD — REAL FIRMS APPLICATION INTEGRATION V1

Local application integration.
NO API
NO HTTP
NO RETRAINING
NO MODEL MODIFICATION
NO DATASET MODIFICATION
NO SYNTHETIC DATA

This module uses the existing production service directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from real_firms_service import RealFirmsProductionService


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "retraining"
    / "real_firms_application_results.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "retraining"
    / "real_firms_application_integration_report.json"
)


# ---------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------

def banner() -> None:
    print("=" * 72)
    print("🔥 FIREGUARD — REAL FIRMS APPLICATION INTEGRATION V1")
    print("=" * 72)
    print("Local application integration.")
    print("API                  : NO")
    print("HTTP                 : NO")
    print("Retraining           : NO")
    print("Model modification   : NO")
    print("Dataset modification : NO")
    print("Synthetic data       : NO")
    print("Fabricated labels    : NO")
    print("=" * 72)


# ---------------------------------------------------------------------
# SERVICE
# ---------------------------------------------------------------------

def load_service() -> RealFirmsProductionService:
    print()
    print("Loading production service...")
    print()

    service = RealFirmsProductionService()

    print("Production service loaded: PASS")

    return service


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

def load_real_firms_data() -> pd.DataFrame:
    print()
    print("=" * 72)
    print("LOADING REAL FIRMS DATA")
    print("=" * 72)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset : {DATASET_PATH}")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


# ---------------------------------------------------------------------
# SINGLE RECORD
# ---------------------------------------------------------------------

def predict_record(
    service: RealFirmsProductionService,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict one real FIRMS record directly through the production service.

    The service is responsible for:
      - feature preparation
      - categorical encoding
      - model inference
      - production thresholds
    """

    result = service.predict(record)

    if not isinstance(result, dict):
        raise TypeError(
            "Production service returned an unexpected result type."
        )

    return result


# ---------------------------------------------------------------------
# BATCH
# ---------------------------------------------------------------------

def predict_dataframe(
    service: RealFirmsProductionService,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the existing production service on real FIRMS records.

    No modification is made to the original dataframe.
    """

    records = df.to_dict(orient="records")

    results = []

    total = len(records)

    print()
    print("=" * 72)
    print("RUNNING LOCAL APPLICATION INFERENCE")
    print("=" * 72)
    print(f"Input rows : {total:,}")

    for index, record in enumerate(records, start=1):
        result = predict_record(service, record)

        output = dict(result)
        output["_source_row"] = index

        results.append(output)

        if index % 500 == 0 or index == total:
            print(
                f"Processed : {index:,}/{total:,}"
            )

    if not results:
        raise RuntimeError("No prediction results were generated.")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# OUTPUT VALIDATION
# ---------------------------------------------------------------------

def validate_output(
    result_df: pd.DataFrame,
) -> Dict[str, Any]:

    print()
    print("=" * 72)
    print("VALIDATING APPLICATION OUTPUT")
    print("=" * 72)

    if result_df.empty:
        raise ValueError("Application output is empty.")

    required_probability_columns = [
        "prob_24h",
        "prob_48h",
        "prob_72h",
    ]

    required_prediction_columns = [
        "pred_24h",
        "pred_48h",
        "pred_72h",
    ]

    missing = [
        column
        for column in (
            required_probability_columns
            + required_prediction_columns
        )
        if column not in result_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing application output columns: "
            + ", ".join(missing)
        )

    print("Required output columns : PASS")

    for column in required_probability_columns:
        values = pd.to_numeric(
            result_df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Invalid probability values in {column}"
            )

        if ((values < 0) | (values > 1)).any():
            raise ValueError(
                f"Probability out of range in {column}"
            )

    print("Probability values      : PASS")

    for column in required_prediction_columns:
        values = pd.to_numeric(
            result_df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Invalid prediction values in {column}"
            )

        if not values.isin([0, 1]).all():
            raise ValueError(
                f"Prediction values must be 0/1 in {column}"
            )

    print("Prediction values       : PASS")

    summary = {}

    for horizon in ["24h", "48h", "72h"]:
        prob_col = f"prob_{horizon}"
        pred_col = f"pred_{horizon}"

        probabilities = result_df[prob_col].astype(float)
        predictions = result_df[pred_col].astype(int)

        summary[horizon] = {
            "rows": int(len(result_df)),
            "mean_probability": float(probabilities.mean()),
            "min_probability": float(probabilities.min()),
            "max_probability": float(probabilities.max()),
            "positive_predictions": int(predictions.sum()),
            "positive_rate": float(predictions.mean()),
        }

    return summary


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

def save_results(
    result_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:

    print()
    print("=" * 72)
    print("SAVING APPLICATION RESULTS")
    print("=" * 72)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    report = {
        "project": "FIREGUARD",
        "component": "REAL FIRMS APPLICATION INTEGRATION V1",
        "integration_mode": "LOCAL_DIRECT_SERVICE",
        "api": False,
        "http": False,
        "retraining": False,
        "model_modified": False,
        "dataset_modified": False,
        "synthetic_data": False,
        "fabricated_labels": False,
        "dataset": str(DATASET_PATH),
        "results": str(OUTPUT_PATH),
        "summary": summary,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Results : {OUTPUT_PATH}")
    print(f"Report  : {REPORT_PATH}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    banner()

    service = load_service()

    df = load_real_firms_data()

    print()
    print("=" * 72)
    print("SERVICE STATUS")
    print("=" * 72)

    if hasattr(service, "status"):
        status = service.status()

        if isinstance(status, dict):
            for key, value in status.items():
                print(f"{key}: {value}")
        else:
            print(status)

    result_df = predict_dataframe(
        service,
        df,
    )

    summary = validate_output(
        result_df,
    )

    save_results(
        result_df,
        summary,
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    for horizon in ["24h", "48h", "72h"]:
        item = summary[horizon]

        print(
            f"{horizon.upper()} | "
            f"Rows={item['rows']} | "
            f"MeanProb={item['mean_probability']:.4f} | "
            f"PositivePredictions={item['positive_predictions']} | "
            f"PositiveRate={item['positive_rate']:.4%}"
        )

    print()
    print("API integration        : NO")
    print("HTTP server             : NO")
    print("Production service      : USED")
    print("Models modified         : NO")
    print("Retraining performed    : NO")
    print("Dataset modified       : NO")
    print("Synthetic data          : NO")
    print("Fabricated labels       : NO")
    print()
    print("STATUS: 🟢 LOCAL APPLICATION INTEGRATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
