import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ================================================================
# FIREGUARD — REAL FIRMS PRODUCTION INFERENCE AUDIT V2
# ================================================================

BASE_DIR = Path(r"C:\Users\vista\Desktop\fireguard_v2.0")

DATASET = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)

MODEL_DIR = BASE_DIR / "saved_models" / "real_firms_v1"

THRESHOLD_CONFIG = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_threshold_config_v1.json"
)

OUTPUT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_production_inference_results.csv"
)

REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_production_inference_audit_report.json"
)


FEATURES = [
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
    "daynight_encoded",
    "satellite_encoded",
    "instrument_encoded",
    "type_encoded",
    "season_encoded",
]


MODEL_FILES = {
    "24h": "fireguard_real_firms_sensor_only_24h_v1.joblib",
    "48h": "fireguard_real_firms_sensor_only_48h_v1.joblib",
    "72h": "fireguard_real_firms_sensor_only_72h_v1.joblib",
}


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def encode_features(df):
    df = df.copy()

    # ------------------------------------------------------------
    # Temporal encodings
    # ------------------------------------------------------------

    if "daynight_encoded" not in df.columns:
        if "daynight" not in df.columns:
            raise ValueError("Missing source column: daynight")

        mapping = {
            "D": 1,
            "N": 0,
            "day": 1,
            "night": 0,
            "DAY": 1,
            "NIGHT": 0,
        }

        df["daynight_encoded"] = (
            df["daynight"]
            .astype(str)
            .str.strip()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

        print("Created: daynight_encoded")

    if "satellite_encoded" not in df.columns:
        if "satellite" not in df.columns:
            raise ValueError("Missing source column: satellite")

        values = sorted(df["satellite"].astype(str).unique())
        mapping = {v: i for i, v in enumerate(values)}

        df["satellite_encoded"] = (
            df["satellite"].astype(str).map(mapping).fillna(-1).astype(int)
        )

        print("Created: satellite_encoded")

    if "instrument_encoded" not in df.columns:
        if "instrument" not in df.columns:
            raise ValueError("Missing source column: instrument")

        values = sorted(df["instrument"].astype(str).unique())
        mapping = {v: i for i, v in enumerate(values)}

        df["instrument_encoded"] = (
            df["instrument"].astype(str).map(mapping).fillna(-1).astype(int)
        )

        print("Created: instrument_encoded")

    if "type_encoded" not in df.columns:
        if "type" not in df.columns:
            raise ValueError("Missing source column: type")

        values = sorted(df["type"].astype(str).unique())
        mapping = {v: i for i, v in enumerate(values)}

        df["type_encoded"] = (
            df["type"].astype(str).map(mapping).fillna(-1).astype(int)
        )

        print("Created: type_encoded")

    if "season_encoded" not in df.columns:
        if "season" not in df.columns:
            raise ValueError("Missing source column: season")

        values = ["winter", "spring", "summer", "autumn"]
        mapping = {v: i for i, v in enumerate(values)}

        df["season_encoded"] = (
            df["season"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

        print("Created: season_encoded")

    return df


def extract_model(obj):
    """
    Models were saved as dictionaries containing the actual Pipeline.
    Extract the estimator without modifying the saved file.
    """

    if hasattr(obj, "predict_proba"):
        return obj

    if isinstance(obj, dict):
        preferred_keys = [
            "model",
            "pipeline",
            "estimator",
            "classifier",
            "clf",
        ]

        for key in preferred_keys:
            candidate = obj.get(key)

            if hasattr(candidate, "predict_proba"):
                return candidate

        for value in obj.values():
            if hasattr(value, "predict_proba"):
                return value

    raise TypeError(
        f"Could not extract predict_proba model from object type: "
        f"{type(obj).__name__}"
    )


def load_thresholds():
    with open(THRESHOLD_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    thresholds = cfg.get("thresholds")

    if not isinstance(thresholds, dict):
        raise ValueError("Threshold configuration does not contain 'thresholds'.")

    result = {}

    for horizon in ["24h", "48h", "72h"]:
        value = thresholds.get(horizon)

        if value is None:
            raise ValueError(
                f"Threshold for {horizon} not found in configuration."
            )

        result[horizon] = float(value)

    return result


def main():

    banner("🔥 FIREGUARD — REAL FIRMS PRODUCTION INFERENCE AUDIT V2")

    print("Production inference only.")
    print("Retraining          : NO")
    print("Model modification  : NO")
    print("Dataset modification: NO")
    print("Synthetic data      : NO")
    print("Fabricated labels    : NO")

    print(f"\nDataset       : {DATASET}")
    print(f"Model dir     : {MODEL_DIR}")
    print(f"Threshold cfg : {THRESHOLD_CONFIG}")

    # ============================================================
    # LOAD DATASET
    # ============================================================

    banner("LOADING PRODUCTION DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    df = pd.read_csv(DATASET)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # ============================================================
    # PREPARE FEATURES
    # ============================================================

    banner("PREPARING PRODUCTION FEATURES")

    df = encode_features(df)

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            "Required production features missing:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    print("\nFINAL FEATURE LIST:")

    for i, feature in enumerate(FEATURES, start=1):
        print(f"{i:02d}. {feature}")

    print(f"\nFeature count : {len(FEATURES)}")

    X = df[FEATURES].copy()

    # Force numeric conversion without altering original dataset.
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    valid_mask = X.notna().all(axis=1)

    X_valid = X.loc[valid_mask].copy()

    print(f"Valid rows    : {len(X_valid):,}")

    if len(X_valid) == 0:
        raise ValueError("No valid production rows available.")

    # ============================================================
    # LOAD THRESHOLDS
    # ============================================================

    banner("LOADING FINAL THRESHOLD CONFIGURATION")

    thresholds = load_thresholds()

    print(f"24H threshold : {thresholds['24h']}")
    print(f"48H threshold : {thresholds['48h']}")
    print(f"72H threshold : {thresholds['72h']}")

    # ============================================================
    # LOAD MODELS
    # ============================================================

    banner("LOADING PRODUCTION MODELS")

    models = {}

    for horizon, filename in MODEL_FILES.items():

        path = MODEL_DIR / filename

        print(f"\nLoading {horizon} model:")
        print(f"  {filename}")

        if not path.exists():
            raise FileNotFoundError(path)

        saved_object = joblib.load(path)

        print(
            "  Saved object type: "
            f"{type(saved_object).__name__}"
        )

        model = extract_model(saved_object)

        print(
            "  Extracted model type: "
            f"{type(model).__name__}"
        )

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"{horizon} model does not support predict_proba."
            )

        models[horizon] = model

        print("  Model loaded: PASS")

    # ============================================================
    # RUN INFERENCE
    # ============================================================

    banner("RUNNING PRODUCTION INFERENCE")

    result = df.copy()

    # Initialize required production columns.
    result["prob_24h"] = np.nan
    result["prob_48h"] = np.nan
    result["prob_72h"] = np.nan

    result["pred_24h"] = np.nan
    result["pred_48h"] = np.nan
    result["pred_72h"] = np.nan

    summary = {}

    for horizon in ["24h", "48h", "72h"]:

        print("\n" + "-" * 72)
        print(f"PRODUCTION INFERENCE / {horizon.upper()}")
        print("-" * 72)

        model = models[horizon]
        threshold = thresholds[horizon]

        probabilities = model.predict_proba(X_valid)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        prob_col = f"prob_{horizon}"
        pred_col = f"pred_{horizon}"

        result.loc[valid_mask, prob_col] = probabilities
        result.loc[valid_mask, pred_col] = predictions

        mean_probability = float(np.mean(probabilities))
        min_probability = float(np.min(probabilities))
        max_probability = float(np.max(probabilities))

        positive_predictions = int(predictions.sum())
        positive_rate = float(
            positive_predictions / len(predictions)
        )

        print(f"Valid rows           : {len(probabilities):,}")
        print(f"Threshold            : {threshold:.2f}")
        print(f"Mean probability     : {mean_probability:.4f}")
        print(f"Min probability      : {min_probability:.4f}")
        print(f"Max probability      : {max_probability:.4f}")
        print(f"Positive predictions : {positive_predictions:,}")
        print(f"Positive rate        : {positive_rate:.4%}")

        summary[horizon] = {
            "threshold": threshold,
            "valid_rows": int(len(probabilities)),
            "mean_probability": mean_probability,
            "min_probability": min_probability,
            "max_probability": max_probability,
            "positive_predictions": positive_predictions,
            "positive_rate": positive_rate,
        }

    # Convert predictions to nullable integer values.
    for col in ["pred_24h", "pred_48h", "pred_72h"]:
        result[col] = result[col].astype("Int64")

    # ============================================================
    # VALIDATE OUTPUT
    # ============================================================

    banner("VALIDATING PRODUCTION OUTPUT")

    required_output_columns = [
        "prob_24h",
        "prob_48h",
        "prob_72h",
        "pred_24h",
        "pred_48h",
        "pred_72h",
    ]

    missing_output = [
        c for c in required_output_columns
        if c not in result.columns
    ]

    if missing_output:
        raise RuntimeError(
            "Production output columns missing:\n"
            + "\n".join(f" - {x}" for x in missing_output)
        )

    for col in ["prob_24h", "prob_48h", "prob_72h"]:

        values = result.loc[valid_mask, col].astype(float)

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Invalid probability values detected in {col}."
            )

        if ((values < 0) | (values > 1)).any():
            raise RuntimeError(
                f"Probability outside [0,1] detected in {col}."
            )

    for col in ["pred_24h", "pred_48h", "pred_72h"]:

        values = result.loc[valid_mask, col].astype(int)

        if not set(values.unique()).issubset({0, 1}):
            raise RuntimeError(
                f"Invalid prediction values detected in {col}."
            )

    print("Required output columns: PASS")
    print("Probability values    : PASS")
    print("Prediction values      : PASS")

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    banner("SAVING PRODUCTION INFERENCE")

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Results : {OUTPUT}")

    # ============================================================
    # REPORT
    # ============================================================

    report = {
        "project": "FireGuard",
        "stage": "real_firms_production_inference_audit_v2",
        "period": "2001-2025",

        "dataset": str(DATASET),
        "model_directory": str(MODEL_DIR),
        "threshold_configuration": str(THRESHOLD_CONFIG),
        "output": str(OUTPUT),

        "retraining": False,
        "model_modification": False,
        "dataset_modification": False,
        "synthetic_data": False,
        "fabricated_labels": False,

        "feature_count": len(FEATURES),
        "features": FEATURES,

        "thresholds": thresholds,

        "models": MODEL_FILES,

        "summary": summary,

        "required_output_columns": required_output_columns,

        "status": "PASS",
    }

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ============================================================
    # FINAL
    # ============================================================

    banner("FINAL RESULT")

    for horizon in ["24h", "48h", "72h"]:

        s = summary[horizon]

        print(
            f"{horizon.upper()} | "
            f"Threshold={s['threshold']:.2f} | "
            f"MeanProb={s['mean_probability']:.4f} | "
            f"PositivePredictions={s['positive_predictions']:,} | "
            f"PositiveRate={s['positive_rate']:.4%}"
        )

    print()
    print("Models modified      : NO")
    print("Dataset modified     : NO")
    print("Retraining performed : NO")
    print("Synthetic data       : NO")
    print("Fabricated labels     : NO")

    print()
    print("STATUS: 🟢 PRODUCTION INFERENCE AUDIT COMPLETE")


if __name__ == "__main__":
    main()
