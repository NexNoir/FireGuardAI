
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ================================================================
# FIREGUARD — REAL FIRMS PRODUCTION INFERENCE AUDIT V1
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

OUTPUT_DIR = BASE_DIR / "data" / "retraining"

RESULTS_FILE = (
    OUTPUT_DIR
    / "real_firms_production_inference_results.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "real_firms_production_inference_audit_report.json"
)


MODEL_FILES = {
    "24h": "fireguard_real_firms_sensor_only_24h_v1.joblib",
    "48h": "fireguard_real_firms_sensor_only_48h_v1.joblib",
    "72h": "fireguard_real_firms_sensor_only_72h_v1.joblib",
}

TARGETS = {
    "24h": "fire_next_24h",
    "48h": "fire_next_48h",
    "72h": "fire_next_72h",
}

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


# Final thresholds selected from the completed threshold audit.
DEFAULT_THRESHOLDS = {
    "24h": 0.35,
    "48h": 0.35,
    "72h": 0.30,
}


def banner(text):
    print("=" * 72)
    print(text)
    print("=" * 72)


def load_thresholds():
    thresholds = DEFAULT_THRESHOLDS.copy()

    if THRESHOLD_CONFIG.exists():
        try:
            with open(THRESHOLD_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            # Accept either:
            # {"24h": 0.35, ...}
            # or:
            # {"thresholds": {"24h": 0.35, ...}}
            source = cfg.get("thresholds", cfg)

            for horizon in thresholds:
                value = source.get(horizon)

                if value is not None:
                    value = float(value)

                    if 0.0 < value < 1.0:
                        thresholds[horizon] = value

        except Exception as exc:
            print(f"WARNING: Could not read threshold config: {exc}")
            print("Using validated default thresholds.")

    return thresholds


def encode_column(df, column):
    """
    Deterministic integer encoding for inference.

    These columns contain categorical FIRMS values.
    Encoding is based on sorted unique values in the production dataset.
    """

    if column not in df.columns:
        raise ValueError(f"Missing source column: {column}")

    values = df[column].astype(str)

    categories = sorted(values.dropna().unique())

    mapping = {value: idx for idx, value in enumerate(categories)}

    return values.map(mapping).fillna(-1).astype(int)


def prepare_features(df):
    print()
    print("=" * 72)
    print("PREPARING PRODUCTION FEATURES")
    print("=" * 72)

    df = df.copy()

    # ------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------

    if "acq_date" not in df.columns:
        raise ValueError("Missing acq_date")

    if "acq_time" not in df.columns:
        raise ValueError("Missing acq_time")

    time_string = (
        df["acq_date"].astype(str).str.strip()
        + " "
        + df["acq_time"].astype(str).str.zfill(4)
    )

    timestamps = pd.to_datetime(
        time_string,
        format="%Y-%m-%d %H%M",
        errors="coerce",
    )

    if timestamps.isna().any():
        # Fallback for HH:MM or already formatted values.
        timestamps = pd.to_datetime(
            time_string,
            errors="coerce",
        )

    if timestamps.isna().any():
        bad = int(timestamps.isna().sum())
        raise ValueError(f"Invalid timestamps: {bad}")

    df["hour"] = timestamps.dt.hour.astype(int)
    df["minute"] = timestamps.dt.minute.astype(int)

    # ------------------------------------------------------------
    # Categorical features
    # ------------------------------------------------------------

    if "daynight_encoded" not in df.columns:
        df["daynight_encoded"] = encode_column(df, "daynight")
        print("Created: daynight_encoded")

    if "satellite_encoded" not in df.columns:
        df["satellite_encoded"] = encode_column(df, "satellite")
        print("Created: satellite_encoded")

    if "instrument_encoded" not in df.columns:
        df["instrument_encoded"] = encode_column(df, "instrument")
        print("Created: instrument_encoded")

    if "type_encoded" not in df.columns:
        df["type_encoded"] = encode_column(df, "type")
        print("Created: type_encoded")

    if "season_encoded" not in df.columns:
        df["season_encoded"] = encode_column(df, "season")
        print("Created: season_encoded")

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            "Required production features missing:\n"
            + "\n".join(f" - {c}" for c in missing)
        )

    X = df[FEATURES].copy()

    # ------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------

    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    valid_mask = X.notna().all(axis=1)

    invalid_rows = int((~valid_mask).sum())

    if invalid_rows:
        print(f"Rows with invalid features removed from inference: {invalid_rows}")

    X_valid = X.loc[valid_mask].copy()
    df_valid = df.loc[valid_mask].copy()

    print()
    print("FINAL FEATURE LIST:")

    for index, feature in enumerate(FEATURES, start=1):
        print(f"{index:02d}. {feature}")

    print()
    print(f"Feature count : {len(FEATURES)}")
    print(f"Valid rows    : {len(X_valid)}")

    return df_valid, X_valid, timestamps.loc[valid_mask]


def extract_model(obj):
    """
    Saved model files are dictionaries containing the actual estimator.
    Supports common keys without modifying the saved object.
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
        "Saved model does not contain an object with predict_proba()."
    )


def load_model(horizon):
    filename = MODEL_FILES[horizon]
    path = MODEL_DIR / filename

    print()
    print(f"Loading {horizon.upper()} model:")
    print(f"  {filename}")

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    saved = joblib.load(path)

    print(f"  Saved object type: {type(saved).__name__}")

    model = extract_model(saved)

    print(f"  Extracted model type: {type(model).__name__}")
    print("  Model loaded: PASS")

    return model


def validate_model_schema(model, horizon):
    expected = FEATURES

    model_features = None

    if hasattr(model, "feature_names_in_"):
        try:
            model_features = list(model.feature_names_in_)
        except Exception:
            model_features = None

    if model_features is not None:
        if model_features != expected:
            raise ValueError(
                f"{horizon.upper()} model feature schema mismatch.\n"
                f"Expected: {expected}\n"
                f"Model:    {model_features}"
            )

    print(f"{horizon.upper()} schema: PASS")


def safe_probability(model, X):
    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Model predict_proba output is not binary."
        )

    result = probabilities[:, 1]

    if not np.isfinite(result).all():
        raise ValueError(
            "Model produced NaN or infinite probabilities."
        )

    result = np.clip(result, 0.0, 1.0)

    return result


def main():

    banner("🔥 FIREGUARD — REAL FIRMS PRODUCTION INFERENCE AUDIT V1")

    print("Production inference only.")
    print("Retraining          : NO")
    print("Model modification  : NO")
    print("Dataset modification: NO")
    print("Synthetic data      : NO")
    print("Fabricated labels    : NO")
    print()

    print(f"Dataset       : {DATASET}")
    print(f"Model dir     : {MODEL_DIR}")
    print(f"Threshold cfg : {THRESHOLD_CONFIG}")

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    banner("LOADING PRODUCTION DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # ------------------------------------------------------------
    # Dataset integrity check
    # ------------------------------------------------------------

    if len(df) == 0:
        raise ValueError("Dataset is empty.")

    # ------------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------------

    df_valid, X_valid, timestamps = prepare_features(df)

    # ------------------------------------------------------------
    # Load thresholds
    # ------------------------------------------------------------

    banner("LOADING FINAL THRESHOLD CONFIGURATION")

    thresholds = load_thresholds()

    print(f"24H threshold : {thresholds['24h']:.2f}")
    print(f"48H threshold : {thresholds['48h']:.2f}")
    print(f"72H threshold : {thresholds['72h']:.2f}")

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    banner("LOADING PRODUCTION MODELS")

    models = {}

    for horizon in ["24h", "48h", "72h"]:

        model = load_model(horizon)

        validate_model_schema(model, horizon)

        models[horizon] = model

    # ------------------------------------------------------------
    # Production inference
    # ------------------------------------------------------------

    banner("RUNNING PRODUCTION INFERENCE")

    output = df_valid.copy()

    output["inference_timestamp"] = datetime.now().isoformat(
        timespec="seconds"
    )

    output["inference_source"] = "real_firms_2001_2025"

    summary = {}

    for horizon in ["24h", "48h", "72h"]:

        print()
        print("-" * 72)
        print(f"PRODUCTION INFERENCE / {horizon.upper()}")
        print("-" * 72)

        model = models[horizon]
        threshold = thresholds[horizon]

        probabilities = safe_probability(
            model,
            X_valid,
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        probability_column = (
            f"fire_probability_{horizon}"
        )

        prediction_column = (
            f"fire_prediction_{horizon}"
        )

        threshold_column = (
            f"fire_threshold_{horizon}"
        )

        output[probability_column] = probabilities
        output[prediction_column] = predictions
        output[threshold_column] = threshold

        positive_predictions = int(predictions.sum())

        print(f"Valid rows           : {len(X_valid):,}")
        print(f"Threshold            : {threshold:.2f}")
        print(
            f"Mean probability     : "
            f"{probabilities.mean():.4f}"
        )
        print(
            f"Max probability      : "
            f"{probabilities.max():.4f}"
        )
        print(
            f"Positive predictions  : "
            f"{positive_predictions:,}"
        )
        print(
            f"Positive rate         : "
            f"{positive_predictions / len(predictions):.4%}"
        )

        summary[horizon] = {
            "threshold": threshold,
            "valid_rows": int(len(X_valid)),
            "positive_predictions": positive_predictions,
            "positive_prediction_rate": float(
                positive_predictions / len(predictions)
            ),
            "mean_probability": float(
                probabilities.mean()
            ),
            "max_probability": float(
                probabilities.max()
            ),
        }

    # ------------------------------------------------------------
    # Add timestamp columns useful for production output
    # ------------------------------------------------------------

    output["event_timestamp"] = timestamps.astype(str).values

    # Keep original row order.
    output.insert(
        0,
        "production_inference_id",
        np.arange(1, len(output) + 1),
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    banner("SAVING PRODUCTION INFERENCE")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "project": "FireGuard",
        "stage": "real_firms_production_inference_audit_v1",
        "status": "PASS",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": str(DATASET),
        "model_directory": str(MODEL_DIR),
        "threshold_configuration": str(
            THRESHOLD_CONFIG
        ),
        "period": "2001-2025",
        "rows_loaded": int(len(df)),
        "rows_inferred": int(len(output)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "thresholds": thresholds,
        "models": MODEL_FILES,
        "summary": summary,
        "retraining": False,
        "models_modified": False,
        "dataset_modified": False,
        "synthetic_data": False,
        "fabricated_labels": False,
        "results_file": str(RESULTS_FILE),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Results : {RESULTS_FILE}")
    print(f"Report  : {REPORT_FILE}")

    # ------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------

    banner("FINAL RESULT")

    for horizon in ["24h", "48h", "72h"]:

        item = summary[horizon]

        print(
            f"{horizon.upper()} | "
            f"Threshold={item['threshold']:.2f} | "
            f"MeanProb={item['mean_probability']:.4f} | "
            f"PositivePredictions="
            f"{item['positive_predictions']:,} | "
            f"PositiveRate="
            f"{item['positive_prediction_rate']:.4%}"
        )

    print()
    print("Models modified      : NO")
    print("Dataset modified     : NO")
    print("Retraining performed : NO")
    print("Synthetic data       : NO")
    print("Fabricated labels    : NO")
    print()
    print("STATUS: 🟢 PRODUCTION INFERENCE AUDIT COMPLETE")


if __name__ == "__main__":
    main()
