import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ================================================================
# FIREGUARD — REAL FIRMS PRODUCTION READINESS AUDIT V3
# ================================================================
# READ-ONLY AUDIT
#
# Retraining          : NO
# Model modification  : NO
# Dataset modification: NO
# Synthetic data      : NO
# Fabricated labels   : NO
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

PRODUCTION_RESULTS = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_production_inference_results.csv"
)

REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_production_readiness_audit_v3_report.json"
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


EXPECTED_THRESHOLDS = {
    "24h": 0.35,
    "48h": 0.35,
    "72h": 0.30,
}


REQUIRED_OUTPUT_COLUMNS = [
    "prob_24h",
    "prob_48h",
    "prob_72h",
    "pred_24h",
    "pred_48h",
    "pred_72h",
]


def banner(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def fail(message):
    banner("❌ PRODUCTION READINESS BLOCKED")
    print(message)
    raise SystemExit(1)


def prepare_features(df):
    """
    Reconstruct inference features in memory only.
    The source dataset is NEVER saved or modified.
    """

    df = df.copy()

    if "daynight_encoded" not in df.columns:

        if "daynight" not in df.columns:
            raise ValueError("Missing source column: daynight")

        mapping = {
            "D": 1,
            "N": 0,
            "DAY": 1,
            "NIGHT": 0,
            "day": 1,
            "night": 0,
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
        mapping = {value: index for index, value in enumerate(values)}

        df["satellite_encoded"] = (
            df["satellite"]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

        print("Created: satellite_encoded")

    if "instrument_encoded" not in df.columns:

        if "instrument" not in df.columns:
            raise ValueError("Missing source column: instrument")

        values = sorted(df["instrument"].astype(str).unique())
        mapping = {value: index for index, value in enumerate(values)}

        df["instrument_encoded"] = (
            df["instrument"]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

        print("Created: instrument_encoded")

    if "type_encoded" not in df.columns:

        if "type" not in df.columns:
            raise ValueError("Missing source column: type")

        values = sorted(df["type"].astype(str).unique())
        mapping = {value: index for index, value in enumerate(values)}

        df["type_encoded"] = (
            df["type"]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

        print("Created: type_encoded")

    if "season_encoded" not in df.columns:

        if "season" not in df.columns:
            raise ValueError("Missing source column: season")

        mapping = {
            "winter": 0,
            "spring": 1,
            "summer": 2,
            "autumn": 3,
        }

        df["season_encoded"] = (
            df["season"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

        print("Created: season_encoded")

    return df


def extract_model(saved_object):
    """
    Existing FireGuard model files may contain either:
    - the model directly
    - a dictionary containing the actual Pipeline/model

    This function only reads/extracts.
    Nothing is modified or re-saved.
    """

    if hasattr(saved_object, "predict_proba"):
        return saved_object

    if isinstance(saved_object, dict):

        preferred_keys = [
            "model",
            "pipeline",
            "estimator",
            "classifier",
            "clf",
        ]

        for key in preferred_keys:

            candidate = saved_object.get(key)

            if hasattr(candidate, "predict_proba"):
                return candidate

        for value in saved_object.values():

            if hasattr(value, "predict_proba"):
                return value

    raise TypeError(
        "Could not extract a valid model with predict_proba."
    )


def get_thresholds(config):
    """
    Reads thresholds robustly from the existing configuration.
    Supports:
        thresholds: {"24h": 0.35, ...}
    """

    thresholds = config.get("thresholds")

    if not isinstance(thresholds, dict):
        raise ValueError(
            "Threshold configuration does not contain a valid "
            "'thresholds' dictionary."
        )

    result = {}

    for horizon in ["24h", "48h", "72h"]:

        if horizon not in thresholds:
            raise ValueError(
                f"Threshold for {horizon} not found in configuration."
            )

        value = float(thresholds[horizon])

        if not (0 < value < 1):
            raise ValueError(
                f"Invalid threshold for {horizon}: {value}"
            )

        result[horizon] = value

    return result


def validate_output(results, dataset_rows):
    """
    Validates the actual production inference CSV.
    """

    missing = [
        column
        for column in REQUIRED_OUTPUT_COLUMNS
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing production output columns: {missing}"
        )

    if len(results) != dataset_rows:
        raise ValueError(
            f"Production row count mismatch. "
            f"Dataset={dataset_rows:,}, "
            f"Results={len(results):,}"
        )

    for horizon in ["24h", "48h", "72h"]:

        prob_col = f"prob_{horizon}"
        pred_col = f"pred_{horizon}"

        probabilities = pd.to_numeric(
            results[prob_col],
            errors="coerce",
        )

        predictions = pd.to_numeric(
            results[pred_col],
            errors="coerce",
        )

        if probabilities.isna().any():
            raise ValueError(
                f"NaN or invalid values found in {prob_col}"
            )

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(
                f"Probability values outside [0,1] found in {prob_col}"
            )

        if predictions.isna().any():
            raise ValueError(
                f"NaN or invalid values found in {pred_col}"
            )

        prediction_values = set(
            predictions.astype(int).unique().tolist()
        )

        if not prediction_values.issubset({0, 1}):
            raise ValueError(
                f"Invalid prediction values in {pred_col}: "
                f"{prediction_values}"
            )

    return True


def main():

    banner("🔥 FIREGUARD — REAL FIRMS PRODUCTION READINESS AUDIT V3")

    print("Production readiness audit only.")
    print("Retraining          : NO")
    print("Model modification  : NO")
    print("Dataset modification: NO")
    print("Synthetic data      : NO")
    print("Fabricated labels   : NO")

    print()
    print(f"Dataset       : {DATASET}")
    print(f"Model dir     : {MODEL_DIR}")
    print(f"Threshold cfg : {THRESHOLD_CONFIG}")
    print(f"Results       : {PRODUCTION_RESULTS}")

    audit = {
        "project": "FireGuard",
        "stage": "real_firms_production_readiness_audit_v3",
        "retraining": False,
        "model_modification": False,
        "dataset_modification": False,
        "synthetic_data": False,
        "fabricated_labels": False,
        "checks": {},
        "status": "BLOCKED",
    }

    # ============================================================
    # 1. CHECK REQUIRED FILES
    # ============================================================

    banner("1. CHECKING REQUIRED FILES")

    required_files = {
        "dataset": DATASET,
        "model_directory": MODEL_DIR,
        "threshold_config": THRESHOLD_CONFIG,
        "production_results": PRODUCTION_RESULTS,
        "model_24h": MODEL_DIR / MODEL_FILES["24h"],
        "model_48h": MODEL_DIR / MODEL_FILES["48h"],
        "model_72h": MODEL_DIR / MODEL_FILES["72h"],
    }

    all_files_ok = True

    for name, path in required_files.items():

        exists = path.exists()

        print(
            f"{name:<20}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        audit["checks"][name] = bool(exists)

        if not exists:
            all_files_ok = False

    if not all_files_ok:
        fail("One or more required production files are missing.")

    # ============================================================
    # 2. LOAD DATASET
    # ============================================================

    banner("2. LOADING PRODUCTION DATASET")

    try:
        df = pd.read_csv(DATASET)
    except Exception as exc:
        fail(f"Could not load production dataset: {exc}")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    if len(df) == 0:
        fail("Production dataset contains zero rows.")

    audit["dataset_rows"] = int(len(df))
    audit["dataset_columns"] = int(len(df.columns))
    audit["checks"]["dataset_load"] = True

    # ============================================================
    # 3. VALIDATE DATASET BASE SCHEMA
    # ============================================================

    banner("3. VALIDATING DATASET SCHEMA")

    required_base_columns = [
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
        "daynight",
        "satellite",
        "instrument",
        "type",
        "season",
    ]

    missing_base = [
        column
        for column in required_base_columns
        if column not in df.columns
    ]

    if missing_base:
        fail(
            "Missing required base dataset columns: "
            f"{missing_base}"
        )

    print("Base schema: PASS")

    audit["checks"]["base_schema"] = True

    # ============================================================
    # 4. PREPARE FEATURES
    # ============================================================

    banner("4. PREPARING PRODUCTION FEATURES")

    try:
        prepared = prepare_features(df)
    except Exception as exc:
        fail(f"Feature preparation failed: {exc}")

    missing_features = [
        column
        for column in FEATURES
        if column not in prepared.columns
    ]

    if missing_features:
        fail(
            "Required production features missing: "
            f"{missing_features}"
        )

    print()
    print("FINAL FEATURE LIST:")

    for index, feature in enumerate(FEATURES, start=1):
        print(f"{index:02d}. {feature}")

    X = prepared[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    valid_mask = X.notna().all(axis=1)

    X_valid = X.loc[valid_mask]

    print()
    print(f"Feature count: {len(FEATURES)}")
    print(
        f"Valid rows   : {len(X_valid):,} / {len(X):,}"
    )

    if len(X_valid) == 0:
        fail("No valid numeric feature rows available.")

    if len(X_valid) != len(X):
        fail(
            "Production dataset contains invalid feature rows. "
            f"Valid={len(X_valid):,}, Total={len(X):,}"
        )

    print("Numeric matrix: PASS")

    audit["checks"]["feature_schema"] = True
    audit["checks"]["numeric_matrix"] = True
    audit["feature_count"] = len(FEATURES)
    audit["valid_feature_rows"] = int(len(X_valid))

    # ============================================================
    # 5. VALIDATE THRESHOLD CONFIGURATION
    # ============================================================

    banner("5. VALIDATING THRESHOLD CONFIGURATION")

    try:
        with open(
            THRESHOLD_CONFIG,
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        thresholds = get_thresholds(config)

    except Exception as exc:
        fail(f"Threshold configuration validation failed: {exc}")

    print("Threshold configuration keys:")

    for key in config.keys():
        print(f"  - {key}")

    print()
    print(f"24h threshold : {thresholds['24h']}")
    print(f"48h threshold : {thresholds['48h']}")
    print(f"72h threshold : {thresholds['72h']}")

    for horizon in ["24h", "48h", "72h"]:

        expected = EXPECTED_THRESHOLDS[horizon]
        actual = thresholds[horizon]

        if not np.isclose(actual, expected):
            fail(
                f"Unexpected {horizon} threshold. "
                f"Expected={expected}, Actual={actual}"
            )

    print("Threshold configuration: PASS")

    audit["checks"]["threshold_configuration"] = True
    audit["thresholds"] = thresholds

    # ============================================================
    # 6. LOAD AND VALIDATE MODELS
    # ============================================================

    banner("6. LOADING PRODUCTION MODELS")

    models = {}

    for horizon, filename in MODEL_FILES.items():

        path = MODEL_DIR / filename

        print()
        print(f"Loading {horizon} model:")
        print(f"  {filename}")

        try:
            saved_object = joblib.load(path)

            print(
                f"  Saved object type: "
                f"{type(saved_object).__name__}"
            )

            model = extract_model(saved_object)

            print(
                f"  Extracted model type: "
                f"{type(model).__name__}"
            )

        except Exception as exc:
            fail(
                f"Could not load {horizon} model: {exc}"
            )

        if not hasattr(model, "predict_proba"):
            fail(
                f"{horizon} model does not support predict_proba."
            )

        models[horizon] = model

        print("  Model loaded: PASS")

    audit["checks"]["models_load"] = True

    # ============================================================
    # 7. MODEL SCHEMA SANITY TEST
    # ============================================================

    banner("7. VALIDATING MODEL FEATURE SCHEMAS")

    for horizon, model in models.items():

        try:
            expected_count = getattr(
                model,
                "n_features_in_",
                None,
            )

            if expected_count is not None:

                expected_count = int(expected_count)

                if expected_count != len(FEATURES):
                    fail(
                        f"{horizon} feature count mismatch. "
                        f"Model={expected_count}, "
                        f"Prepared={len(FEATURES)}"
                    )

            sample = X_valid.iloc[:1]

            probabilities = model.predict_proba(sample)[:, 1]

            probability = float(probabilities[0])

            if not np.isfinite(probability):
                fail(
                    f"{horizon} model produced non-finite probability."
                )

            if probability < 0 or probability > 1:
                fail(
                    f"{horizon} model probability outside [0,1]."
                )

        except SystemExit:
            raise

        except Exception as exc:
            fail(
                f"{horizon} model schema sanity test failed: {exc}"
            )

        print(f"{horizon.upper()} schema: PASS")

    audit["checks"]["model_schema_sanity"] = True

    # ============================================================
    # 8. VALIDATE PRODUCTION OUTPUT
    # ============================================================

    banner("8. VALIDATING PRODUCTION OUTPUT")

    try:
        results = pd.read_csv(PRODUCTION_RESULTS)
    except Exception as exc:
        fail(
            f"Could not load production results file: {exc}"
        )

    print(f"Rows    : {len(results):,}")
    print(f"Columns : {len(results.columns)}")

    try:
        validate_output(
            results=results,
            dataset_rows=len(df),
        )
    except Exception as exc:
        fail(f"Production output validation failed: {exc}")

    print("Required output columns: PASS")
    print("Row count match        : PASS")
    print("Probability ranges     : PASS")
    print("Prediction values      : PASS")

    audit["checks"]["production_output"] = True

    # ============================================================
    # 9. VALIDATE THRESHOLD CONSISTENCY
    # ============================================================

    banner("9. VALIDATING THRESHOLD CONSISTENCY")

    consistency_summary = {}

    for horizon in ["24h", "48h", "72h"]:

        prob_col = f"prob_{horizon}"
        pred_col = f"pred_{horizon}"

        probabilities = pd.to_numeric(
            results[prob_col],
            errors="coerce",
        ).to_numpy()

        actual_predictions = pd.to_numeric(
            results[pred_col],
            errors="coerce",
        ).astype(int).to_numpy()

        expected_predictions = (
            probabilities >= thresholds[horizon]
        ).astype(int)

        mismatches = int(
            np.sum(
                actual_predictions != expected_predictions
            )
        )

        if mismatches != 0:
            fail(
                f"Threshold consistency failed for {horizon}. "
                f"Mismatches={mismatches}"
            )

        positive_predictions = int(
            actual_predictions.sum()
        )

        positive_rate = float(
            positive_predictions / len(actual_predictions)
        )

        print()
        print(f"{horizon.upper()} threshold: {thresholds[horizon]}")
        print(f"Positive predictions: {positive_predictions:,}")
        print(f"Positive rate       : {positive_rate:.4%}")
        print("Threshold consistency: PASS")

        consistency_summary[horizon] = {
            "threshold": thresholds[horizon],
            "positive_predictions": positive_predictions,
            "positive_rate": positive_rate,
            "mismatches": mismatches,
        }

    audit["checks"]["threshold_consistency"] = True
    audit["production_summary"] = consistency_summary

    # ============================================================
    # 10. FINAL READINESS
    # ============================================================

    banner("10. FINAL PRODUCTION READINESS CHECK")

    all_checks_pass = all(
        bool(value)
        for value in audit["checks"].values()
    )

    if not all_checks_pass:
        fail("One or more readiness checks failed.")

    audit["status"] = "PASS"
    audit["ready_for_production_inference"] = True

    # ============================================================
    # SAVE REPORT
    # ============================================================

    banner("SAVING PRODUCTION READINESS REPORT")

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Report : {REPORT}")

    # ============================================================
    # FINAL RESULT
    # ============================================================

    banner("FINAL RESULT")

    print("All required files             : PASS")
    print("Dataset schema                 : PASS")
    print("Production feature schema      : PASS")
    print("Numeric feature matrix         : PASS")
    print("Threshold configuration        : PASS")
    print("Production models              : PASS")
    print("Model schema sanity            : PASS")
    print("Production output schema       : PASS")
    print("Probability validation         : PASS")
    print("Prediction validation          : PASS")
    print("Threshold consistency          : PASS")

    print()
    print("Models modified       : NO")
    print("Retraining performed  : NO")
    print("Dataset modified      : NO")
    print("Synthetic data        : NO")
    print("Fabricated labels     : NO")

    print()
    print("STATUS: 🟢 PRODUCTION READINESS PASS")
    print("READY FOR: PRODUCTION INFERENCE INTEGRATION")


if __name__ == "__main__":
    main()