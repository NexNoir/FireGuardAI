# ============================================================================
# FIREGUARD — REAL FIRMS PRODUCTION SMOKE TEST V1
# ============================================================================
# Purpose:
#   Validate the production inference integration end-to-end.
#
# Safety guarantees:
#   - NO retraining
#   - NO model modification
#   - NO dataset modification
#   - NO synthetic data
#   - NO fabricated labels
#
# The script:
#   1. Loads existing production models
#   2. Loads existing threshold configuration
#   3. Loads real FIRMS production dataset
#   4. Builds the exact required feature schema
#   5. Runs inference on a small real-data sample
#   6. Validates probabilities and threshold predictions
#   7. Tests a reusable production inference function
#   8. Saves only smoke-test output/report files
# ============================================================================

import os
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_DIR = r"C:\Users\vista\Desktop\fireguard_v2.0"

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "retraining",
    "real_firms_forecast_dataset_2001_2025.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "real_firms_v1"
)

THRESHOLD_CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "retraining",
    "real_firms_threshold_config_v1.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "retraining"
)

SMOKE_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "real_firms_production_smoke_test_results.csv"
)

SMOKE_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "real_firms_production_smoke_test_report.json"
)


MODEL_FILES = {
    "24h": "fireguard_real_firms_sensor_only_24h_v1.joblib",
    "48h": "fireguard_real_firms_sensor_only_48h_v1.joblib",
    "72h": "fireguard_real_firms_sensor_only_72h_v1.joblib",
}


FEATURE_COLUMNS = [
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


SAMPLE_SIZE = 100


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def line(char="=", length=72):
    print(char * length)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


def sub_section(title):
    print()
    line("-")
    print(title)
    line("-")


# ============================================================================
# FILE VALIDATION
# ============================================================================

def require_file(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {name}\n{path}"
        )


def require_directory(path, name):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Required directory not found: {name}\n{path}"
        )


# ============================================================================
# MODEL EXTRACTION
# ============================================================================

def extract_model(saved_object):
    """
    Supports the existing FIREGUARD joblib structure.

    The current saved files may contain:
        - model directly
        - dict containing model
        - dict containing pipeline
        - dict containing estimator
    """

    if isinstance(saved_object, dict):

        preferred_keys = [
            "model",
            "pipeline",
            "estimator",
            "classifier",
            "best_model"
        ]

        for key in preferred_keys:
            if key in saved_object:
                candidate = saved_object[key]

                if hasattr(candidate, "predict_proba"):
                    return candidate

        for value in saved_object.values():
            if hasattr(value, "predict_proba"):
                return value

        raise ValueError(
            "Saved dictionary does not contain an object "
            "with predict_proba()."
        )

    if hasattr(saved_object, "predict_proba"):
        return saved_object

    raise ValueError(
        f"Unsupported saved model type: "
        f"{type(saved_object).__name__}"
    )


# ============================================================================
# THRESHOLD CONFIGURATION
# ============================================================================

def load_threshold_config(path):

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "thresholds" not in config:
        raise KeyError(
            "Threshold configuration does not contain 'thresholds'."
        )

    raw_thresholds = config["thresholds"]

    thresholds = {}

    aliases = {
        "24h": ["24h", "24H", "fire_next_24h"],
        "48h": ["48h", "48H", "fire_next_48h"],
        "72h": ["72h", "72H", "fire_next_72h"],
    }

    for horizon, keys in aliases.items():

        found = False

        for key in keys:
            if key in raw_thresholds:
                thresholds[horizon] = float(raw_thresholds[key])
                found = True
                break

        if not found:
            raise KeyError(
                f"Threshold for {horizon} not found in configuration."
            )

    for horizon, threshold in thresholds.items():

        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"Invalid threshold for {horizon}: {threshold}"
            )

    return config, thresholds


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def encode_column_as_categories(df, column_name):

    if column_name not in df.columns:
        raise KeyError(
            f"Required source column missing for encoding: {column_name}"
        )

    values = (
        df[column_name]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    categories = sorted(values.unique())

    mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    encoded = values.map(mapping).astype(float)

    return encoded


def prepare_features(df):

    work = df.copy()

    created = []

    # ------------------------------------------------------------------------
    # Temporal columns
    # ------------------------------------------------------------------------

    if "hour" not in work.columns:

        if "acq_time" not in work.columns:
            raise KeyError(
                "Missing both 'hour' and 'acq_time'."
            )

        acq_time = (
            work["acq_time"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )

        work["hour"] = pd.to_numeric(
            acq_time.str.slice(0, 2),
            errors="coerce"
        )

        created.append("hour")

    if "minute" not in work.columns:

        if "acq_time" not in work.columns:
            raise KeyError(
                "Missing both 'minute' and 'acq_time'."
            )

        acq_time = (
            work["acq_time"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )

        work["minute"] = pd.to_numeric(
            acq_time.str.slice(2, 4),
            errors="coerce"
        )

        created.append("minute")

    # ------------------------------------------------------------------------
    # Day/Night
    # ------------------------------------------------------------------------

    if "daynight_encoded" not in work.columns:

        if "daynight" not in work.columns:
            raise KeyError(
                "Missing both 'daynight_encoded' and 'daynight'."
            )

        values = (
            work["daynight"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        mapping = {
            "D": 1,
            "DAY": 1,
            "N": 0,
            "NIGHT": 0,
        }

        work["daynight_encoded"] = (
            values.map(mapping)
            .fillna(-1)
            .astype(float)
        )

        created.append("daynight_encoded")

    # ------------------------------------------------------------------------
    # Satellite
    # ------------------------------------------------------------------------

    if "satellite_encoded" not in work.columns:

        work["satellite_encoded"] = encode_column_as_categories(
            work,
            "satellite"
        )

        created.append("satellite_encoded")

    # ------------------------------------------------------------------------
    # Instrument
    # ------------------------------------------------------------------------

    if "instrument_encoded" not in work.columns:

        work["instrument_encoded"] = encode_column_as_categories(
            work,
            "instrument"
        )

        created.append("instrument_encoded")

    # ------------------------------------------------------------------------
    # Type
    # ------------------------------------------------------------------------

    if "type_encoded" not in work.columns:

        work["type_encoded"] = encode_column_as_categories(
            work,
            "type"
        )

        created.append("type_encoded")

    # ------------------------------------------------------------------------
    # Season
    # ------------------------------------------------------------------------

    if "season_encoded" not in work.columns:

        date_column = None

        for candidate in ["detection_datetime", "acq_date"]:
            if candidate in work.columns:
                date_column = candidate
                break

        if date_column is None:
            raise KeyError(
                "Cannot create season_encoded. "
                "No detection_datetime or acq_date found."
            )

        dates = pd.to_datetime(
            work[date_column],
            errors="coerce"
        )

        if dates.isna().any():
            raise ValueError(
                "Invalid dates found while creating season_encoded."
            )

        month = dates.dt.month

        def get_season(month_value):

            if month_value in [12, 1, 2]:
                return 0

            if month_value in [3, 4, 5]:
                return 1

            if month_value in [6, 7, 8]:
                return 2

            return 3

        work["season_encoded"] = month.map(
            get_season
        ).astype(float)

        created.append("season_encoded")

    # ------------------------------------------------------------------------
    # Validate required features
    # ------------------------------------------------------------------------

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in work.columns
    ]

    if missing:
        raise KeyError(
            "Required feature columns missing:\n - "
            + "\n - ".join(missing)
        )

    X = work[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    invalid_mask = X.isna().any(axis=1)

    if invalid_mask.any():

        invalid_count = int(invalid_mask.sum())

        raise ValueError(
            f"Smoke-test sample contains "
            f"{invalid_count} invalid feature rows."
        )

    X = X.astype(float)

    return work, X, created


# ============================================================================
# MODEL FEATURE SCHEMA VALIDATION
# ============================================================================

def validate_model_schema(model, feature_columns, horizon):

    expected_count = len(feature_columns)

    if hasattr(model, "n_features_in_"):

        actual_count = int(model.n_features_in_)

        if actual_count != expected_count:
            raise ValueError(
                f"{horizon} model expects "
                f"{actual_count} features but production schema has "
                f"{expected_count}."
            )

    if hasattr(model, "feature_names_in_"):

        actual_names = list(model.feature_names_in_)

        if actual_names != feature_columns:
            raise ValueError(
                f"{horizon} model feature names do not match "
                "production feature schema."
            )


# ============================================================================
# REUSABLE PRODUCTION INFERENCE FUNCTION
# ============================================================================

def run_production_inference(
    input_dataframe,
    models,
    thresholds
):
    """
    Reusable production inference entry point.

    Returns:
        dataframe containing the original input plus:
            prob_24h / pred_24h
            prob_48h / pred_48h
            prob_72h / pred_72h
    """

    prepared_df, X, _ = prepare_features(input_dataframe)

    output = prepared_df.copy()

    for horizon in ["24h", "48h", "72h"]:

        model = models[horizon]
        threshold = thresholds[horizon]

        validate_model_schema(
            model,
            FEATURE_COLUMNS,
            horizon
        )

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        output[f"prob_{horizon}"] = probabilities
        output[f"pred_{horizon}"] = predictions

    return output


# ============================================================================
# MAIN
# ============================================================================

def main():

    section("🔥 FIREGUARD — REAL FIRMS PRODUCTION SMOKE TEST V1")

    print("Production integration validation only.")
    print("Retraining          : NO")
    print("Model modification  : NO")
    print("Dataset modification: NO")
    print("Synthetic data      : NO")
    print("Fabricated labels   : NO")
    print()

    print(f"Dataset       : {DATASET_PATH}")
    print(f"Model dir     : {MODEL_DIR}")
    print(f"Threshold cfg : {THRESHOLD_CONFIG_PATH}")
    print(f"Sample size   : {SAMPLE_SIZE}")

    report = {
        "project": "FIREGUARD",
        "audit": "REAL FIRMS PRODUCTION SMOKE TEST V1",
        "timestamp": datetime.now().isoformat(),
        "retraining": False,
        "model_modified": False,
        "dataset_modified": False,
        "synthetic_data": False,
        "fabricated_labels": False,
        "status": "STARTED"
    }

    # ========================================================================
    # 1. CHECK REQUIRED FILES
    # ========================================================================

    section("1. CHECKING REQUIRED FILES")

    require_file(DATASET_PATH, "production dataset")
    print("dataset             : FOUND")

    require_directory(MODEL_DIR, "model directory")
    print("model_directory     : FOUND")

    require_file(
        THRESHOLD_CONFIG_PATH,
        "threshold configuration"
    )
    print("threshold_config    : FOUND")

    for horizon, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        require_file(
            path,
            f"{horizon} model"
        )

        print(f"model_{horizon:<3}         : FOUND")

    # ========================================================================
    # 2. LOAD REAL DATA
    # ========================================================================

    section("2. LOADING REAL FIRMS PRODUCTION DATA")

    df = pd.read_csv(DATASET_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    if len(df) == 0:
        raise ValueError("Dataset is empty.")

    # ========================================================================
    # 3. SELECT REAL SAMPLE
    # ========================================================================

    section("3. SELECTING REAL-DATA SMOKE TEST SAMPLE")

    actual_sample_size = min(SAMPLE_SIZE, len(df))

    sample_df = (
        df
        .sort_index()
        .head(actual_sample_size)
        .copy()
    )

    print(f"Source rows       : {len(df):,}")
    print(f"Smoke-test rows   : {len(sample_df):,}")
    print("Synthetic rows    : 0")
    print("Dataset modified  : NO")

    # ========================================================================
    # 4. PREPARE FEATURES
    # ========================================================================

    section("4. PREPARING PRODUCTION FEATURES")

    prepared_sample, X_sample, created = prepare_features(
        sample_df
    )

    if created:
        for column in created:
            print(f"Created in memory: {column}")
    else:
        print("No derived feature columns needed.")

    print()
    print("FINAL FEATURE LIST:")

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1
    ):
        print(f"{index:02d}. {feature}")

    print()
    print(f"Feature count : {len(FEATURE_COLUMNS)}")
    print(f"Valid rows    : {len(X_sample):,}")
    print("Numeric matrix: PASS")

    # ========================================================================
    # 5. LOAD THRESHOLDS
    # ========================================================================

    section("5. LOADING THRESHOLD CONFIGURATION")

    config, thresholds = load_threshold_config(
        THRESHOLD_CONFIG_PATH
    )

    for horizon in ["24h", "48h", "72h"]:
        print(
            f"{horizon.upper()} threshold : "
            f"{thresholds[horizon]}"
        )

    print()
    print("Threshold configuration: PASS")

    # ========================================================================
    # 6. LOAD MODELS
    # ========================================================================

    section("6. LOADING PRODUCTION MODELS")

    models = {}

    for horizon in ["24h", "48h", "72h"]:

        filename = MODEL_FILES[horizon]
        path = os.path.join(MODEL_DIR, filename)

        print()
        print(f"Loading {horizon} model:")
        print(f"  {filename}")

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

        validate_model_schema(
            model,
            FEATURE_COLUMNS,
            horizon
        )

        models[horizon] = model

        print("  Model loaded: PASS")
        print("  Schema      : PASS")

    # ========================================================================
    # 7. DIRECT PREDICTION TEST
    # ========================================================================

    section("7. RUNNING DIRECT PRODUCTION PREDICTION TEST")

    direct_results = {}

    for horizon in ["24h", "48h", "72h"]:

        sub_section(f"DIRECT TEST / {horizon.upper()}")

        model = models[horizon]
        threshold = thresholds[horizon]

        probabilities = model.predict_proba(
            X_sample
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{horizon} produced non-finite probabilities."
            )

        if (probabilities < 0).any() or (
            probabilities > 1
        ).any():
            raise ValueError(
                f"{horizon} probabilities are outside [0, 1]."
            )

        unique_predictions = set(
            np.unique(predictions).tolist()
        )

        if not unique_predictions.issubset({0, 1}):
            raise ValueError(
                f"{horizon} predictions are invalid."
            )

        positive_count = int(predictions.sum())

        print(
            f"Rows                : "
            f"{len(probabilities):,}"
        )

        print(
            f"Threshold           : {threshold}"
        )

        print(
            f"Mean probability    : "
            f"{float(np.mean(probabilities)):.4f}"
        )

        print(
            f"Min probability     : "
            f"{float(np.min(probabilities)):.4f}"
        )

        print(
            f"Max probability     : "
            f"{float(np.max(probabilities)):.4f}"
        )

        print(
            f"Positive predictions: "
            f"{positive_count:,}"
        )

        print(
            f"Positive rate       : "
            f"{positive_count / len(predictions) * 100:.4f}%"
        )

        print("Probability validation: PASS")
        print("Prediction validation : PASS")

        direct_results[horizon] = {
            "probabilities": probabilities,
            "predictions": predictions,
            "positive_count": positive_count
        }

    # ========================================================================
    # 8. REUSABLE FUNCTION TEST
    # ========================================================================

    section("8. TESTING REUSABLE PRODUCTION INFERENCE FUNCTION")

    function_output = run_production_inference(
        sample_df,
        models,
        thresholds
    )

    required_output_columns = []

    for horizon in ["24h", "48h", "72h"]:
        required_output_columns.extend([
            f"prob_{horizon}",
            f"pred_{horizon}"
        ])

    missing_output_columns = [
        col for col in required_output_columns
        if col not in function_output.columns
    ]

    if missing_output_columns:
        raise ValueError(
            "Reusable inference output missing columns:\n - "
            + "\n - ".join(missing_output_columns)
        )

    if len(function_output) != len(sample_df):
        raise ValueError(
            "Reusable inference changed row count."
        )

    for horizon in ["24h", "48h", "72h"]:

        probability_column = f"prob_{horizon}"
        prediction_column = f"pred_{horizon}"

        function_probabilities = (
            function_output[probability_column]
            .to_numpy()
        )

        function_predictions = (
            function_output[prediction_column]
            .to_numpy()
        )

        direct_probabilities = (
            direct_results[horizon]["probabilities"]
        )

        direct_predictions = (
            direct_results[horizon]["predictions"]
        )

        probabilities_match = np.allclose(
            function_probabilities,
            direct_probabilities,
            rtol=0.0,
            atol=1e-12
        )

        predictions_match = np.array_equal(
            function_predictions,
            direct_predictions
        )

        if not probabilities_match:
            raise ValueError(
                f"{horizon} reusable function probabilities "
                "do not match direct inference."
            )

        if not predictions_match:
            raise ValueError(
                f"{horizon} reusable function predictions "
                "do not match direct inference."
            )

        print(f"{horizon.upper()} function equivalence: PASS")

    print()
    print("Reusable production inference function: PASS")

    # ========================================================================
    # 9. VALIDATE OUTPUT
    # ========================================================================

    section("9. VALIDATING SMOKE TEST OUTPUT")

    if len(function_output) != actual_sample_size:
        raise ValueError(
            "Output row count does not match smoke-test sample."
        )

    for horizon in ["24h", "48h", "72h"]:

        probability_column = f"prob_{horizon}"
        prediction_column = f"pred_{horizon}"

        probabilities = function_output[
            probability_column
        ].to_numpy()

        predictions = function_output[
            prediction_column
        ].to_numpy()

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Invalid output probabilities for {horizon}."
            )

        if (probabilities < 0).any() or (
            probabilities > 1
        ).any():
            raise ValueError(
                f"Out-of-range output probabilities for {horizon}."
            )

        if not set(
            np.unique(predictions).tolist()
        ).issubset({0, 1}):
            raise ValueError(
                f"Invalid output predictions for {horizon}."
            )

    print(f"Output rows             : {len(function_output):,}")
    print(
        f"Required output columns : "
        f"{len(required_output_columns)}"
    )
    print("Output schema           : PASS")
    print("Probability ranges      : PASS")
    print("Prediction values       : PASS")

    # ========================================================================
    # 10. SAVE OUTPUT
    # ========================================================================

    section("10. SAVING SMOKE TEST RESULTS")

    function_output.to_csv(
        SMOKE_RESULTS_PATH,
        index=False
    )

    print(f"Results : {SMOKE_RESULTS_PATH}")

    # ========================================================================
    # BUILD REPORT
    # ========================================================================

    report["status"] = "PASS"
    report["sample_size"] = int(actual_sample_size)
    report["feature_count"] = int(len(FEATURE_COLUMNS))
    report["feature_columns"] = FEATURE_COLUMNS
    report["thresholds"] = thresholds
    report["dataset_path"] = DATASET_PATH
    report["model_directory"] = MODEL_DIR
    report["threshold_config"] = THRESHOLD_CONFIG_PATH
    report["results_path"] = SMOKE_RESULTS_PATH
    report["models_modified"] = False
    report["dataset_modified"] = False
    report["retraining_performed"] = False
    report["synthetic_data"] = False
    report["fabricated_labels"] = False
    report["results"] = {}

    for horizon in ["24h", "48h", "72h"]:

        probabilities = direct_results[horizon][
            "probabilities"
        ]

        predictions = direct_results[horizon][
            "predictions"
        ]

        report["results"][horizon] = {
            "threshold": float(thresholds[horizon]),
            "rows": int(len(probabilities)),
            "mean_probability": float(
                np.mean(probabilities)
            ),
            "min_probability": float(
                np.min(probabilities)
            ),
            "max_probability": float(
                np.max(probabilities)
            ),
            "positive_predictions": int(
                predictions.sum()
            ),
            "positive_rate": float(
                predictions.mean()
            )
        }

    with open(
        SMOKE_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Report  : {SMOKE_REPORT_PATH}")

    # ========================================================================
    # FINAL RESULT
    # ========================================================================

    section("FINAL RESULT")

    for horizon in ["24h", "48h", "72h"]:

        item = report["results"][horizon]

        print(
            f"{horizon.upper()} | "
            f"Threshold={item['threshold']} | "
            f"MeanProb={item['mean_probability']:.4f} | "
            f"PositivePredictions="
            f"{item['positive_predictions']:,} | "
            f"PositiveRate="
            f"{item['positive_rate'] * 100:.4f}%"
        )

    print()
    print("Required files              : PASS")
    print("Production feature schema   : PASS")
    print("Numeric feature matrix      : PASS")
    print("Threshold configuration     : PASS")
    print("Production models           : PASS")
    print("Model feature schemas       : PASS")
    print("Direct inference            : PASS")
    print("Probability validation      : PASS")
    print("Prediction validation       : PASS")
    print("Reusable inference function : PASS")
    print("Direct/function equivalence : PASS")
    print("Output schema               : PASS")

    print()
    print("Models modified       : NO")
    print("Retraining performed  : NO")
    print("Dataset modified      : NO")
    print("Synthetic data        : NO")
    print("Fabricated labels     : NO")

    print()
    print("STATUS: 🟢 PRODUCTION INTEGRATION SMOKE TEST PASS")
    print("READY FOR: APPLICATION / API INTEGRATION")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print()
        line("=")
        print("❌ PRODUCTION SMOKE TEST FAILED")
        line("=")
        print(str(error))
        sys.exit(1)