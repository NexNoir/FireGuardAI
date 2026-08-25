
"""
FireGuard — REAL FIRMS Inference Audit V1
=========================================

Inference only.
NO retraining.
NO model modification.
NO dataset modification.
NO synthetic data.
NO fabricated labels.

Models:
    real_firms_v1
    sensor_only / 24h
    sensor_only / 48h
    sensor_only / 72h

Dataset:
    data/retraining/real_firms_forecast_dataset_2001_2025.csv

IMPORTANT:
    The models were trained with 15 features.
    Five encoded categorical features are reconstructed here
    from the original real FIRMS columns.

    Encoding is deterministic and based on sorted category values.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "saved_models"
    / "real_firms_v1"
)

REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_inference_audit_report.json"
)

OUTPUT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_inference_results_2001_2025.csv"
)


MODEL_SPECS = {
    24:
        MODEL_DIR
        / "fireguard_real_firms_sensor_only_24h_v1.joblib",

    48:
        MODEL_DIR
        / "fireguard_real_firms_sensor_only_48h_v1.joblib",

    72:
        MODEL_DIR
        / "fireguard_real_firms_sensor_only_72h_v1.joblib",
}


RAW_REQUIRED = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "bright_t31",
    "frp",
    "daynight",
    "type",
    "season",
]


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


TARGETS = {
    24: "fire_next_24h",
    48: "fire_next_48h",
    72: "fire_next_72h",
}


# ======================================================================
# HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def status(label: str, value: str) -> None:
    print(f"{label:<34}: {value}")


# ======================================================================
# ENCODING
# ======================================================================

def deterministic_encode(
    series: pd.Series,
) -> tuple[pd.Series, dict]:

    values = (
        series
        .astype(str)
        .fillna("")
        .unique()
        .tolist()
    )

    values = sorted(values)

    mapping = {
        value: index
        for index, value in enumerate(values)
    }

    encoded = (
        series
        .astype(str)
        .map(mapping)
    )

    return encoded, mapping


# ======================================================================
# LOAD DATA
# ======================================================================

def load_dataset() -> pd.DataFrame:

    section("LOADING REAL FIRMS DATASET")

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
            "Required columns missing:\n"
            + "\n".join(
                f" - {x}"
                for x in missing
            )
        )

    return df


# ======================================================================
# PREPARE FEATURES
# ======================================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    section("PREPARING REAL FIRMS FEATURES")

    out = df.copy()

    # --------------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------------

    date_part = pd.to_datetime(
        out["acq_date"],
        errors="coerce",
    )

    if date_part.isna().any():
        raise ValueError(
            "Invalid acq_date values detected."
        )

    time_text = (
        out["acq_time"]
        .astype(str)
        .str.zfill(4)
    )

    hour = pd.to_numeric(
        time_text.str[:2],
        errors="coerce",
    )

    minute = pd.to_numeric(
        time_text.str[2:4],
        errors="coerce",
    )

    out["hour"] = hour
    out["minute"] = minute

    # --------------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------------

    numeric_features = [
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

    for col in numeric_features:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    # --------------------------------------------------------------
    # Deterministic categorical encoding
    # --------------------------------------------------------------

    mappings = {}

    categorical_sources = {
        "daynight_encoded": "daynight",
        "satellite_encoded": "satellite",
        "instrument_encoded": "instrument",
        "type_encoded": "type",
        "season_encoded": "season",
    }

    for encoded_col, source_col in categorical_sources.items():

        encoded, mapping = deterministic_encode(
            out[source_col]
        )

        out[encoded_col] = encoded

        mappings[encoded_col] = mapping

        print(
            f"Created: {encoded_col}"
        )

    # --------------------------------------------------------------
    # Check all model features
    # --------------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in out.columns
    ]

    if missing_features:
        raise ValueError(
            "Model features could not be created:\n"
            + "\n".join(
                f" - {x}"
                for x in missing_features
            )
        )

    feature_frame = out[FEATURES].copy()

    missing_values = feature_frame.isna().sum()

    bad = missing_values[
        missing_values > 0
    ]

    if len(bad) > 0:

        print()
        print("Missing feature values:")

        print(
            bad.to_string()
        )

        raise ValueError(
            "Feature matrix contains missing values."
        )

    print()
    print("FINAL FEATURE LIST:")

    for index, feature in enumerate(
        FEATURES,
        start=1,
    ):
        print(
            f"{index:02d}. {feature}"
        )

    print()
    print(
        f"Available feature count: "
        f"{len(FEATURES)}"
    )

    return feature_frame, mappings


# ======================================================================
# MODEL FEATURE SCHEMA
# ======================================================================

def extract_model_features(
    artifact,
):

    if isinstance(artifact, dict):

        if artifact.get(
            "feature_names"
        ) is not None:
            return list(
                artifact["feature_names"]
            )

        if artifact.get(
            "features"
        ) is not None:
            return list(
                artifact["features"]
            )

        model = artifact.get(
            "model"
        )

        if (
            model is not None
            and hasattr(
                model,
                "feature_names_in_",
            )
        ):
            return list(
                model.feature_names_in_
            )

    if hasattr(
        artifact,
        "feature_names_in_",
    ):
        return list(
            artifact.feature_names_in_
        )

    return None


def extract_model(
    artifact,
):

    if isinstance(artifact, dict):

        if artifact.get("model") is not None:
            return artifact["model"]

        if artifact.get("estimator") is not None:
            return artifact["estimator"]

    return artifact


# ======================================================================
# LOAD MODELS
# ======================================================================

def load_models():

    section("LOADING REAL FIRMS MODELS")

    models = {}

    for horizon, path in MODEL_SPECS.items():

        print()
        print(
            f"MODEL: {horizon}H"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        artifact = joblib.load(path)

        model = extract_model(
            artifact
        )

        model_features = extract_model_features(
            artifact
        )

        if model_features is None:

            warnings.warn(
                f"Could not extract feature schema "
                f"from {path.name}"
            )

        else:

            print(
                f"Model feature count: "
                f"{len(model_features)}"
            )

            if list(model_features) != FEATURES:

                print(
                    "WARNING: Model feature order "
                    "differs from expected FEATURES."
                )

                print(
                    "Model features:"
                )

                for feature in model_features:
                    print(
                        f" - {feature}"
                    )

        models[horizon] = {
            "model": model,
            "artifact": artifact,
            "feature_names": model_features,
            "path": str(path),
        }

        status(
            "Model",
            "LOADED"
        )

    return models


# ======================================================================
# INFERENCE
# ======================================================================

def run_inference(
    df: pd.DataFrame,
    X: pd.DataFrame,
    models: dict,
) -> tuple[pd.DataFrame, dict]:

    section("RUNNING INFERENCE")

    results = df.copy()

    report = {}

    for horizon, info in models.items():

        print()
        print("-" * 72)
        print(
            f"INFERENCE / {horizon}H"
        )
        print("-" * 72)

        model = info["model"]

        model_features = info["feature_names"]

        if model_features is not None:

            missing = [
                x
                for x in model_features
                if x not in X.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing model features for "
                    f"{horizon}h:\n"
                    + "\n".join(
                        f" - {x}"
                        for x in missing
                    )
                )

            X_model = X[
                model_features
            ]

        else:

            X_model = X[
                FEATURES
            ]

        # ----------------------------------------------------------
        # Probability
        # ----------------------------------------------------------

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                f"Model {horizon}h does not "
                f"support predict_proba()."
            )

        probabilities = model.predict_proba(
            X_model
        )

        if probabilities.shape[1] == 2:

            positive_probability = (
                probabilities[:, 1]
            )

        else:

            raise ValueError(
                f"Unexpected probability shape "
                f"for {horizon}h: "
                f"{probabilities.shape}"
            )

        predictions = (
            positive_probability >= 0.5
        ).astype(int)

        results[
            f"prediction_probability_{horizon}h"
        ] = positive_probability

        results[
            f"prediction_{horizon}h"
        ] = predictions

        # ----------------------------------------------------------
        # Real target comparison
        # ----------------------------------------------------------

        target = TARGETS[horizon]

        if target in results.columns:

            y_true = pd.to_numeric(
                results[target],
                errors="coerce",
            )

            valid = (
                y_true.notna()
            )

            if valid.any():

                actual = y_true[
                    valid
                ].astype(int)

                predicted = predictions[
                    valid.to_numpy()
                ]

                accuracy = float(
                    (
                        actual.to_numpy()
                        == predicted
                    ).mean()
                )

                tp = int(
                    (
                        (actual == 1)
                        & (predicted == 1)
                    ).sum()
                )

                fp = int(
                    (
                        (actual == 0)
                        & (predicted == 1)
                    ).sum()
                )

                fn = int(
                    (
                        (actual == 1)
                        & (predicted == 0)
                    ).sum()
                )

                precision = (
                    tp / (tp + fp)
                    if (tp + fp) > 0
                    else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if (tp + fn) > 0
                    else 0.0
                )

                f1 = (
                    2
                    * precision
                    * recall
                    / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                print(
                    f"Target available       : YES"
                )

                print(
                    f"Accuracy               : "
                    f"{accuracy:.4f}"
                )

                print(
                    f"Precision              : "
                    f"{precision:.4f}"
                )

                print(
                    f"Recall                 : "
                    f"{recall:.4f}"
                )

                print(
                    f"F1                    : "
                    f"{f1:.4f}"
                )

                report[str(horizon)] = {
                    "status": "PASS",
                    "rows": int(len(results)),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            else:

                print(
                    "Target available       : YES"
                )

                print(
                    "Valid target rows      : 0"
                )

                report[str(horizon)] = {
                    "status": "NO_VALID_TARGETS",
                    "rows": int(len(results)),
                }

        else:

            print(
                f"Target {target:<20}: "
                "NOT AVAILABLE"
            )

            report[str(horizon)] = {
                "status": "INFERENCE_ONLY",
                "rows": int(len(results)),
            }

        print(
            f"Predictions generated  : "
            f"{len(predictions):,}"
        )

    return results, report


# ======================================================================
# SAVE
# ======================================================================

def save_results(
    results: pd.DataFrame,
    report: dict,
    mappings: dict,
) -> None:

    section("SAVING INFERENCE AUDIT")

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    final_report = {
        "status": "PASS",
        "dataset": str(DATASET),
        "model_directory": str(MODEL_DIR),
        "output": str(OUTPUT),
        "rows": int(len(results)),
        "features": FEATURES,
        "models": report,
        "encoding_mappings": mappings,
        "retraining": False,
        "model_modification": False,
        "dataset_modification": False,
        "synthetic_data": False,
        "fabricated_labels": False,
    }

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Results : {OUTPUT}"
    )

    print(
        f"Report  : {REPORT}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    section(
        "🔥 FIREGUARD — REAL FIRMS INFERENCE AUDIT V1"
    )

    print(
        "Inference only."
    )

    print(
        "Retraining          : NO"
    )

    print(
        "Model modification  : NO"
    )

    print(
        "Dataset modification: NO"
    )

    print(
        "Synthetic data      : NO"
    )

    print(
        f"Model directory     : {MODEL_DIR}"
    )

    # --------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------------

    X, mappings = prepare_features(
        df
    )

    # --------------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------------

    models = load_models()

    # --------------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------------

    results, report = run_inference(
        df,
        X,
        models,
    )

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    save_results(
        results,
        report,
        mappings,
    )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    section("FINAL RESULT")

    print(
        "Inference completed."
    )

    print(
        "Models modified      : NO"
    )

    print(
        "Dataset modified     : NO"
    )

    print(
        "Retraining performed : NO"
    )

    print(
        "Synthetic data       : NO"
    )

    print(
        "Fabricated labels    : NO"
    )

    print()
    print(
        "STATUS: 🟢 INFERENCE AUDIT COMPLETE"
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
        print("=" * 72)
        print("❌ FATAL ERROR")
        print("=" * 72)

        print(
            str(exc)
        )

        sys.exit(1)