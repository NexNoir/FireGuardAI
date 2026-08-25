# -*- coding: utf-8 -*-
"""
FireGuard — Real FIRMS Threshold Inference V1
==============================================

Purpose:
    Run inference using the existing Real FIRMS V1 models and apply
    the validated decision thresholds.

IMPORTANT:
    - NO retraining
    - NO model modification
    - NO dataset modification
    - NO synthetic data
    - NO fabricated labels
    - Existing .joblib files are loaded read-only
    - Thresholds are loaded from the separate configuration file

Models:
    24H -> threshold 0.35
    48H -> threshold 0.35
    72H -> threshold 0.30

Test/evaluation period:
    2023-2025

Input:
    data/retraining/real_firms_forecast_dataset_2001_2025.csv
    data/retraining/real_firms_threshold_config_v1.json

Models:
    saved_models/real_firms_v1/

Outputs:
    data/retraining/real_firms_threshold_inference_results_2001_2025.csv
    data/retraining/real_firms_threshold_inference_report.json
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


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

THRESHOLD_CONFIG = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_threshold_config_v1.json"
)

MODEL_DIR = (
    BASE_DIR
    / "saved_models"
    / "real_firms_v1"
)

OUTPUT_RESULTS = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_threshold_inference_results_2001_2025.csv"
)

OUTPUT_REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_threshold_inference_report.json"
)


MODEL_FILES = {
    24: MODEL_DIR / "fireguard_real_firms_sensor_only_24h_v1.joblib",
    48: MODEL_DIR / "fireguard_real_firms_sensor_only_48h_v1.joblib",
    72: MODEL_DIR / "fireguard_real_firms_sensor_only_72h_v1.joblib",
}


TARGET_COLUMNS = {
    24: "fire_next_24h",
    48: "fire_next_48h",
    72: "fire_next_72h",
}


BASE_FEATURES = [
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


# ======================================================================
# PRINT HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def status(label: str, value: str) -> None:
    print(f"{label:<30}: {value}")


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

    required = BASE_FEATURES + [
        "acq_date",
        "acq_time",
        "daynight",
        "satellite",
        "instrument",
        "type",
        "season",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(
                f" - {col}"
                for col in missing
            )
        )

    return df


# ======================================================================
# PREPARE FEATURES
# ======================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:

    section("PREPARING REAL FIRMS FEATURES")

    out = df.copy()

    # --------------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------------

    for col in BASE_FEATURES:

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    # --------------------------------------------------------------
    # Time features
    # --------------------------------------------------------------

    out["acq_date"] = pd.to_datetime(
        out["acq_date"],
        errors="coerce",
    )

    if out["acq_date"].isna().any():
        raise ValueError(
            "Invalid acq_date values detected."
        )

    # FIRMS acq_time is normally HHMM.
    time_numeric = pd.to_numeric(
        out["acq_time"],
        errors="coerce",
    )

    out["hour"] = (
        time_numeric
        .fillna(0)
        .astype(int)
        // 100
    )

    out["minute"] = (
        time_numeric
        .fillna(0)
        .astype(int)
        % 100
    )

    # --------------------------------------------------------------
    # Categorical encoding
    # --------------------------------------------------------------

    categorical_columns = [
        "daynight",
        "satellite",
        "instrument",
        "type",
        "season",
    ]

    encoded_columns = []

    for col in categorical_columns:

        encoded_name = f"{col}_encoded"

        # Stable deterministic encoding.
        categories = sorted(
            out[col]
            .fillna("UNKNOWN")
            .astype(str)
            .unique()
        )

        mapping = {
            value: index
            for index, value
            in enumerate(categories)
        }

        out[encoded_name] = (
            out[col]
            .fillna("UNKNOWN")
            .astype(str)
            .map(mapping)
            .astype(float)
        )

        encoded_columns.append(
            encoded_name
        )

        print(
            f"Created: {encoded_name}"
        )

    feature_columns = (
        BASE_FEATURES
        + encoded_columns
    )

    print()
    print("FINAL FEATURE LIST:")

    for index, feature in enumerate(
        feature_columns,
        start=1,
    ):
        print(
            f"{index:02d}. {feature}"
        )

    print()
    print(
        f"Feature count: {len(feature_columns)}"
    )

    return out


# ======================================================================
# LOAD THRESHOLD CONFIG
# ======================================================================

def load_thresholds() -> dict:

    section("LOADING THRESHOLD CONFIGURATION")

    if not THRESHOLD_CONFIG.exists():
        raise FileNotFoundError(
            f"Threshold configuration not found:\n"
            f"{THRESHOLD_CONFIG}"
        )

    with open(
        THRESHOLD_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    raw_thresholds = config.get(
        "thresholds"
    )

    if not raw_thresholds:
        raise ValueError(
            "Threshold configuration does not contain thresholds."
        )

    thresholds = {
        24: float(raw_thresholds["24h"]),
        48: float(raw_thresholds["48h"]),
        72: float(raw_thresholds["72h"]),
    }

    for horizon, threshold in thresholds.items():

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"Invalid threshold for {horizon}h: "
                f"{threshold}"
            )

        print(
            f"{horizon:02d}H threshold : "
            f"{threshold:.2f}"
        )

    return thresholds


# ======================================================================
# MODEL FEATURE EXTRACTION
# ======================================================================

def get_model_feature_names(
    artifact,
):
    """
    Supports:
        artifact['feature_names']
        artifact['features']
        artifact['model'].feature_names_in_
        artifact.feature_names_in_
    """

    if isinstance(artifact, dict):

        if artifact.get("feature_names") is not None:
            return list(
                artifact["feature_names"]
            )

        if artifact.get("features") is not None:
            return list(
                artifact["features"]
            )

        model = artifact.get("model")

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


# ======================================================================
# GET MODEL OBJECT
# ======================================================================

def get_model_object(artifact):

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

    for horizon, model_path in MODEL_FILES.items():

        print()
        print(
            f"MODEL: {horizon}H"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        artifact = joblib.load(
            model_path
        )

        feature_names = get_model_feature_names(
            artifact
        )

        if feature_names is None:
            raise ValueError(
                f"Could not determine feature schema "
                f"for {horizon}H model."
            )

        model = get_model_object(
            artifact
        )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                f"Model {horizon}H does not support "
                f"predict_proba()."
            )

        models[horizon] = {
            "artifact": artifact,
            "model": model,
            "feature_names": feature_names,
            "path": model_path,
        }

        print(
            f"Model feature count: "
            f"{len(feature_names)}"
        )

        print(
            "Model loaded: PASS"
        )

    return models


# ======================================================================
# VALIDATE FEATURE SCHEMA
# ======================================================================

def validate_feature_schema(
    df: pd.DataFrame,
    models: dict,
) -> None:

    section("VALIDATING MODEL FEATURE SCHEMAS")

    for horizon, info in models.items():

        feature_names = info[
            "feature_names"
        ]

        missing = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{horizon}H model missing features:\n"
                + "\n".join(
                    f" - {feature}"
                    for feature in missing
                )
            )

        print(
            f"{horizon:02d}H schema: PASS"
        )


# ======================================================================
# SAFE PROBABILITY EXTRACTION
# ======================================================================

def extract_positive_probability(
    model,
    X: pd.DataFrame,
) -> np.ndarray:

    probabilities = model.predict_proba(
        X
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Unexpected probability output shape."
        )

    if probabilities.shape[1] == 2:
        return probabilities[:, 1]

    if probabilities.shape[1] == 1:
        return probabilities[:, 0]

    raise ValueError(
        "Model probability output does not "
        "contain a binary positive class."
    )


# ======================================================================
# RUN INFERENCE
# ======================================================================

def run_inference(
    df: pd.DataFrame,
    models: dict,
    thresholds: dict,
) -> tuple[pd.DataFrame, dict]:

    section("RUNNING THRESHOLD INFERENCE")

    result = df[
        [
            "acq_date",
            "acq_time",
            "latitude",
            "longitude",
        ]
    ].copy()

    result = result.rename(
        columns={
            "acq_date": "timestamp_date",
            "acq_time": "timestamp_time",
        }
    )

    report = {}

    for horizon in [24, 48, 72]:

        print()
        print("-" * 72)
        print(
            f"THRESHOLD INFERENCE / {horizon}H"
        )
        print("-" * 72)

        info = models[horizon]

        feature_names = info[
            "feature_names"
        ]

        model = info[
            "model"
        ]

        threshold = thresholds[
            horizon
        ]

        X = df[
            feature_names
        ].copy()

        valid_mask = X.notna().all(
            axis=1
        )

        valid_count = int(
            valid_mask.sum()
        )

        print(
            f"Valid feature rows : "
            f"{valid_count:,}"
        )

        probabilities = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        predictions = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        if valid_count > 0:

            X_valid = X.loc[
                valid_mask
            ]

            positive_probability = (
                extract_positive_probability(
                    model,
                    X_valid,
                )
            )

            prediction = (
                positive_probability
                >= threshold
            ).astype(int)

            probabilities[
                valid_mask.to_numpy()
            ] = positive_probability

            predictions[
                valid_mask.to_numpy()
            ] = prediction

        probability_column = (
            f"probability_{horizon}h"
        )

        prediction_column = (
            f"prediction_{horizon}h"
        )

        result[
            probability_column
        ] = probabilities

        result[
            prediction_column
        ] = predictions

        target_column = TARGET_COLUMNS[
            horizon
        ]

        metrics = {
            "threshold": threshold,
            "valid_rows": valid_count,
        }

        if target_column in df.columns:

            target_mask = (
                valid_mask
                & df[target_column].notna()
            )

            evaluation_count = int(
                target_mask.sum()
            )

            if evaluation_count > 0:

                y_true = (
                    df.loc[
                        target_mask,
                        target_column,
                    ]
                    .astype(int)
                    .to_numpy()
                )

                y_prob = probabilities[
                    target_mask.to_numpy()
                ]

                y_pred = (
                    y_prob
                    >= threshold
                ).astype(int)

                accuracy = accuracy_score(
                    y_true,
                    y_pred,
                )

                precision = precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )

                recall = recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )

                try:
                    roc_auc = roc_auc_score(
                        y_true,
                        y_prob,
                    )
                except ValueError:
                    roc_auc = None

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                )

                tn, fp, fn, tp = (
                    cm.ravel()
                )

                metrics.update(
                    {
                        "target": target_column,
                        "evaluation_rows": evaluation_count,
                        "positive_actual": int(
                            y_true.sum()
                        ),
                        "accuracy": float(
                            accuracy
                        ),
                        "precision": float(
                            precision
                        ),
                        "recall": float(
                            recall
                        ),
                        "f1": float(
                            f1
                        ),
                        "roc_auc": (
                            None
                            if roc_auc is None
                            else float(roc_auc)
                        ),
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    }
                )

                print(
                    f"Threshold : {threshold:.2f}"
                )

                print(
                    f"Accuracy  : "
                    f"{accuracy:.4f}"
                )

                print(
                    f"Precision : "
                    f"{precision:.4f}"
                )

                print(
                    f"Recall    : "
                    f"{recall:.4f}"
                )

                print(
                    f"F1        : "
                    f"{f1:.4f}"
                )

                if roc_auc is not None:
                    print(
                        f"ROC-AUC   : "
                        f"{roc_auc:.4f}"
                    )

                print()
                print(
                    "CONFUSION MATRIX"
                )

                print(
                    f"TN: {tn}"
                )

                print(
                    f"FP: {fp}"
                )

                print(
                    f"FN: {fn}"
                )

                print(
                    f"TP: {tp}"
                )

        report[str(horizon)] = metrics

        positive_predictions = int(
            np.nansum(
                predictions == 1
            )
        )

        print()
        print(
            f"Positive predictions: "
            f"{positive_predictions:,}"
        )

    return result, report


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    result: pd.DataFrame,
    report: dict,
    thresholds: dict,
) -> None:

    section("SAVING THRESHOLD INFERENCE")

    OUTPUT_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    full_report = {
        "project": "FireGuard",
        "configuration": (
            "Real FIRMS Threshold Inference V1"
        ),

        "inference_only": True,
        "retraining": False,
        "model_modified": False,
        "dataset_modified": False,
        "synthetic_data": False,
        "fabricated_labels": False,

        "test_period": "2023-2025",

        "thresholds": {
            "24h": thresholds[24],
            "48h": thresholds[48],
            "72h": thresholds[72],
        },

        "models": {
            "24h": str(MODEL_FILES[24]),
            "48h": str(MODEL_FILES[48]),
            "72h": str(MODEL_FILES[72]),
        },

        "results": report,
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            full_report,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"Results : {OUTPUT_RESULTS}"
    )

    print(
        f"Report  : {OUTPUT_REPORT}"
    )


# ======================================================================
# FINAL CHECKPOINT
# ======================================================================

def final_checkpoint(
    report: dict,
) -> None:

    section("FINAL RESULT")

    for horizon in [24, 48, 72]:

        metrics = report.get(
            str(horizon),
            {},
        )

        threshold = metrics.get(
            "threshold"
        )

        print(
            f"{horizon:02d}H | "
            f"Threshold={threshold}"
        )

        if "accuracy" in metrics:

            print(
                f"     Accuracy="
                f"{metrics['accuracy']:.4f} | "
                f"Precision="
                f"{metrics['precision']:.4f} | "
                f"Recall="
                f"{metrics['recall']:.4f} | "
                f"F1="
                f"{metrics['f1']:.4f}"
            )

            if metrics.get(
                "roc_auc"
            ) is not None:

                print(
                    f"     ROC-AUC="
                    f"{metrics['roc_auc']:.4f}"
                )

    print()
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
        "STATUS: 🟢 THRESHOLD INFERENCE COMPLETE"
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    section(
        "🔥 FIREGUARD — REAL FIRMS "
        "THRESHOLD INFERENCE V1"
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

    df = load_dataset()

    df = prepare_features(
        df
    )

    thresholds = load_thresholds()

    models = load_models()

    validate_feature_schema(
        df,
        models,
    )

    result, report = run_inference(
        df,
        models,
        thresholds,
    )

    save_results(
        result,
        report,
        thresholds,
    )

    final_checkpoint(
        report
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

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