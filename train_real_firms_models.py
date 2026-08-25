
# -*- coding: utf-8 -*-
"""
FireGuard — Real FIRMS Model Training
=====================================

Trains separate real-data forecasting models for:
    - 24 hours
    - 48 hours
    - 72 hours

Training data:
    data/retraining/real_firms_forecast_dataset_2001_2025.csv

Time split:
    Train      : 2001-2020
    Validation : 2021-2022
    Test       : 2023-2025

IMPORTANT:
    - Uses only real columns derived from FIRMS data.
    - No synthetic sensor values.
    - No fabricated labels.
    - Old models are not modified.
    - New models are saved only in saved_models/real_firms/.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

MODEL_DIR = (
    BASE_DIR
    / "saved_models"
    / "real_firms"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "real_firms"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGETS = {
    "24h": "fire_next_24h",
    "48h": "fire_next_48h",
    "72h": "fire_next_72h",
}


# Only real FIRMS / real time / derived geographic features.
FEATURES = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "confidence",
    "bright_t31",
    "frp",
    "daynight_encoded",
    "type",
    "hour",
    "minute",
    "day_of_week",
    "day_of_year",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]


# Time split boundaries
TRAIN_END_YEAR = 2020

VALIDATION_START_YEAR = 2021
VALIDATION_END_YEAR = 2022

TEST_START_YEAR = 2023
TEST_END_YEAR = 2025


# Model parameters
RANDOM_STATE = 42

N_ESTIMATORS = 500

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 2

N_JOBS = -1


# ======================================================================
# PRINT HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def status(label: str, value) -> None:
    print(f"{label:<32}: {value}")


# ======================================================================
# LOAD DATA
# ======================================================================

def load_dataset() -> pd.DataFrame:

    section("LOADING REAL FIRMS FORECAST DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(DATASET)

    status("Dataset", DATASET)
    status("Rows", f"{len(df):,}")
    status("Columns", len(df.columns))

    required = (
        ["detection_time", "year"]
        + FEATURES
        + list(TARGETS.values())
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    return df


# ======================================================================
# PREPARE DATA
# ======================================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("PREPARING REAL FEATURES")

    out = df.copy()

    out["detection_time"] = pd.to_datetime(
        out["detection_time"],
        errors="coerce",
    )

    if out["detection_time"].isna().any():
        raise ValueError(
            "Invalid detection_time values found."
        )

    # --------------------------------------------------------------
    # DAY/NIGHT
    # --------------------------------------------------------------

    daynight = (
        out["daynight"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {
        "D": 1,
        "N": 0,
    }

    out["daynight_encoded"] = daynight.map(mapping)

    unknown_daynight = int(
        out["daynight_encoded"].isna().sum()
    )

    status(
        "Unknown day/night values",
        unknown_daynight,
    )

    if unknown_daynight > 0:
        raise ValueError(
            "Unknown daynight values found. "
            "No values will be fabricated."
        )

    # --------------------------------------------------------------
    # TYPE
    # --------------------------------------------------------------

    out["type"] = pd.to_numeric(
        out["type"],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------------

    numeric_features = [
        feature
        for feature in FEATURES
        if feature != "daynight_encoded"
    ]

    for feature in numeric_features:

        out[feature] = pd.to_numeric(
            out[feature],
            errors="coerce",
        )

    missing_features = (
        out[FEATURES]
        .isna()
        .sum()
    )

    print()
    print("Missing feature values:")

    print(
        missing_features.to_string()
    )

    if int(missing_features.sum()) > 0:

        raise ValueError(
            "Missing feature values found. "
            "Training stopped instead of fabricating values."
        )

    # --------------------------------------------------------------
    # SORT
    # --------------------------------------------------------------

    out = out.sort_values(
        "detection_time"
    ).reset_index(drop=True)

    status(
        "First timestamp",
        out["detection_time"].min(),
    )

    status(
        "Last timestamp",
        out["detection_time"].max(),
    )

    return out


# ======================================================================
# TIME SPLIT
# ======================================================================

def split_by_time(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    section("CREATING CHRONOLOGICAL SPLITS")

    train = df.loc[
        df["year"] <= TRAIN_END_YEAR
    ].copy()

    validation = df.loc[
        (
            df["year"] >= VALIDATION_START_YEAR
        )
        & (
            df["year"] <= VALIDATION_END_YEAR
        )
    ].copy()

    test = df.loc[
        (
            df["year"] >= TEST_START_YEAR
        )
        & (
            df["year"] <= TEST_END_YEAR
        )
    ].copy()

    status(
        "Train years",
        f"2001-{TRAIN_END_YEAR}",
    )

    status(
        "Train rows",
        f"{len(train):,}",
    )

    status(
        "Validation years",
        f"{VALIDATION_START_YEAR}-{VALIDATION_END_YEAR}",
    )

    status(
        "Validation rows",
        f"{len(validation):,}",
    )

    status(
        "Test years",
        f"{TEST_START_YEAR}-{TEST_END_YEAR}",
    )

    status(
        "Test rows",
        f"{len(test):,}",
    )

    if len(train) == 0:
        raise ValueError("Training split is empty.")

    if len(validation) == 0:
        raise ValueError("Validation split is empty.")

    if len(test) == 0:
        raise ValueError("Test split is empty.")

    return train, validation, test


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    # Metrics requiring both classes.
    if len(np.unique(y_true)) >= 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["average_precision"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:

        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


# ======================================================================
# THRESHOLD SEARCH
# ======================================================================

def find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, dict]:

    best_threshold = 0.50
    best_metrics = None
    best_f1 = -1.0

    # Search validation thresholds.
    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        current_f1 = metrics["f1"]

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


# ======================================================================
# TRAIN ONE MODEL
# ======================================================================

def train_one_model(
    horizon: str,
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:

    section(
        f"TRAINING REAL FIRMS MODEL — {horizon.upper()}"
    )

    X_train = train[FEATURES]
    y_train = train[target].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation[target].astype(int)

    X_test = test[FEATURES]
    y_test = test[target].astype(int)

    status(
        "Target",
        target,
    )

    status(
        "Positive train",
        int(y_train.sum()),
    )

    status(
        "Positive validation",
        int(y_validation.sum()),
    )

    status(
        "Positive test",
        int(y_test.sum()),
    )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        class_weight="balanced_subsample",
    )

    print()
    print("Training model...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train,
            y_train,
        )

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    best_threshold, validation_metrics = (
        find_best_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    section(
        f"{horizon.upper()} VALIDATION RESULTS"
    )

    for key, value in validation_metrics.items():

        if value is None:
            print(f"{key:<22}: N/A")
        else:
            print(
                f"{key:<22}: "
                f"{value:.6f}"
            )

    # --------------------------------------------------------------
    # TEST
    # --------------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    section(
        f"{horizon.upper()} TEST RESULTS"
    )

    for key, value in test_metrics.items():

        if value is None:
            print(f"{key:<22}: N/A")
        else:
            print(
                f"{key:<22}: "
                f"{value:.6f}"
            )

    # --------------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------------

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print("Top feature importance:")

    print(
        importance
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"real_firms_forecast_{horizon}.joblib"
    )

    artifact = {
        "model": model,
        "feature_names": FEATURES,
        "target": target,
        "horizon": horizon,
        "threshold": best_threshold,
        "dataset": str(DATASET),
        "training_period": "2001-2020",
        "validation_period": "2021-2022",
        "test_period": "2023-2025",
        "synthetic_data": False,
        "fabricated_labels": False,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    joblib.dump(
        artifact,
        model_path,
    )

    status(
        "Model saved",
        model_path,
    )

    # --------------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------------

    return {
        "horizon": horizon,
        "target": target,
        "model_path": str(model_path),
        "threshold": best_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_importance": (
            importance
            .to_dict(orient="records")
        ),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }


# ======================================================================
# SAVE FINAL REPORT
# ======================================================================

def save_final_report(
    results: dict,
) -> None:

    section("SAVING TRAINING REPORT")

    report_path = (
        REPORT_DIR
        / "real_firms_training_report.json"
    )

    report = {
        "dataset": str(DATASET),
        "features": FEATURES,
        "periods": {
            "train": "2001-2020",
            "validation": "2021-2022",
            "test": "2023-2025",
        },
        "synthetic_data": False,
        "fabricated_labels": False,
        "models": results,
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    status(
        "Report saved",
        report_path,
    )


# ======================================================================
# FINAL SUMMARY
# ======================================================================

def final_summary(
    results: dict,
) -> None:

    section("FINAL TRAINING SUMMARY")

    print("SYNTHETIC DATA USED : NO")
    print("FABRICATED LABELS   : NO")
    print("OLD MODELS MODIFIED : NO")

    print()

    for horizon, result in results.items():

        metrics = result["test_metrics"]

        print(
            f"{horizon.upper()} MODEL"
        )

        print(
            f"  Threshold : "
            f"{result['threshold']:.2f}"
        )

        print(
            f"  Accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{metrics['f1']:.4f}"
        )

        if metrics["roc_auc"] is not None:

            print(
                f"  ROC-AUC   : "
                f"{metrics['roc_auc']:.4f}"
            )

        print()

    print("STATUS: TRAINING COMPLETE")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    section(
        "FIREGUARD — REAL FIRMS MODEL TRAINING"
    )

    print("PERIOD             : 2001-2025")
    print("TRAIN              : 2001-2020")
    print("VALIDATION         : 2021-2022")
    print("TEST               : 2023-2025")
    print("SYNTHETIC DATA     : NO")
    print("FABRICATED LABELS  : NO")
    print("OLD MODELS MODIFIED: NO")

    # 1. Load
    df = load_dataset()

    # 2. Prepare
    df = prepare_dataset(df)

    # 3. Split chronologically
    train, validation, test = split_by_time(df)

    # 4. Train 24h / 48h / 72h
    results = {}

    for horizon, target in TARGETS.items():

        results[horizon] = train_one_model(
            horizon,
            target,
            train,
            validation,
            test,
        )

    # 5. Save report
    save_final_report(results)

    # 6. Final summary
    final_summary(results)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print("Training interrupted by user.")

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 72)
        print("FATAL ERROR")
        print("=" * 72)
        print(str(exc))

        sys.exit(1)
