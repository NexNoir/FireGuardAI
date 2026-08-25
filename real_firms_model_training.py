# -*- coding: utf-8 -*-
"""
FireGuard — Real FIRMS Model Training
=====================================

Real FIRMS data only
Period: 2001-2025

TRAIN      : 2001-2020
VALIDATION : 2021-2022
TEST       : 2023-2025

Horizons:
    24h
    48h
    72h

Models:
    sensor_only
    sensor_plus_flame

IMPORTANT:
    - No synthetic data
    - No fabricated labels
    - Original dataset is never modified
    - Existing models are never overwritten
    - New models are saved under:
        saved_models/real_firms_v1/
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


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

OUTPUT_DIR = (
    BASE_DIR
    / "saved_models"
    / "real_firms_v1"
)

REPORT_FILE = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_model_training_report.json"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# MODEL CONFIG
# ======================================================================

HORIZONS = {
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
    "daynight_encoded",
]


OPTIONAL_FEATURES = [
    "satellite_encoded",
    "instrument_encoded",
    "type_encoded",
    "season_encoded",
]


# ======================================================================
# PRINT
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ======================================================================
# LOAD
# ======================================================================

def load_dataset() -> pd.DataFrame:

    section("LOADING REAL FIRMS FORECAST DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(DATASET)

    print(f"Dataset : {DATASET}")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    if "acq_date" not in df.columns:
        raise ValueError(
            "Required column missing: acq_date"
        )

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    if df["acq_date"].isna().any():
        raise ValueError(
            "Invalid acq_date values detected."
        )

    df = df.sort_values(
        "acq_date"
    ).reset_index(drop=True)

    return df


# ======================================================================
# ENCODE CATEGORICAL FEATURES
# ======================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:

    section("PREPARING REAL FIRMS FEATURES")

    out = df.copy()

    # --------------------------------------------------------------
    # Timestamp features
    # --------------------------------------------------------------

    if "acq_time" in out.columns:

        time_numeric = pd.to_numeric(
            out["acq_time"],
            errors="coerce",
        )

        out["hour"] = (
            time_numeric // 100
        ).astype("Int64")

        out["minute"] = (
            time_numeric % 100
        ).astype("Int64")

    elif "timestamp" in out.columns:

        timestamp = pd.to_datetime(
            out["timestamp"],
            errors="coerce",
        )

        out["hour"] = timestamp.dt.hour
        out["minute"] = timestamp.dt.minute

    else:

        raise ValueError(
            "Neither acq_time nor timestamp exists."
        )

    # --------------------------------------------------------------
    # daynight
    #
    # FIX FOR PREVIOUS ERROR:
    # If daynight_encoded does not exist, create it from daynight.
    # --------------------------------------------------------------

    if "daynight_encoded" not in out.columns:

        if "daynight" in out.columns:

            mapping = {
                "D": 1,
                "N": 0,
                "day": 1,
                "night": 0,
                "Day": 1,
                "Night": 0,
            }

            out["daynight_encoded"] = (
                out["daynight"]
                .astype(str)
                .str.strip()
                .map(mapping)
            )

        else:

            # If no day/night source exists,
            # keep it unknown rather than fabricate it.
            out["daynight_encoded"] = np.nan

    # --------------------------------------------------------------
    # Categorical encoding
    # --------------------------------------------------------------

    categorical_maps = {
        "satellite": "satellite_encoded",
        "instrument": "instrument_encoded",
        "type": "type_encoded",
        "season": "season_encoded",
    }

    for source, target in categorical_maps.items():

        if target in out.columns:
            continue

        if source in out.columns:

            categories = (
                out[source]
                .astype(str)
                .fillna("UNKNOWN")
                .astype("category")
            )

            out[target] = (
                categories.cat.codes
                .replace(-1, np.nan)
            )

    # --------------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------------

    numeric_candidates = (
        BASE_FEATURES
        + OPTIONAL_FEATURES
    )

    for col in numeric_candidates:

        if col in out.columns:

            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    print()
    print("Prepared features:")

    available = []

    for col in (
        BASE_FEATURES
        + OPTIONAL_FEATURES
    ):

        if col in out.columns:

            available.append(col)
            print(f" - {col}")

    print()
    print(
        f"Available feature count: {len(available)}"
    )

    return out


# ======================================================================
# FEATURE SELECTION
# ======================================================================

def select_features(
    df: pd.DataFrame,
) -> list[str]:

    features = []

    for col in BASE_FEATURES:

        if col in df.columns:
            features.append(col)

    for col in OPTIONAL_FEATURES:

        if col in df.columns:
            features.append(col)

    if not features:
        raise ValueError(
            "No usable features found."
        )

    return features


# ======================================================================
# SPLIT
# ======================================================================

def split_by_year(
    df: pd.DataFrame,
):

    train = df[
        df["acq_date"].dt.year <= 2020
    ].copy()

    validation = df[
        df["acq_date"].dt.year.between(
            2021,
            2022,
        )
    ].copy()

    test = df[
        df["acq_date"].dt.year.between(
            2023,
            2025,
        )
    ].copy()

    return train, validation, test


# ======================================================================
# METRICS
# ======================================================================

def evaluate(
    model,
    X,
    y,
) -> dict:

    prediction = model.predict(X)

    result = {
        "samples": int(len(y)),
        "positive": int(np.sum(y)),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(
            accuracy_score(y, prediction)
        ),
        "precision": float(
            precision_score(
                y,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                prediction,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y,
                prediction,
            ).tolist()
        ),
    }

    if len(np.unique(y)) == 2:

        probability = model.predict_proba(X)[:, 1]

        result["roc_auc"] = float(
            roc_auc_score(
                y,
                probability,
            )
        )

    else:

        result["roc_auc"] = None

    return result


# ======================================================================
# TRAIN ONE MODEL
# ======================================================================

def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    experiment: str,
    horizon: int,
) -> dict:

    section(
        f"TRAINING {experiment.upper()} / {horizon}H"
    )

    train = train.dropna(
        subset=[target]
    ).copy()

    validation = validation.dropna(
        subset=[target]
    ).copy()

    test = test.dropna(
        subset=[target]
    ).copy()

    X_train = train[features]
    y_train = train[target].astype(int)

    X_validation = validation[features]
    y_validation = validation[target].astype(int)

    X_test = test[features]
    y_test = test[target].astype(int)

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    print(
        f"Features   : {len(features)}"
    )

    print(
        f"Positive train: "
        f"{int(y_train.sum()):,} "
        f"({y_train.mean() * 100:.2f}%)"
    )

    if y_train.nunique() < 2:

        raise ValueError(
            f"Target {target} has only one class in training."
        )

    # --------------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print()
    print("Fitting model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
    )

    print()
    print("VALIDATION")

    print(
        f"Accuracy  : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{validation_metrics['f1']:.4f}"
    )

    if validation_metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC   : "
            f"{validation_metrics['roc_auc']:.4f}"
        )

    print()
    print("TEST")

    print(
        f"Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{test_metrics['f1']:.4f}"
    )

    if test_metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC   : "
            f"{test_metrics['roc_auc']:.4f}"
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    model_name = (
        f"fireguard_real_firms_"
        f"{experiment}_"
        f"{horizon}h_v1.joblib"
    )

    model_path = (
        OUTPUT_DIR
        / model_name
    )

    artifact = {
        "model": model,
        "feature_names": features,
        "target": target,
        "horizon_hours": horizon,
        "experiment": experiment,
        "training_period": "2001-2020",
        "validation_period": "2021-2022",
        "test_period": "2023-2025",
        "dataset": str(DATASET),
        "synthetic_data": False,
        "fabricated_labels": False,
        "random_state": 42,
    }

    joblib.dump(
        artifact,
        model_path,
    )

    print()
    print(
        f"Saved: {model_path}"
    )

    return {
        "model_path": str(model_path),
        "experiment": experiment,
        "horizon": horizon,
        "features": features,
        "target": target,
        "train": {
            "rows": len(train),
            "positive": int(y_train.sum()),
            "positive_rate": float(y_train.mean()),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    section(
        "🔥 FIREGUARD — REAL FIRMS MODEL TRAINING"
    )

    print(
        "PERIOD             : 2001-2025"
    )

    print(
        "TRAIN              : 2001-2020"
    )

    print(
        "VALIDATION         : 2021-2022"
    )

    print(
        "TEST               : 2023-2025"
    )

    print(
        "SYNTHETIC DATA     : NO"
    )

    print(
        "FABRICATED LABELS  : NO"
    )

    print(
        "OLD MODELS MODIFIED: NO"
    )

    print(
        f"NEW MODEL DIR      : {OUTPUT_DIR}"
    )

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------------
    # Prepare
    # --------------------------------------------------------------

    df = prepare_features(df)

    features = select_features(df)

    print()
    print("FINAL FEATURE LIST:")

    for i, feature in enumerate(
        features,
        start=1,
    ):
        print(
            f"{i:02d}. {feature}"
        )

    # --------------------------------------------------------------
    # Check targets
    # --------------------------------------------------------------

    for horizon, target in HORIZONS.items():

        if target not in df.columns:

            raise ValueError(
                f"Required target missing: {target}"
            )

    # --------------------------------------------------------------
    # Split
    # --------------------------------------------------------------

    train, validation, test = split_by_year(
        df
    )

    print()
    print("DATA SPLIT")

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    results = []

    for horizon, target in HORIZONS.items():

        # ----------------------------------------------------------
        # sensor_only
        # ----------------------------------------------------------

        result = train_model(
            train=train,
            validation=validation,
            test=test,
            features=features,
            target=target,
            experiment="sensor_only",
            horizon=horizon,
        )

        results.append(result)

        # ----------------------------------------------------------
        # sensor_plus_flame
        #
        # Historical FIRMS does not contain actual sensor flame.
        # Therefore this experiment is NOT trained.
        # ----------------------------------------------------------

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    report = {
        "status": "PASS",
        "dataset": str(DATASET),
        "period": "2001-2025",
        "train_period": "2001-2020",
        "validation_period": "2021-2022",
        "test_period": "2023-2025",
        "synthetic_data": False,
        "fabricated_labels": False,
        "old_models_modified": False,
        "new_model_directory": str(OUTPUT_DIR),
        "features": features,
        "models_trained": len(results),
        "models": results,
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    section("FINAL RESULT")

    print(
        f"Models trained : {len(results)}"
    )

    print(
        f"Models saved   : {OUTPUT_DIR}"
    )

    print(
        f"Report         : {REPORT_FILE}"
    )

    print()
    print(
        "Original dataset modified : NO"
    )

    print(
        "Old models modified       : NO"
    )

    print(
        "Synthetic data            : NO"
    )

    print(
        "Fabricated labels         : NO"
    )

    print()
    print(
        "STATUS: 🟢 REAL FIRMS RETRAINING COMPLETE"
    )


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print("Interrupted by user.")

    except Exception as exc:

        print()
        print("=" * 72)
        print("❌ FATAL ERROR")
        print("=" * 72)
        print(str(exc))

        raise