
"""
FireGuard — REAL FIRMS FINAL TEST EVALUATION V1

Purpose:
    Evaluate the already-trained REAL FIRMS V1 models only on
    the held-out TEST period: 2023-2025.

NO retraining.
NO model modification.
NO dataset modification.
NO synthetic data.
NO fabricated labels.

Models:
    sensor_only / 24h
    sensor_only / 48h
    sensor_only / 72h

Test period:
    2023-01-01 -> 2025-12-31
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

MODEL_DIR = (
    BASE_DIR
    / "saved_models"
    / "real_firms_v1"
)

OUTPUT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_test_evaluation_results.csv"
)

REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_test_evaluation_report.json"
)


TEST_START = 2023
TEST_END = 2025


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


def extract_model(artifact):

    if isinstance(artifact, dict):

        if artifact.get("model") is not None:
            return artifact["model"]

        if artifact.get("estimator") is not None:
            return artifact["estimator"]

    return artifact


def extract_model_features(artifact):

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
# LOAD DATA
# ======================================================================

def load_data() -> pd.DataFrame:

    section("LOADING REAL FIRMS DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(DATASET)

    print(
        f"Dataset : {DATASET}"
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    if "year" not in df.columns:
        raise ValueError(
            "Column 'year' is missing."
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df = df[
        df["year"].between(
            TEST_START,
            TEST_END,
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows found in test period."
        )

    print()
    print(
        f"TEST PERIOD: "
        f"{TEST_START}-{TEST_END}"
    )

    print(
        f"Test rows: {len(df):,}"
    )

    print(
        f"First year: "
        f"{int(df['year'].min())}"
    )

    print(
        f"Last year : "
        f"{int(df['year'].max())}"
    )

    return df.reset_index(
        drop=True
    )


# ======================================================================
# PREPARE FEATURES
# ======================================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("PREPARING TEST FEATURES")

    out = df.copy()

    # --------------------------------------------------------------
    # Hour / minute
    # --------------------------------------------------------------

    time_text = (
        out["acq_time"]
        .astype(str)
        .str.zfill(4)
    )

    out["hour"] = pd.to_numeric(
        time_text.str[:2],
        errors="coerce",
    )

    out["minute"] = pd.to_numeric(
        time_text.str[2:4],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Numeric
    # --------------------------------------------------------------

    numeric = [
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

    for col in numeric:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    # --------------------------------------------------------------
    # Categorical
    # --------------------------------------------------------------

    mappings = {
        "daynight_encoded": "daynight",
        "satellite_encoded": "satellite",
        "instrument_encoded": "instrument",
        "type_encoded": "type",
        "season_encoded": "season",
    }

    for encoded, source in mappings.items():

        out[encoded], _ = deterministic_encode(
            out[source]
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in out.columns
    ]

    if missing:
        raise ValueError(
            "Missing test features:\n"
            + "\n".join(
                f" - {x}"
                for x in missing
            )
        )

    X = out[
        FEATURES
    ].copy()

    missing_values = X.isna().sum()

    bad = missing_values[
        missing_values > 0
    ]

    if len(bad) > 0:

        print(
            "Missing feature values:"
        )

        print(
            bad.to_string()
        )

        raise ValueError(
            "Test feature matrix contains "
            "missing values."
        )

    print(
        f"Feature count: {len(FEATURES)}"
    )

    return X


# ======================================================================
# LOAD MODELS
# ======================================================================

def load_models():

    section("LOADING EXISTING MODELS")

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

        artifact = joblib.load(
            path
        )

        model = extract_model(
            artifact
        )

        model_features = (
            extract_model_features(
                artifact
            )
        )

        if model_features is None:
            raise ValueError(
                f"Feature schema unavailable "
                f"for {path.name}"
            )

        print(
            f"Model feature count: "
            f"{len(model_features)}"
        )

        if list(model_features) != FEATURES:

            print(
                "WARNING: feature order differs."
            )

        print(
            "Model loaded: PASS"
        )

        models[horizon] = {
            "model": model,
            "feature_names": model_features,
            "path": str(path),
        }

    return models


# ======================================================================
# EVALUATE
# ======================================================================

def evaluate_model(
    horizon: int,
    model_info: dict,
    X: pd.DataFrame,
    df: pd.DataFrame,
) -> dict:

    target = TARGETS[horizon]

    if target not in df.columns:
        raise ValueError(
            f"Target missing: {target}"
        )

    y = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    valid = y.notna()

    if not valid.any():
        raise ValueError(
            f"No valid target values for {target}"
        )

    X_valid = X.loc[
        valid
    ]

    y_valid = y.loc[
        valid
    ].astype(int)

    model = model_info["model"]

    model_features = model_info[
        "feature_names"
    ]

    X_model = X_valid[
        model_features
    ]

    probabilities = model.predict_proba(
        X_model
    )

    if probabilities.shape[1] != 2:
        raise ValueError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    probability = probabilities[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_valid,
        prediction,
    )

    precision = precision_score(
        y_valid,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_valid,
        prediction,
        zero_division=0,
    )

    if y_valid.nunique() == 2:

        roc_auc = roc_auc_score(
            y_valid,
            probability,
        )

    else:

        roc_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        prediction,
        labels=[0, 1],
    ).ravel()

    print()
    print(
        "-" * 72
    )

    print(
        f"FINAL TEST / {horizon}H"
    )

    print(
        "-" * 72
    )

    print(
        f"Test rows        : "
        f"{len(y_valid):,}"
    )

    print(
        f"Positive actual  : "
        f"{int(y_valid.sum()):,}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision        : "
        f"{precision:.4f}"
    )

    print(
        f"Recall           : "
        f"{recall:.4f}"
    )

    print(
        f"F1               : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC          : "
        f"{roc_auc:.4f}"
    )

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(
        f"TN: {tn:,}"
    )

    print(
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,}"
    )

    print(
        f"TP: {tp:,}"
    )

    return {
        "horizon": horizon,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "rows": int(len(y_valid)),
        "positive_actual": int(y_valid.sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            None
            if np.isnan(roc_auc)
            else float(roc_auc)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    section(
        "🔥 FIREGUARD — REAL FIRMS FINAL TEST EVALUATION V1"
    )

    print(
        "Evaluation only."
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

    print()
    print(
        f"TEST PERIOD: "
        f"{TEST_START}-{TEST_END}"
    )

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------------

    X = prepare_features(
        df
    )

    # --------------------------------------------------------------
    # MODELS
    # --------------------------------------------------------------

    models = load_models()

    # --------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------

    section(
        "FINAL TEST EVALUATION"
    )

    results = []

    for horizon in [
        24,
        48,
        72,
    ]:

        result = evaluate_model(
            horizon,
            models[horizon],
            X,
            df,
        )

        results.append(
            result
        )

    # --------------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------------

    report = {
        "status": "PASS",
        "evaluation_type": "FINAL_TEST_ONLY",
        "test_period": (
            f"{TEST_START}-{TEST_END}"
        ),
        "dataset": str(DATASET),
        "model_directory": str(MODEL_DIR),
        "models_modified": False,
        "dataset_modified": False,
        "retraining": False,
        "synthetic_data": False,
        "fabricated_labels": False,
        "results": results,
    }

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        f"Results : {OUTPUT}"
    )

    print(
        f"Report  : {REPORT}"
    )

    print()

    for result in results:

        print(
            f"{result['horizon']}H | "
            f"Accuracy={result['accuracy']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"ROC-AUC={result['roc_auc']:.4f}"
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
        "STATUS: 🟢 FINAL TEST EVALUATION COMPLETE"
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