# -*- coding: utf-8 -*-
"""
FireGuard — Real FIRMS Threshold Audit V1

Purpose:
    Evaluate different probability thresholds for the existing
    Real FIRMS models on the untouched TEST period 2023-2025.

IMPORTANT:
    - NO retraining
    - NO model modification
    - NO dataset modification
    - NO synthetic data
    - NO fabricated labels
    - Existing models are loaded read-only
"""

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

OUTPUT_RESULTS = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_threshold_audit_results.csv"
)

OUTPUT_REPORT = (
    BASE_DIR
    / "data"
    / "retraining"
    / "real_firms_threshold_audit_report.json"
)


MODEL_SPECS = {
    24: MODEL_DIR / "fireguard_real_firms_sensor_only_24h_v1.joblib",
    48: MODEL_DIR / "fireguard_real_firms_sensor_only_48h_v1.joblib",
    72: MODEL_DIR / "fireguard_real_firms_sensor_only_72h_v1.joblib",
}


TARGET_COLUMNS = {
    24: "fire_next_24h",
    48: "fire_next_48h",
    72: "fire_next_72h",
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


# Thresholds to test.
THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.05),
    2,
)


# ======================================================================
# HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def status(label: str, value: str) -> None:
    print(f"{label:<30}: {value}")


# ======================================================================
# FEATURE PREPARATION
# ======================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:

    section("PREPARING TEST FEATURES")

    out = df.copy()

    # --------------------------------------------------------------
    # Categorical encoding
    # --------------------------------------------------------------

    mappings = {
        "daynight": "daynight_encoded",
        "satellite": "satellite_encoded",
        "instrument": "instrument_encoded",
        "type": "type_encoded",
        "season": "season_encoded",
    }

    for source, target in mappings.items():

        if target in out.columns:
            continue

        if source not in out.columns:
            raise ValueError(
                f"Missing source column for encoding: {source}"
            )

        # Match the training preparation:
        # category codes are generated from the same dataset schema.
        categories = pd.Categorical(
            out[source]
        )

        out[target] = categories.codes

        print(f"Created: {target}")

    # --------------------------------------------------------------
    # Time features
    # --------------------------------------------------------------

    if "acq_time" in out.columns:

        time_string = (
            out["acq_time"]
            .astype(str)
            .str.zfill(4)
        )

        out["hour"] = pd.to_numeric(
            time_string.str[:2],
            errors="coerce",
        )

        out["minute"] = pd.to_numeric(
            time_string.str[2:4],
            errors="coerce",
        )

    elif "timestamp" in out.columns:

        timestamp = pd.to_datetime(
            out["timestamp"],
            errors="coerce",
        )

        out["hour"] = timestamp.dt.hour
        out["minute"] = timestamp.dt.minute

    else:

        raise ValueError(
            "Neither acq_time nor timestamp is available."
        )

    missing = [
        col
        for col in FEATURE_COLUMNS
        if col not in out.columns
    ]

    if missing:
        raise ValueError(
            "Required model features are missing:\n"
            + "\n".join(
                f" - {x}"
                for x in missing
            )
        )

    X = out[FEATURE_COLUMNS].copy()

    # Numeric conversion
    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    if X.isna().any().any():

        missing_counts = (
            X.isna()
            .sum()
        )

        bad = missing_counts[
            missing_counts > 0
        ]

        print()
        print("Missing feature values:")

        print(
            bad.to_string()
        )

        raise ValueError(
            "NaN values detected in model features."
        )

    print()
    print("FINAL FEATURE LIST:")

    for i, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        print(
            f"{i:02d}. {feature}"
        )

    print()
    print(
        f"Feature count: {len(FEATURE_COLUMNS)}"
    )

    return X


# ======================================================================
# MODEL FEATURE CHECK
# ======================================================================

def get_model_features(model):

    if isinstance(model, dict):

        if model.get("feature_names") is not None:
            return list(
                model["feature_names"]
            )

        if model.get("features") is not None:
            return list(
                model["features"]
            )

        inner = model.get("model")

        if inner is not None and hasattr(
            inner,
            "feature_names_in_",
        ):
            return list(
                inner.feature_names_in_
            )

    if hasattr(
        model,
        "feature_names_in_",
    ):
        return list(
            model.feature_names_in_
        )

    return None


def get_prediction_model(artifact):

    if isinstance(artifact, dict):

        if artifact.get("model") is not None:
            return artifact["model"]

        if artifact.get("estimator") is not None:
            return artifact["estimator"]

    return artifact


# ======================================================================
# LOAD DATA
# ======================================================================

def load_test_data():

    section("LOADING REAL FIRMS TEST DATA")

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
            "Required column 'year' is missing."
        )

    # --------------------------------------------------------------
    # TEST PERIOD
    # --------------------------------------------------------------

    test_df = df[
        df["year"].between(
            2023,
            2025,
        )
    ].copy()

    test_df = test_df.reset_index(
        drop=True
    )

    if len(test_df) == 0:
        raise ValueError(
            "No test rows found for 2023-2025."
        )

    print()
    print(
        "TEST PERIOD: 2023-2025"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        f"First year: {test_df['year'].min()}"
    )

    print(
        f"Last year : {test_df['year'].max()}"
    )

    return test_df


# ======================================================================
# THRESHOLD EVALUATION
# ======================================================================

def evaluate_thresholds(
    y_true,
    probabilities,
    horizon,
):

    rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions,
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        rows.append({
            "horizon_hours": horizon,
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return rows


# ======================================================================
# MAIN
# ======================================================================

def main():

    section(
        "🔥 FIREGUARD — REAL FIRMS THRESHOLD AUDIT V1"
    )

    print(
        "Threshold optimization audit only."
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
        "Fabricated labels    : NO"
    )

    print(
        f"Model directory     : {MODEL_DIR}"
    )

    print()
    print(
        "Threshold range     : 0.10 -> 0.90"
    )

    # --------------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------------

    df = load_test_data()

    # --------------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------------

    X = prepare_features(
        df
    )

    all_results = []
    report = {
        "period": "2023-2025",
        "retraining": False,
        "model_modification": False,
        "dataset_modification": False,
        "synthetic_data": False,
        "fabricated_labels": False,
        "thresholds": [
            float(x)
            for x in THRESHOLDS
        ],
        "models": {},
    }

    # --------------------------------------------------------------
    # EACH HORIZON
    # --------------------------------------------------------------

    for horizon, model_path in MODEL_SPECS.items():

        section(
            f"THRESHOLD AUDIT / {horizon}H"
        )

        target_column = TARGET_COLUMNS[
            horizon
        ]

        if target_column not in df.columns:
            print(
                f"Target missing: {target_column}"
            )

            report["models"][
                str(horizon)
            ] = {
                "status": "TARGET_MISSING"
            }

            continue

        if not model_path.exists():
            print(
                f"Model not found: {model_path}"
            )

            report["models"][
                str(horizon)
            ] = {
                "status": "MODEL_NOT_FOUND"
            }

            continue

        y_true = pd.to_numeric(
            df[target_column],
            errors="coerce",
        )

        valid = (
            y_true.notna()
            & X.notna().all(axis=1)
        )

        X_test = X.loc[
            valid
        ]

        y_test = y_true.loc[
            valid
        ].astype(int)

        print(
            f"Target              : {target_column}"
        )

        print(
            f"Valid test rows     : {len(y_test):,}"
        )

        print(
            f"Positive actual     : {int(y_test.sum()):,}"
        )

        # ----------------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------------

        print()
        print(
            "Loading model..."
        )

        artifact = joblib.load(
            model_path
        )

        model = get_prediction_model(
            artifact
        )

        model_features = get_model_features(
            artifact
        )

        if model_features is not None:

            if list(model_features) != FEATURE_COLUMNS:

                print()
                print(
                    "MODEL FEATURE ORDER:"
                )

                for i, feature in enumerate(
                    model_features,
                    start=1,
                ):
                    print(
                        f"{i:02d}. {feature}"
                    )

                raise ValueError(
                    f"Feature schema mismatch for {horizon}h model."
                )

        print(
            "Model loaded: PASS"
        )

        # ----------------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------------

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise ValueError(
                f"Model for {horizon}h does not "
                "support predict_proba()."
            )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        print(
            f"ROC-AUC             : {roc_auc:.4f}"
        )

        # ----------------------------------------------------------
        # EVALUATE THRESHOLDS
        # ----------------------------------------------------------

        rows = evaluate_thresholds(
            y_test,
            probabilities,
            horizon,
        )

        all_results.extend(
            rows
        )

        results_df = pd.DataFrame(
            rows
        )

        # Best F1
        best_f1_row = (
            results_df
            .sort_values(
                [
                    "f1",
                    "recall",
                    "precision",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        # Best Recall
        best_recall_row = (
            results_df
            .sort_values(
                [
                    "recall",
                    "precision",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        # Best Precision
        best_precision_row = (
            results_df
            .sort_values(
                [
                    "precision",
                    "recall",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        report["models"][
            str(horizon)
        ] = {
            "status": "PASS",
            "model": str(model_path),
            "target": target_column,
            "test_rows": int(len(y_test)),
            "positive_actual": int(y_test.sum()),
            "roc_auc": float(roc_auc),

            "best_f1": {
                "threshold": float(
                    best_f1_row["threshold"]
                ),
                "accuracy": float(
                    best_f1_row["accuracy"]
                ),
                "precision": float(
                    best_f1_row["precision"]
                ),
                "recall": float(
                    best_f1_row["recall"]
                ),
                "f1": float(
                    best_f1_row["f1"]
                ),
            },

            "best_recall": {
                "threshold": float(
                    best_recall_row["threshold"]
                ),
                "accuracy": float(
                    best_recall_row["accuracy"]
                ),
                "precision": float(
                    best_recall_row["precision"]
                ),
                "recall": float(
                    best_recall_row["recall"]
                ),
                "f1": float(
                    best_recall_row["f1"]
                ),
            },

            "best_precision": {
                "threshold": float(
                    best_precision_row["threshold"]
                ),
                "accuracy": float(
                    best_precision_row["accuracy"]
                ),
                "precision": float(
                    best_precision_row["precision"]
                ),
                "recall": float(
                    best_precision_row["recall"]
                ),
                "f1": float(
                    best_precision_row["f1"]
                ),
            },
        }

        # ----------------------------------------------------------
        # PRINT TABLE
        # ----------------------------------------------------------

        print()
        print(
            f"{'THRESHOLD':<12}"
            f"{'ACC':>10}"
            f"{'PREC':>10}"
            f"{'RECALL':>10}"
            f"{'F1':>10}"
        )

        print(
            "-" * 52
        )

        for _, row in results_df.iterrows():

            print(
                f"{row['threshold']:<12.2f}"
                f"{row['accuracy']:>10.4f}"
                f"{row['precision']:>10.4f}"
                f"{row['recall']:>10.4f}"
                f"{row['f1']:>10.4f}"
            )

        print()
        print(
            "BEST F1"
        )

        print(
            f"Threshold : "
            f"{best_f1_row['threshold']:.2f}"
        )

        print(
            f"Accuracy  : "
            f"{best_f1_row['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{best_f1_row['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{best_f1_row['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{best_f1_row['f1']:.4f}"
        )

        print()
        print(
            "BEST RECALL"
        )

        print(
            f"Threshold : "
            f"{best_recall_row['threshold']:.2f}"
        )

        print(
            f"Precision : "
            f"{best_recall_row['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{best_recall_row['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{best_recall_row['f1']:.4f}"
        )

        print()
        print(
            "BEST PRECISION"
        )

        print(
            f"Threshold : "
            f"{best_precision_row['threshold']:.2f}"
        )

        print(
            f"Precision : "
            f"{best_precision_row['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{best_precision_row['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{best_precision_row['f1']:.4f}"
        )

    # --------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------

    section(
        "SAVING THRESHOLD AUDIT"
    )

    OUTPUT_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Results : {OUTPUT_RESULTS}"
    )

    print(
        f"Report  : {OUTPUT_REPORT}"
    )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        "Threshold audit completed."
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
        "STATUS: 🟢 THRESHOLD AUDIT COMPLETE"
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
        print(
            "=" * 72
        )

        print(
            "❌ FATAL ERROR"
        )

        print(
            "=" * 72
        )

        print(
            str(exc)
        )

        sys.exit(1)