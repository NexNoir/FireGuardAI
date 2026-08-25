from pathlib import Path
import json
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

BASE = Path(r"C:\Users\vista\Desktop\fireguard_v2.0")

DATASET = (
    BASE
    / "data"
    / "retraining"
    / "real_firms_forecast_dataset_2001_2025.csv"
)

MODEL_DIR = BASE / "saved_models" / "real_firms_v1"

THRESHOLD_FILE = (
    BASE
    / "data"
    / "retraining"
    / "real_firms_threshold_config_v1.json"
)

OUT_CSV = (
    BASE
    / "data"
    / "retraining"
    / "real_firms_final_threshold_test_results.csv"
)

OUT_JSON = (
    BASE
    / "data"
    / "retraining"
    / "real_firms_final_threshold_test_report.json"
)

TARGETS = {
    "24h": "fire_next_24h",
    "48h": "fire_next_48h",
    "72h": "fire_next_72h",
}

MODELS = {
    "24h": MODEL_DIR / "fireguard_real_firms_sensor_only_24h_v1.joblib",
    "48h": MODEL_DIR / "fireguard_real_firms_sensor_only_48h_v1.joblib",
    "72h": MODEL_DIR / "fireguard_real_firms_sensor_only_72h_v1.joblib",
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


def encode_column(series):
    values = series.fillna("UNKNOWN").astype(str)
    categories = {
        value: index
        for index, value in enumerate(sorted(values.unique()))
    }
    return values.map(categories).astype(float)


def extract_model(obj, path):
    """
    The saved joblib may be:
      - a model directly
      - a dictionary containing the model

    This function searches common dictionary keys and then
    recursively searches nested dictionaries.
    """

    if hasattr(obj, "predict_proba"):
        return obj

    if isinstance(obj, dict):

        preferred_keys = [
            "model",
            "estimator",
            "classifier",
            "pipeline",
            "clf",
            "best_model",
            "trained_model",
        ]

        for key in preferred_keys:
            if key in obj:
                candidate = extract_model(
                    obj[key],
                    path
                )

                if candidate is not None:
                    return candidate

        for key, value in obj.items():
            candidate = extract_model(
                value,
                path
            )

            if candidate is not None:
                return candidate

    return None


def load_model(path):
    print(f"Loading: {path.name}")

    obj = joblib.load(path)

    print(f"Saved object type: {type(obj).__name__}")

    model = extract_model(obj, path)

    if model is None:
        raise RuntimeError(
            f"Could not extract a predict_proba model from:\n{path}\n"
            f"Saved object type: {type(obj).__name__}"
        )

    print(
        f"Extracted model type: "
        f"{type(model).__name__}"
    )

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Extracted object does not support predict_proba:\n{path}"
        )

    return model


print("=" * 72)
print("🔥 FIREGUARD — REAL FIRMS FINAL THRESHOLD TEST V2")
print("=" * 72)

print("Evaluation only.")
print("Retraining          : NO")
print("Model modification  : NO")
print("Dataset modification: NO")
print("Synthetic data      : NO")
print("Fabricated labels    : NO")
print()
print("TEST PERIOD: 2023-2025")
print()

# ---------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------

print("=" * 72)
print("LOADING REAL FIRMS DATASET")
print("=" * 72)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET}"
    )

df = pd.read_csv(DATASET)

print(f"Dataset : {DATASET}")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print()

if "year" not in df.columns:
    raise RuntimeError(
        "Required column missing: year"
    )

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

test_df = df[
    df["year"].between(2023, 2025)
].copy()

print(
    f"TEST PERIOD: "
    f"{int(test_df['year'].min())}-"
    f"{int(test_df['year'].max())}"
)

print(
    f"Test rows: {len(test_df):,}"
)

if len(test_df) == 0:
    raise RuntimeError(
        "No rows found for 2023-2025."
    )

print()

# ---------------------------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------------------------

print("=" * 72)
print("PREPARING TEST FEATURES")
print("=" * 72)

# hour / minute
if "acq_time" in test_df.columns:

    time_text = (
        test_df["acq_time"]
        .fillna(0)
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
        .str.zfill(4)
    )

    test_df["hour"] = pd.to_numeric(
        time_text.str[:2],
        errors="coerce"
    ).fillna(0)

    test_df["minute"] = pd.to_numeric(
        time_text.str[2:4],
        errors="coerce"
    ).fillna(0)

else:

    if "hour" not in test_df.columns:
        test_df["hour"] = 0

    if "minute" not in test_df.columns:
        test_df["minute"] = 0


# categorical encodings
for column in [
    "daynight",
    "satellite",
    "instrument",
    "type",
    "season",
]:

    encoded = f"{column}_encoded"

    if encoded not in test_df.columns:

        if column not in test_df.columns:
            raise RuntimeError(
                f"Cannot create {encoded}; "
                f"source column missing: {column}"
            )

        test_df[encoded] = encode_column(
            test_df[column]
        )

        print(f"Created: {encoded}")


missing = [
    feature
    for feature in FEATURES
    if feature not in test_df.columns
]

if missing:
    raise RuntimeError(
        "Required features missing:\n - "
        + "\n - ".join(missing)
    )

X = test_df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

valid_features = X.notna().all(axis=1)

test_df = test_df.loc[
    valid_features
].copy()

X = X.loc[
    valid_features
].copy()

print()
print("FINAL FEATURE LIST:")

for index, feature in enumerate(
    FEATURES,
    1
):
    print(
        f"{index:02d}. {feature}"
    )

print()
print(
    f"Feature count: {len(FEATURES)}"
)

print(
    f"Valid test rows: {len(X):,}"
)

# ---------------------------------------------------------------------
# LOAD THRESHOLDS
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("LOADING THRESHOLD CONFIGURATION")
print("=" * 72)

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(
        f"Threshold file not found:\n{THRESHOLD_FILE}"
    )

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


def read_threshold(
    horizon
):

    possible_keys = [
        horizon,
        horizon.replace("h", ""),
        f"{horizon}_threshold",
        f"threshold_{horizon}",
    ]

    for key in possible_keys:

        if key in config:
            return float(
                config[key]
            )

    if isinstance(
        config.get("thresholds"),
        dict
    ):

        thresholds = config["thresholds"]

        for key in possible_keys:

            if key in thresholds:
                return float(
                    thresholds[key]
                )

    raise RuntimeError(
        f"Threshold not found for {horizon}"
    )


thresholds = {
    horizon: read_threshold(horizon)
    for horizon in TARGETS
}

for horizon in [
    "24h",
    "48h",
    "72h"
]:

    print(
        f"{horizon.upper()} threshold : "
        f"{thresholds[horizon]:.2f}"
    )

# ---------------------------------------------------------------------
# RUN FINAL TEST
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("FINAL THRESHOLD TEST")
print("=" * 72)

results = []
summary = {}

for horizon in [
    "24h",
    "48h",
    "72h"
]:

    target = TARGETS[horizon]
    model_path = MODELS[horizon]
    threshold = thresholds[horizon]

    print()
    print("-" * 72)
    print(
        f"FINAL THRESHOLD TEST / "
        f"{horizon.upper()}"
    )
    print("-" * 72)

    if target not in test_df.columns:
        raise RuntimeError(
            f"Target missing: {target}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    valid_target = test_df[target].notna()

    X_eval = X.loc[
        valid_target
    ]

    y_true = pd.to_numeric(
        test_df.loc[
            valid_target,
            target
        ],
        errors="coerce"
    )

    target_valid = y_true.notna()

    X_eval = X_eval.loc[
        target_valid
    ]

    y_true = (
        y_true.loc[
            target_valid
        ]
        .astype(int)
        .to_numpy()
    )

    print(
        f"Test rows        : "
        f"{len(y_true):,}"
    )

    print(
        f"Positive actual  : "
        f"{int(y_true.sum()):,}"
    )

    print(
        f"Threshold        : "
        f"{threshold:.2f}"
    )

    print()
    print("Loading model...")

    model = load_model(
        model_path
    )

    # Validate model feature count
    if hasattr(
        model,
        "n_features_in_"
    ):

        expected = int(
            model.n_features_in_
        )

        if expected != len(FEATURES):

            raise RuntimeError(
                f"{horizon} model expects "
                f"{expected} features, "
                f"but current feature list "
                f"contains {len(FEATURES)}."
            )

    print("Model loaded: PASS")

    # Predict
    probabilities = (
        model
        .predict_proba(X_eval)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )
        .ravel()
    )

    print()
    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print()
    print("CONFUSION MATRIX")

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

    print()
    print(
        "Positive predictions: "
        f"{int(predictions.sum()):,}"
    )

    summary[horizon] = {
        "threshold": threshold,
        "test_rows": int(
            len(y_true)
        ),
        "positive_actual": int(
            y_true.sum()
        ),
        "positive_predictions": int(
            predictions.sum()
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
        "roc_auc": float(
            roc_auc
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    eval_indices = (
        test_df.loc[
            valid_target
        ]
        .loc[
            target_valid
        ]
        .index
    )

    for idx, probability, prediction, actual in zip(
        eval_indices,
        probabilities,
        predictions,
        y_true
    ):

        results.append({
            "row_index": int(idx),
            "year": int(
                test_df.loc[
                    idx,
                    "year"
                ]
            ),
            "horizon": horizon,
            "target": target,
            "threshold": threshold,
            "probability": float(
                probability
            ),
            "prediction": int(
                prediction
            ),
            "actual": int(
                actual
            ),
        })

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("SAVING FINAL THRESHOLD TEST")
print("=" * 72)

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUT_CSV,
    index=False
)

report = {
    "project": "FireGuard",
    "evaluation": (
        "real_firms_final_threshold_test_v2"
    ),
    "test_period": "2023-2025",
    "dataset": str(DATASET),
    "model_directory": str(MODEL_DIR),
    "threshold_config": str(
        THRESHOLD_FILE
    ),
    "features": FEATURES,
    "thresholds": thresholds,
    "summary": summary,
    "models_modified": False,
    "dataset_modified": False,
    "retraining_performed": False,
    "synthetic_data": False,
    "fabricated_labels": False,
    "status": "PASS",
}

with open(
    OUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Results : {OUT_CSV}"
)

print(
    f"Report  : {OUT_JSON}"
)

# ---------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("FINAL RESULT")
print("=" * 72)

for horizon in [
    "24h",
    "48h",
    "72h"
]:

    r = summary[horizon]

    print(
        f"{horizon.upper()} | "
        f"Threshold={r['threshold']:.2f} | "
        f"Accuracy={r['accuracy']:.4f} | "
        f"Precision={r['precision']:.4f} | "
        f"Recall={r['recall']:.4f} | "
        f"F1={r['f1']:.4f} | "
        f"ROC-AUC={r['roc_auc']:.4f}"
    )

print()
print("Models modified      : NO")
print("Dataset modified     : NO")
print("Retraining performed : NO")
print("Synthetic data       : NO")
print("Fabricated labels    : NO")
print()
print(
    "STATUS: 🟢 FINAL THRESHOLD TEST COMPLETE"
)