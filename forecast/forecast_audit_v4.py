import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# FireGuard - Forecast Quality Audit v4
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = (
    BASE_DIR
    / "data"
    / "fireguard_forecast_60days.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "saved_models"
)

METRICS_FILE = (
    MODEL_DIR
    / "forecast_metrics_v3.json"
)

HORIZONS = {
    "24h": "fire_next_24h",
    "48h": "fire_next_48h",
    "72h": "fire_next_72h",
}

EXPERIMENTS = {
    "sensor_only": False,
    "sensor_plus_flame": True,
}

V3_MODEL_NAMES = [
    "fireguard_forecast_sensor_only_24h_v3.joblib",
    "fireguard_forecast_sensor_only_48h_v3.joblib",
    "fireguard_forecast_sensor_only_72h_v3.joblib",
    "fireguard_forecast_sensor_plus_flame_24h_v3.joblib",
    "fireguard_forecast_sensor_plus_flame_48h_v3.joblib",
    "fireguard_forecast_sensor_plus_flame_72h_v3.joblib",
]


# ============================================================
# Feature schema (exact match with training v3)
# ============================================================

BASE_FEATURES = [
    "temperature",
    "humidity",
    "smoke",
    "hour",
    "minute",

    "smoke_change_1m",
    "smoke_change_5m",
    "smoke_change_15m",
    "smoke_change_30m",
    "smoke_change_60m",

    "temperature_change_5m",
    "temperature_change_15m",
    "temperature_change_30m",
    "temperature_change_60m",

    "humidity_change_5m",
    "humidity_change_15m",
    "humidity_change_30m",

    "smoke_mean_5m",
    "smoke_mean_15m",
    "smoke_mean_30m",
    "smoke_mean_60m",

    "smoke_std_15m",
    "smoke_std_30m",
    "smoke_max_30m",

    "temperature_mean_15m",
    "temperature_mean_30m",

    "humidity_mean_15m",
    "humidity_mean_30m",
]

FLAME_FEATURES = [
    "flame"
]


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def event_id_from_fire_series(fire_series):
    """
    هر بازه متوالی fire_now=1 یک Fire Event محسوب می‌شود.
    """

    fire = fire_series.astype(int).to_numpy()

    starts = (
        (fire == 1)
        &
        (
            np.r_[
                True,
                fire[:-1] == 0
            ]
        )
    )

    event_id = np.cumsum(starts)

    event_id[fire == 0] = 0

    return event_id


def load_v3_model(model_path):
    """
    Load a v3 model payload.
    Supports both new dict format and legacy raw estimator.
    Returns (estimator, features_list, threshold_or_None)
    """
    payload = joblib.load(model_path)

    if isinstance(payload, dict) and "model" in payload:
        estimator = payload["model"]
        features = list(payload.get("features", []))
        threshold = payload.get("threshold", None)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except Exception:
                threshold = None
        return estimator, features, threshold

    # Legacy raw model
    return payload, None, None


# ============================================================
# Feature engineering audit
# ============================================================

def build_features(df):
    """
    همان Featureهای قابل استفاده در زمان prediction را
    مستقل از مدل محاسبه می‌کند.

    هیچ Future Target در این قسمت استفاده نمی‌شود.
    """

    out = df.copy()

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    out["timestamp"] = pd.to_datetime(
        out["timestamp"]
    )

    out["hour"] = (
        out["timestamp"].dt.hour
    )

    out["minute"] = (
        out["timestamp"].dt.minute
    )

    # --------------------------------------------------------
    # Changes
    # --------------------------------------------------------

    out["smoke_change_1m"] = (
        out["smoke"]
        .diff(1)
    )

    out["smoke_change_5m"] = (
        out["smoke"]
        .diff(5)
    )

    out["smoke_change_15m"] = (
        out["smoke"]
        .diff(15)
    )

    out["smoke_change_30m"] = (
        out["smoke"]
        .diff(30)
    )

    out["smoke_change_60m"] = (
        out["smoke"]
        .diff(60)
    )

    out["temperature_change_5m"] = (
        out["temperature"]
        .diff(5)
    )

    out["temperature_change_15m"] = (
        out["temperature"]
        .diff(15)
    )

    out["temperature_change_30m"] = (
        out["temperature"]
        .diff(30)
    )

    out["temperature_change_60m"] = (
        out["temperature"]
        .diff(60)
    )

    out["humidity_change_5m"] = (
        out["humidity"]
        .diff(5)
    )

    out["humidity_change_15m"] = (
        out["humidity"]
        .diff(15)
    )

    out["humidity_change_30m"] = (
        out["humidity"]
        .diff(30)
    )

    # --------------------------------------------------------
    # Rolling
    # --------------------------------------------------------

    out["smoke_mean_5m"] = (
        out["smoke"]
        .rolling(5)
        .mean()
    )

    out["smoke_mean_15m"] = (
        out["smoke"]
        .rolling(15)
        .mean()
    )

    out["smoke_mean_30m"] = (
        out["smoke"]
        .rolling(30)
        .mean()
    )

    out["smoke_mean_60m"] = (
        out["smoke"]
        .rolling(60)
        .mean()
    )

    out["smoke_std_15m"] = (
        out["smoke"]
        .rolling(15)
        .std()
    )

    out["smoke_std_30m"] = (
        out["smoke"]
        .rolling(30)
        .std()
    )

    out["smoke_max_30m"] = (
        out["smoke"]
        .rolling(30)
        .max()
    )

    out["temperature_mean_15m"] = (
        out["temperature"]
        .rolling(15)
        .mean()
    )

    out["temperature_mean_30m"] = (
        out["temperature"]
        .rolling(30)
        .mean()
    )

    out["humidity_mean_15m"] = (
        out["humidity"]
        .rolling(15)
        .mean()
    )

    out["humidity_mean_30m"] = (
        out["humidity"]
        .rolling(30)
        .mean()
    )

    return out


# ============================================================
# Load dataset
# ============================================================

print_section(
    "🔥 FireGuard Forecast Quality Audit v4"
)

print(
    f"Dataset : {DATASET}"
)

print(
    f"Models  : {MODEL_DIR}"
)

if not DATASET.exists():
    print()
    print("❌ Dataset پیدا نشد.")
    print("مسیر را بررسی کن.")
    raise SystemExit(1)


df = pd.read_csv(
    DATASET
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


print(
    f"Records : {len(df):,}"
)

print(
    f"Start   : {df['timestamp'].min()}"
)

print(
    f"End     : {df['timestamp'].max()}"
)


# ============================================================
# 1. Dataset Audit
# ============================================================

print_section(
    "1. DATASET AUDIT"
)

print("Fire events:")

if "fire_now" not in df.columns:
    print(
        "❌ fire_now در Dataset وجود ندارد."
    )
    raise SystemExit(1)


df["event_id"] = event_id_from_fire_series(
    df["fire_now"]
)

event_ids = sorted(
    x
    for x in df["event_id"].unique()
    if x != 0
)

print(
    f"Fire Event Count : {len(event_ids)}"
)

event_table = []

for eid in event_ids:

    event_rows = df[
        df["event_id"] == eid
    ]

    event_table.append(
        {
            "event_id": int(eid),
            "start": str(
                event_rows["timestamp"].min()
            ),
            "end": str(
                event_rows["timestamp"].max()
            ),
            "records": int(
                len(event_rows)
            ),
        }
    )


events_df = pd.DataFrame(
    event_table
)

if len(events_df):

    print(
        events_df.to_string(
            index=False
        )
    )


print()
print("Target distributions:")

target_summary = {}

for horizon, column in HORIZONS.items():

    if column not in df.columns:

        print(
            f"{horizon}: ❌ target missing"
        )

        continue

    counts = (
        df[column]
        .value_counts(
            dropna=True
        )
        .sort_index()
    )

    positives = int(
        counts.get(1.0, 0)
    )

    negatives = int(
        counts.get(0.0, 0)
    )

    total = positives + negatives

    positive_pct = (
        100 * positives / total
        if total
        else 0
    )

    print()
    print(horizon)
    print(
        f"Positive : {positives:,} "
        f"({positive_pct:.2f}%)"
    )
    print(
        f"Negative : {negatives:,}"
    )

    target_summary[horizon] = {
        "positive": positives,
        "negative": negatives,
        "positive_percent": positive_pct,
    }


# ============================================================
# 2. Positive samples per event
# ============================================================

print_section(
    "2. POSITIVE SAMPLES / EVENT AUDIT"
)

for horizon, column in HORIZONS.items():

    if column not in df.columns:
        continue

    positive = df[
        df[column] == 1
    ]

    if positive.empty:
        continue

    by_event = (
        positive
        .groupby("event_id")
        .size()
    )

    by_event = (
        by_event[
            by_event.index != 0
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        f"{horizon}:"
    )

    print(
        by_event.to_string()
    )

    print(
        f"Positive rows : {len(positive):,}"
    )

    print(
        f"Positive events represented : "
        f"{len(by_event)}"
    )


# ============================================================
# 3. Temporal Split Audit
# ============================================================

print_section(
    "3. TEMPORAL SPLIT AUDIT"
)

n = len(df)

train_end = int(
    n * 0.70
)

validation_end = int(
    n * 0.85
)

train = df.iloc[
    :train_end
]

validation = df.iloc[
    train_end:validation_end
]

test = df.iloc[
    validation_end:
]


def describe_split(name, part):

    print()
    print(name)

    print(
        f"Records : {len(part):,}"
    )

    print(
        f"Start   : {part['timestamp'].min()}"
    )

    print(
        f"End     : {part['timestamp'].max()}"
    )


describe_split(
    "TRAIN",
    train
)

describe_split(
    "VALIDATION",
    validation
)

describe_split(
    "TEST",
    test
)

chronological = (
    train["timestamp"].max()
    <
    validation["timestamp"].min()
    <
    validation["timestamp"].max()
    <
    test["timestamp"].min()
)

print()

if chronological:
    print(
        "✅ Chronological ordering: PASS"
    )
else:
    print(
        "❌ Chronological ordering: FAIL"
    )


# ============================================================
# 4. Event Leakage Audit
# ============================================================

print_section(
    "4. EVENT LEAKAGE AUDIT"
)

def event_set(part):

    return set(
        int(x)
        for x in part["event_id"].unique()
        if x != 0
    )


train_events = event_set(train)
validation_events = event_set(validation)
test_events = event_set(test)

print(
    f"Train events      : {sorted(train_events)}"
)

print(
    f"Validation events : {sorted(validation_events)}"
)

print(
    f"Test events       : {sorted(test_events)}"
)

train_test_overlap = (
    train_events
    &
    test_events
)

validation_test_overlap = (
    validation_events
    &
    test_events
)

train_validation_overlap = (
    train_events
    &
    validation_events
)

print()

print(
    "Train/Test overlap:",
    sorted(train_test_overlap)
)

print(
    "Validation/Test overlap:",
    sorted(validation_test_overlap)
)

print(
    "Train/Validation overlap:",
    sorted(train_validation_overlap)
)

if (
    train_test_overlap
    or validation_test_overlap
    or train_validation_overlap
):

    print()
    print(
        "🔴 EVENT LEAKAGE DETECTED"
    )

else:

    print()
    print(
        "🟢 No event overlap detected."
    )


# ============================================================
# 5. Boundary Event Audit
# ============================================================

print_section(
    "5. SPLIT BOUNDARY EVENT AUDIT"
)

for name, boundary in [
    (
        "TRAIN → VALIDATION",
        train_end
    ),
    (
        "VALIDATION → TEST",
        validation_end
    ),
]:

    if boundary <= 0 or boundary >= len(df):
        continue

    before = df.iloc[
        boundary - 1
    ]

    after = df.iloc[
        boundary
    ]

    print()
    print(name)

    print(
        f"Before : "
        f"{before['timestamp']} | "
        f"event={before['event_id']} | "
        f"fire={before['fire_now']}"
    )

    print(
        f"After  : "
        f"{after['timestamp']} | "
        f"event={after['event_id']} | "
        f"fire={after['fire_now']}"
    )

    if (
        before["event_id"]
        != 0
        and
        before["event_id"]
        ==
        after["event_id"]
    ):

        print(
            "🔴 SAME FIRE EVENT CROSSES SPLIT"
        )

    else:

        print(
            "🟢 No active fire event crosses boundary"
        )


# ============================================================
# 6. Target Formula Audit
# ============================================================

print_section(
    "6. TARGET AUDIT"
)

print(
    "Expected target definition:"
)

print(
    "target[t] = 1 if any fire_now[t+1 : "
    "t+horizon] == 1"
)

print()

target_ok = True

fire = (
    df["fire_now"]
    .astype(int)
    .to_numpy()
)

for horizon, column in HORIZONS.items():

    minutes = {
        "24h": 1440,
        "48h": 2880,
        "72h": 4320,
    }[horizon]

    if column not in df.columns:
        print(
            f"{horizon}: ❌ missing"
        )
        target_ok = False
        continue

    mismatches = 0

    max_i = len(df) - minutes

    for i in range(
        max_i
    ):

        expected = int(
            np.any(
                fire[
                    i + 1:
                    i + minutes + 1
                ]
                == 1
            )
        )

        actual = df.iloc[
            i
        ][column]

        if int(actual) != expected:
            mismatches += 1

            if mismatches >= 10:
                break

    print(
        f"{horizon}: "
        f"mismatches={mismatches}"
    )

    if mismatches:
        target_ok = False


if target_ok:
    print(
        "🟢 Target construction: PASS"
    )
else:
    print(
        "🔴 Target construction: FAIL"
    )


# ============================================================
# 7. Feature Audit
# ============================================================

print_section(
    "7. FEATURE AUDIT"
)

try:

    feature_df = build_features(
        df
    )

except Exception as e:

    print(
        "❌ Feature engineering failed:"
    )

    print(e)

    feature_df = None


feature_sets = {
    "sensor_only": BASE_FEATURES,
    "sensor_plus_flame":
        BASE_FEATURES + FLAME_FEATURES,
}


feature_audit_ok = True

if feature_df is not None:

    for name, features in feature_sets.items():

        print()
        print(
            f"{name}:"
        )

        missing = [
            f
            for f in features
            if f not in feature_df.columns
        ]

        if missing:

            print(
                "❌ Missing:"
            )

            print(
                missing
            )
            feature_audit_ok = False

        else:

            print(
                "✅ All features available."
            )

        future_like = [
            f
            for f in features
            if any(
                token in f.lower()
                for token in [
                    "future",
                    "next",
                    "target"
                ]
            )
        ]

        if future_like:

            print(
                "⚠️ Suspicious future-named features:"
            )

            print(
                future_like
            )
            feature_audit_ok = False

        else:

            print(
                "✅ No future/target feature names."
            )

else:
    feature_audit_ok = False


# ============================================================
# 8. NaN Audit
# ============================================================

print_section(
    "8. NaN AUDIT"
)

nan_audit_ok = True

if feature_df is not None:

    for name, features in feature_sets.items():

        missing = (
            feature_df[features]
            .isna()
            .sum()
        )

        nonzero = (
            missing[
                missing > 0
            ]
        )

        print()
        print(name)

        if nonzero.empty:

            print(
                "✅ No NaN."
            )

        else:

            print(
                nonzero.to_string()
            )

            print(
                "ℹ️ These are expected at the "
                "beginning of rolling windows."
            )

else:
    nan_audit_ok = False


# ============================================================
# 9. Model Metrics Audit
# ============================================================

print_section(
    "9. SAVED MODEL METRICS AUDIT"
)

metrics_json = {}
metrics_ok = False

if not METRICS_FILE.exists():

    print(
        "⚠️ forecast_metrics_v3.json پیدا نشد."
    )

else:

    try:

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            metrics_json = json.load(f)

        print(
            "✅ Metrics JSON loaded."
        )
        metrics_ok = True

    except Exception as e:

        print(
            "❌ Could not read metrics JSON:"
        )

        print(e)


# ============================================================
# Find model files
# ============================================================

print_section(
    "10. SAVED MODEL CHECK"
)

all_joblib = sorted(
    MODEL_DIR.glob(
        "*.joblib"
    )
)

v3_files = []
legacy_files = []

for path in all_joblib:
    if path.name in V3_MODEL_NAMES:
        v3_files.append(path)
    else:
        legacy_files.append(path)

print("V3 MODELS")
if v3_files:
    for path in v3_files:
        print(f"✅ {path.name}")
else:
    print("❌ No v3 models found.")

print()
print("LEGACY MODELS")
if legacy_files:
    for path in legacy_files:
        print(f"ℹ️ {path.name}  (Ignored for independent evaluation)")
else:
    print("ℹ️ No legacy models present.")


# ============================================================
# Recalculate metrics from saved v3 models only
# ============================================================

print_section(
    "11. INDEPENDENT TEST METRICS (v3 only)"
)

audit_results = {}
schema_results = {}
predict_errors = []
all_v3_exist = True
all_schemas_match = True
all_evals_ok = True

for experiment, include_flame in EXPERIMENTS.items():

    expected_features = (
        BASE_FEATURES
        + (
            FLAME_FEATURES
            if include_flame
            else []
        )
    )

    for horizon, target in HORIZONS.items():

        model_name = (
            f"fireguard_forecast_"
            f"{experiment}_"
            f"{horizon}_v3.joblib"
        )

        model_path = (
            MODEL_DIR
            / model_name
        )

        print()
        print(
            "-" * 70
        )

        print(
            f"{experiment.upper()} / {horizon}"
        )

        print(
            f"Model: {model_name}"
        )

        if not model_path.exists():

            print(
                "❌ Model file not found."
            )
            all_v3_exist = False
            all_evals_ok = False
            continue

        try:

            estimator, saved_features, saved_threshold = load_v3_model(
                model_path
            )

        except Exception as e:

            print(
                "❌ Model load failed:"
            )
            print(e)
            predict_errors.append(f"{experiment}/{horizon}: load failed")
            all_evals_ok = False
            continue

        # ----------------------------------------------------
        # Schema check
        # ----------------------------------------------------

        model_feature_names = getattr(
            estimator,
            "feature_names_in_",
            None
        )

        if model_feature_names is not None:
            actual = list(model_feature_names)
        elif saved_features:
            actual = list(saved_features)
        else:
            actual = None

        schema_pass = False

        if actual is None:
            print("⚠️ Model does not expose feature_names_in_ and no saved features.")
            schema_pass = False
            all_schemas_match = False
        else:
            missing = [x for x in expected_features if x not in actual]
            extra = [x for x in actual if x not in expected_features]
            order_match = (actual == expected_features)

            print(f"Expected count : {len(expected_features)}")
            print(f"Model count    : {len(actual)}")
            print(f"Order match    : {order_match}")

            if missing:
                print("❌ Missing:")
                print(missing)
            if extra:
                print("⚠️ Extra:")
                print(extra)

            if (not missing) and (not extra) and order_match:
                print("🟢 Schema match: PASS")
                schema_pass = True
            else:
                print("🔴 Schema mismatch: FAIL")
                schema_pass = False
                all_schemas_match = False

        schema_results[f"{experiment}_{horizon}"] = schema_pass

        if not schema_pass:
            all_evals_ok = False
            continue

        # ----------------------------------------------------
        # Build test features with exact order
        # ----------------------------------------------------

        test_part = test.copy()

        X_full = build_features(test_part)

        # Ensure all expected columns exist
        missing_cols = [c for c in expected_features if c not in X_full.columns]
        if missing_cols:
            print(f"❌ Missing columns in feature matrix: {missing_cols}")
            all_evals_ok = False
            continue

        X = X_full[expected_features].copy()
        y = test_part[target].astype(int)

        valid = (
            X.notna().all(axis=1)
            &
            y.notna()
        )

        X = X.loc[valid]
        y = y.loc[valid]

        # Final assert on columns
        if list(X.columns) != expected_features:
            print("❌ Feature order after selection does not match expected.")
            all_evals_ok = False
            continue

        # ----------------------------------------------------
        # Threshold (from metrics JSON or model metadata)
        # ----------------------------------------------------

        threshold = 0.50

        # Prefer metrics JSON structure
        try:
            if (
                experiment in metrics_json
                and horizon in metrics_json[experiment]
            ):
                th = metrics_json[experiment][horizon].get("threshold")
                if th is not None:
                    threshold = float(th)
        except Exception:
            pass

        # Fallback to saved threshold inside model payload
        if saved_threshold is not None:
            threshold = float(saved_threshold)

        # ----------------------------------------------------
        # predict_proba – never silently ignore errors
        # ----------------------------------------------------

        try:
            # Always pass DataFrame with correct column names/order
            probabilities = estimator.predict_proba(X)[:, 1]
        except Exception as e:
            print("❌ predict_proba failed:")
            print(e)
            predict_errors.append(f"{experiment}/{horizon}: {e}")
            all_evals_ok = False
            continue

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y,
            predictions
        )

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        try:
            pr_auc = average_precision_score(y, probabilities)
        except Exception:
            pr_auc = None

        try:
            roc_auc = roc_auc_score(y, probabilities)
        except Exception:
            roc_auc = None

        print()
        print("Independent Test Metrics:")
        print(f"Threshold : {threshold:.4f}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")

        if pr_auc is not None:
            print(f"PR-AUC    : {pr_auc:.4f}")
        else:
            print("PR-AUC    : N/A")

        if roc_auc is not None:
            print(f"ROC-AUC   : {roc_auc:.4f}")
        else:
            print("ROC-AUC   : N/A")

        print("Confusion Matrix:")
        print(cm)

        print()
        print(
            f"TN={tn} "
            f"FP={fp} "
            f"FN={fn} "
            f"TP={tp}"
        )

        audit_results[
            f"{experiment}_{horizon}"
        ] = {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "schema_pass": schema_pass,
        }


# ============================================================
# 12. Probability Calibration Audit
# ============================================================

print_section(
    "12. PROBABILITY CALIBRATION AUDIT"
)

print(
    "Current Random Forest probabilities are"
)

print(
    "UN-CALIBRATED PROBABILITY"
)

print()

print(
    "Therefore values such as 77.42%, 60.31%,"
)

print(
    "or 97.70% must NOT yet be described as"
)

print(
    "'real probability of fire'."
)

print()

print(
    "Calibration method detected in this audit:"
)

print(
    "❌ No automatic calibration."
)


# ============================================================
# 13. Feature schema audit against saved model (summary)
# ============================================================

print_section(
    "13. TRAINING / INFERENCE SCHEMA AUDIT (v3)"
)

for experiment, include_flame in EXPERIMENTS.items():

    expected = (
        BASE_FEATURES
        + (
            FLAME_FEATURES
            if include_flame
            else []
        )
    )

    for horizon in HORIZONS:

        model_name = (
            f"fireguard_forecast_"
            f"{experiment}_"
            f"{horizon}_v3.joblib"
        )

        model_path = (
            MODEL_DIR
            / model_name
        )

        key = f"{experiment}_{horizon}"

        print()
        print(f"{experiment} / {horizon}")

        if not model_path.exists():
            print("❌ Model file missing.")
            continue

        try:
            estimator, saved_features, _ = load_v3_model(model_path)
        except Exception as e:
            print(f"❌ Load failed: {e}")
            continue

        model_feature_names = getattr(
            estimator,
            "feature_names_in_",
            None
        )

        if model_feature_names is not None:
            actual = list(model_feature_names)
        elif saved_features:
            actual = list(saved_features)
        else:
            print("⚠️ No feature names available.")
            continue

        missing = [x for x in expected if x not in actual]
        extra = [x for x in actual if x not in expected]
        order_match = (actual == expected)

        print(f"Expected count : {len(expected)}")
        print(f"Model count    : {len(actual)}")
        print(f"Order match    : {order_match}")

        if missing:
            print("❌ Missing:")
            print(missing)
        if extra:
            print("⚠️ Extra:")
            print(extra)

        if (not missing) and (not extra) and order_match:
            print("🟢 Schema match: PASS")
        else:
            print("🔴 Schema mismatch.")


# ============================================================
# 14. Final Gate
# ============================================================

print_section(
    "14. FINAL QUALITY GATE"
)

fail_reasons = []
warnings = []


# Event leakage
if (
    train_test_overlap
    or validation_test_overlap
    or train_validation_overlap
):

    fail_reasons.append(
        "EVENT LEAKAGE / EVENT OVERLAP"
    )


# Chronology
if not chronological:

    fail_reasons.append(
        "TEMPORAL ORDERING FAILURE"
    )


# Target
if not target_ok:

    fail_reasons.append(
        "TARGET CONSTRUCTION FAILURE"
    )


# Feature audit
if not feature_audit_ok:
    fail_reasons.append(
        "FEATURE AUDIT FAILURE"
    )


# NaN (informational only – expected at start of rolling windows)
# Do not treat as failure


# Metrics JSON
if not metrics_ok:
    fail_reasons.append(
        "METRICS JSON MISSING OR UNREADABLE"
    )


# All 6 v3 models must exist
if not all_v3_exist:
    fail_reasons.append(
        "ONE OR MORE V3 MODELS MISSING"
    )


# All schemas must match
if not all_schemas_match:
    fail_reasons.append(
        "ONE OR MORE V3 SCHEMA MISMATCHES"
    )


# All independent evaluations must succeed
if not all_evals_ok:
    fail_reasons.append(
        "ONE OR MORE INDEPENDENT TEST EVALUATIONS FAILED"
    )


# Predict errors
if predict_errors:
    fail_reasons.append(
        "PREDICT_PROBA ERRORS DETECTED"
    )
    for err in predict_errors:
        print(f"  - {err}")


# Calibration
warnings.append(
    "PROBABILITIES ARE UN-CALIBRATED"
)


# Metrics completeness
if not audit_results:

    fail_reasons.append(
        "NO INDEPENDENT TEST METRICS"
    )

else:

    for key, result in audit_results.items():

        if result.get("pr_auc") is None:

            warnings.append(
                f"{key}: PR-AUC unavailable"
            )

        if result.get("roc_auc") is None:

            warnings.append(
                f"{key}: ROC-AUC unavailable"
            )


if fail_reasons:

    print(
        "🔴 FAIL"
    )

    print()

    print(
        "Reasons:"
    )

    for reason in fail_reasons:

        print(
            f"- {reason}"
        )

elif warnings:

    print(
        "🟡 PASS WITH WARNINGS"
    )

    print()

    print(
        "Warnings:"
    )

    for warning in warnings:

        print(
            f"- {warning}"
        )

else:

    print(
        "🟢 PASS"
    )


# ============================================================
# Final summary
# ============================================================

print()
print(
    "=" * 70
)

print(
    "AUDIT SUMMARY"
)

print(
    "=" * 70
)

print(
    f"Fire events : {len(event_ids)}"
)

print(
    f"Train       : {len(train):,}"
)

print(
    f"Validation  : {len(validation):,}"
)

print(
    f"Test        : {len(test):,}"
)

print()

for horizon, summary in target_summary.items():

    print(
        f"{horizon}: "
        f"positive={summary['positive']:,} "
        f"negative={summary['negative']:,} "
        f"positive%={summary['positive_percent']:.2f}%"
    )

print()
print("Independent Test Metrics Summary (v3):")
print("-" * 70)

for experiment in ("sensor_only", "sensor_plus_flame"):
    for horizon in ("24h", "48h", "72h"):
        key = f"{experiment}_{horizon}"
        if key in audit_results:
            r = audit_results[key]
            print(
                f"{experiment.upper()} / {horizon} | "
                f"Thr={r['threshold']:.2f} | "
                f"Acc={r['accuracy']:.3f} | "
                f"P={r['precision']:.3f} | "
                f"R={r['recall']:.3f} | "
                f"F1={r['f1']:.3f}"
            )
        else:
            print(
                f"{experiment.upper()} / {horizon} | ❌ NO RESULT"
            )

print()

if fail_reasons:

    print(
        "FINAL STATUS: 🔴 FAIL"
    )

elif warnings:

    print(
        "FINAL STATUS: 🟡 PASS WITH WARNINGS"
    )

else:

    print(
        "FINAL STATUS: 🟢 PASS"
    )

print(
    "=" * 70
)

print()
print(
    "⚠️ این اسکریپت هیچ مدل یا Datasetی را تغییر نمی‌دهد."
)

print(
    "⚠️ فقط Audit انجام می‌دهد."
)

print()
print(
    "حالا خروجی کامل این فایل را بفرست."
)
