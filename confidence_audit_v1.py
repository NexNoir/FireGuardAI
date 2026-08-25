"""
FireGuard — Confidence & Uncertainty Audit v2
==============================================

READ-ONLY AUDIT

Purpose:
- Run real inference against the reconstructed V4 feature dataset.
- Use the exact feature schema stored in each existing model.
- Keep probability, confidence, calibration, and uncertainty separate.
- Use only the model's existing calibration mechanism.
- Calculate metric sampling uncertainty with 1000 bootstrap iterations.
- Never train, recalibrate, modify, overwrite, or fabricate anything.

This file intentionally writes no model, dataset, or audit-result file.
"""

from __future__ import annotations

from pathlib import Path
import inspect
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "data" / "fireguard_forecast_60days_v4_features_audit.csv"
MODEL_DIR = BASE_DIR / "saved_models"

BOOTSTRAP_N = 1000
RANDOM_SEED = 42
MIN_BOOTSTRAP_VALID = 50
MIN_SAMPLES = 1

EXPERIMENTS = [
    ("sensor_only", "SENSOR_ONLY"),
    ("sensor_plus_flame", "SENSOR_PLUS_FLAME"),
]

HORIZONS = [24, 48, 72]

FUTURE_TARGETS = {
    24: "fire_next_24h",
    48: "fire_next_48h",
    72: "fire_next_72h",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def display_value(value):
    if value is None:
        return "NOT_AVAILABLE"
    return value


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def normalize_name(name):
    return str(name).strip().lower()


def is_binary_series(series):
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return len(values) > 0 and set(values.tolist()).issubset({0, 1})


def has_predict_proba(obj):
    return callable(getattr(obj, "predict_proba", None))


def has_predict(obj):
    return callable(getattr(obj, "predict", None))


# ============================================================
# MODEL DISCOVERY
# ============================================================

def model_filename(experiment, horizon):
    return (
        f"fireguard_forecast_{experiment}_{horizon}h_v4.joblib"
    )


def find_model(experiment, horizon):
    """
    Read-only exact lookup first, then a recursive read-only lookup.

    No model is copied, saved, moved, or modified.
    """
    filename = model_filename(experiment, horizon)

    candidates = [
        MODEL_DIR / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "model" / filename,
        BASE_DIR / "artifacts" / filename,
        BASE_DIR / filename,
        BASE_DIR.parent / "saved_models" / filename,
        BASE_DIR.parent / "models" / filename,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    search_roots = [
        BASE_DIR,
        BASE_DIR.parent,
    ]

    for root in search_roots:
        if not root.exists():
            continue
        try:
            matches = list(root.rglob(filename))
        except Exception:
            continue
        if matches:
            return matches[0]

    return None


def load_model(experiment, horizon):
    path = find_model(experiment, horizon)
    if path is None:
        return None, None
    return joblib.load(path), path


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    df = pd.read_csv(DATASET)

    print(f"Dataset: {DATASET}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# OBJECT INSPECTION
# ============================================================

def describe_object(obj):
    if obj is None:
        return "None"

    return (
        f"{type(obj).__module__}.{type(obj).__name__}"
    )


def inspect_artifact(artifact):
    """
    Read-only structural inspection.

    The function does not alter the loaded object.
    """
    print(f"Artifact type: {describe_object(artifact)}")

    if isinstance(artifact, dict):
        print(f"Artifact keys: {sorted(map(str, artifact.keys()))}")

        for key in ("model", "calibrator", "features", "feature_names"):
            if key in artifact:
                print(
                    f"  {key}: {describe_object(artifact[key])}"
                )

    else:
        if hasattr(artifact, "steps"):
            try:
                print(
                    "Pipeline steps:",
                    [name for name, _ in artifact.steps],
                )
            except Exception:
                pass

        if hasattr(artifact, "named_steps"):
            try:
                print(
                    "Named steps:",
                    list(artifact.named_steps.keys()),
                )
            except Exception:
                pass


# ============================================================
# FEATURE SCHEMA EXTRACTION
# ============================================================

def _feature_names_from_object(obj):
    """
    Extract a feature schema only when the object itself exposes one.

    Order is preserved exactly as stored by the object.
    No feature order is guessed.
    """
    if obj is None:
        return None

    feature_names = getattr(obj, "feature_names_in_", None)
    if feature_names is not None:
        return [str(x) for x in feature_names]

    # Common custom names used by saved FireGuard artifacts.
    for attr in (
        "feature_names",
        "features",
        "required_features",
        "input_features",
    ):
        value = getattr(obj, attr, None)
        if value is not None:
            if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
                return [str(x) for x in value]

    return None


def extract_required_features(artifact):
    """
    Extract required features from the model artifact without guessing.

    Priority:
    1. Pipeline/model object's feature_names_in_
    2. Explicit artifact feature schema
    3. Nested model schema
    4. Explicit wrapper schema

    If no schema is exposed, inference is refused.
    """
    # A dict artifact is common for custom FireGuard saved objects.
    if isinstance(artifact, dict):
        model = artifact.get("model")
        explicit = artifact.get("feature_names")
        if explicit is None:
            explicit = artifact.get("features")
        if explicit is None:
            explicit = artifact.get("required_features")

        if explicit is not None:
            if isinstance(explicit, (list, tuple, np.ndarray, pd.Index)):
                return [str(x) for x in explicit]

        names = _feature_names_from_object(model)
        if names:
            return names

        names = _feature_names_from_object(artifact)
        if names:
            return names

        # Pipeline/wrapper may be nested under common keys.
        for key in ("pipeline", "estimator", "classifier", "model"):
            nested = artifact.get(key)
            names = _feature_names_from_object(nested)
            if names:
                return names

        return None

    # Direct sklearn estimator / Pipeline / wrapper.
    names = _feature_names_from_object(artifact)
    if names:
        return names

    # Some wrappers expose a nested estimator.
    for attr in ("estimator", "classifier", "model", "pipeline"):
        nested = getattr(artifact, attr, None)
        names = _feature_names_from_object(nested)
        if names:
            return names

    return None


def validate_feature_schema(df, required_features, target_col):
    """
    Validate exact required feature availability and future-target leakage.

    No missing value is fabricated or filled here.
    """
    if not required_features:
        return {
            "ok": False,
            "missing": [],
            "leakage": [
                "Model exposes no readable feature schema; "
                "inference refused rather than guessing."
            ],
        }

    missing = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    leakage = []

    target_lower = normalize_name(target_col) if target_col else None

    future_targets = set(
        normalize_name(name)
        for name in FUTURE_TARGETS.values()
    )

    for feature in required_features:
        f_lower = normalize_name(feature)

        if target_lower and f_lower == target_lower:
            leakage.append(
                f"Current target '{target_col}' is requested as a model feature."
            )

        if f_lower in future_targets:
            # This is allowed only when the model explicitly requests it,
            # but the user requirement says future target leakage must stop
            # unless the model explicitly requests it. Since the schema is
            # exactly what the model requests, mark it as explicitly requested
            # and let the audit expose it.
            leakage.append(
                f"Future target '{feature}' is explicitly requested by the model."
            )

    return {
        "ok": len(missing) == 0 and len(leakage) == 0,
        "missing": missing,
        "leakage": leakage,
    }


# ============================================================
# CALIBRATION DETECTION
# ============================================================

def calibration_method(obj):
    """
    Detect an existing calibration method without creating one.

    Returns YES/NO/UNKNOWN and the method name.
    """
    if obj is None:
        return "UNKNOWN", "NOT_AVAILABLE"

    cls_name = type(obj).__name__.lower()

    # CalibratedClassifierCV and similar sklearn calibration wrappers.
    if "calibratedclassifier" in cls_name:
        method = getattr(obj, "method", None)
        if method is not None:
            return "YES", str(method)
        return "YES", "UNKNOWN"

    # Direct calibration objects / wrappers.
    for attr in ("calibration_method", "method"):
        value = getattr(obj, attr, None)
        if value is not None:
            value = str(value).lower()
            if value in {"sigmoid", "isotonic"}:
                return "YES", value

    # IsotonicRegression / logistic calibrator naming.
    if "isotonic" in cls_name:
        return "YES", "isotonic"

    if (
        "sigmoid" in cls_name
        or "logistic" in cls_name
        or "platt" in cls_name
    ):
        return "YES", "sigmoid"

    return "NO", "NOT_AVAILABLE"


def detect_calibration(artifact):
    """
    Inspect the actual object graph, read-only.

    Preference is given to a wrapper's own predict_proba because that is
    the safest way to use an already calibrated sklearn pipeline.
    """
    objects = []

    if isinstance(artifact, dict):
        for key in (
            "calibrator",
            "model",
            "pipeline",
            "estimator",
            "classifier",
        ):
            if key in artifact:
                objects.append(artifact[key])
    else:
        objects.append(artifact)
        for attr in ("calibrator", "model", "pipeline", "estimator"):
            nested = getattr(artifact, attr, None)
            if nested is not None:
                objects.append(nested)

    for obj in objects:
        detected, method = calibration_method(obj)
        if detected == "YES":
            return True, method

    # A custom artifact can explicitly carry a calibrator.
    if isinstance(artifact, dict) and artifact.get("calibrator") is not None:
        cal = artifact["calibrator"]
        detected, method = calibration_method(cal)
        if detected == "YES":
            return True, method
        return True, "UNKNOWN"

    return False, "NOT_AVAILABLE"


# ============================================================
# INFERENCE
# ============================================================

def get_primary_predictor(artifact):
    """
    Select an inference object without guessing.

    If the outer artifact can predict probabilities, it is preferred.
    Otherwise a direct model/pipeline inside a dict is used.
    """
    if has_predict_proba(artifact):
        return artifact

    if isinstance(artifact, dict):
        for key in (
            "pipeline",
            "model",
            "estimator",
            "classifier",
        ):
            obj = artifact.get(key)
            if has_predict_proba(obj):
                return obj

    return None


def call_predict_proba(predictor, X):
    """
    Call the existing model's predict_proba exactly.

    No post-processing, calibration, clipping, fallback prediction, or
    fabricated probability is performed here.
    """
    if not has_predict_proba(predictor):
        raise RuntimeError(
            f"{describe_object(predictor)} has no predict_proba()."
        )

    proba = predictor.predict_proba(X)

    arr = np.asarray(proba)

    if arr.ndim != 2 or arr.shape[0] != len(X):
        raise RuntimeError(
            "predict_proba() returned an unexpected shape: "
            f"{arr.shape}"
        )

    if arr.shape[1] == 2:
        classes = getattr(predictor, "classes_", None)

        if classes is not None:
            classes = list(np.asarray(classes))
            if 1 in classes:
                positive_index = classes.index(1)
            elif "1" in classes:
                positive_index = classes.index("1")
            else:
                raise RuntimeError(
                    "Binary predict_proba() has no class label 1."
                )
        else:
            # A two-column binary sklearn probability matrix conventionally
            # maps columns to classes_. If classes_ is absent, the mapping
            # cannot be proven safely.
            raise RuntimeError(
                "Binary predict_proba() returned two columns but the "
                "predictor exposes no classes_; positive class cannot be "
                "identified safely."
            )

        probability = arr[:, positive_index]

    elif arr.shape[1] == 1:
        probability = arr[:, 0]
    else:
        raise RuntimeError(
            "predict_proba() is multiclass; FireGuard audit requires a "
            "binary positive-class probability."
        )

    probability = np.asarray(probability, dtype=float)

    if not np.all(np.isfinite(probability)):
        raise RuntimeError(
            "predict_proba() returned NaN or infinite probability values."
        )

    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RuntimeError(
            "predict_proba() returned values outside [0, 1]."
        )

    return probability


def get_probability(artifact, X):
    """
    Obtain probability from the existing model.

    Supported safely:
    - sklearn Pipeline
    - CalibratedClassifierCV
    - direct sklearn classifier
    - custom wrapper exposing predict_proba()
    - dict artifact whose model/pipeline exposes predict_proba()

    A separate calibrator is used only when the artifact's actual public
    inference structure exposes it in a demonstrably usable way.

    If the separate calibrator's input contract cannot be established,
    inference is refused rather than guessed.
    """
    # Safest path: outer artifact is itself the calibrated predictor.
    predictor = get_primary_predictor(artifact)

    if predictor is not None:
        return call_predict_proba(predictor, X), describe_object(predictor)

    if not isinstance(artifact, dict):
        raise RuntimeError(
            "No usable predict_proba() predictor was found."
        )

    model = artifact.get("model")
    calibrator = artifact.get("calibrator")

    if model is None or calibrator is None:
        raise RuntimeError(
            "Artifact has neither a usable predictor nor a complete "
            "model + calibrator pair."
        )

    if not has_predict_proba(model):
        raise RuntimeError(
            f"Underlying model {describe_object(model)} has no predict_proba()."
        )

    # A separate calibrator must advertise a usable input schema/contract.
    # We do not guess whether it expects X, probabilities, or decision scores.
    cal_names = _feature_names_from_object(calibrator)

    if cal_names:
        # Calibrator itself claims to consume feature rows.
        missing = [f for f in cal_names if f not in X.columns]
        if missing:
            raise RuntimeError(
                "Separate calibrator declares input features that are not "
                f"available: {missing}"
            )
        if not has_predict_proba(calibrator):
            raise RuntimeError(
                "Separate calibrator exposes feature schema but no "
                "predict_proba()."
            )
        return call_predict_proba(calibrator, X), describe_object(calibrator)

    # If a calibrator is a standard sklearn CalibratedClassifierCV, it should
    # have been caught as the primary predictor. Do not invent an input
    # convention for an opaque custom object.
    raise RuntimeError(
        "Separate calibrator exists, but its probability-input contract "
        "cannot be established safely. Inference refused."
    )


# ============================================================
# TARGET / LEAKAGE
# ============================================================

def get_target_column(df, horizon):
    target = FUTURE_TARGETS[horizon]

    if target not in df.columns:
        return None

    return target


def prepare_target(df, target_col):
    """
    Read an already-defined future target from the V4 audit dataset.

    No future target is constructed from fire_now.
    """
    series = pd.to_numeric(df[target_col], errors="coerce")

    valid = series.notna()
    y = series.loc[valid].to_numpy(dtype=float)

    if len(y) == 0:
        raise RuntimeError(
            f"Target '{target_col}' contains no valid numeric samples."
        )

    if not np.all(np.isin(y, [0, 1])):
        raise RuntimeError(
            f"Target '{target_col}' is not binary 0/1."
        )

    return valid.to_numpy(), y.astype(int)


def check_input_data(df, required_features, target_col):
    """
    Refuse inference if required features contain missing/non-numeric values.

    No imputation is performed.
    """
    schema = validate_feature_schema(
        df,
        required_features,
        target_col,
    )

    if not schema["ok"]:
        return schema, None, None

    X = df.loc[:, required_features].copy()

    # Models in this audit are expected to use numeric engineered features.
    # Convertibility is checked without changing the source dataframe.
    for feature in required_features:
        numeric = pd.to_numeric(X[feature], errors="coerce")
        if numeric.isna().any():
            bad = int(numeric.isna().sum())
            return (
                {
                    "ok": False,
                    "missing": [],
                    "leakage": [],
                    "invalid_values": [
                        f"{feature}: {bad} non-numeric/NaN values"
                    ],
                },
                None,
                None,
            )
        X[feature] = numeric

    return schema, X, None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_prob):
    """
    Calculate metrics independently.

    A metric that cannot be computed is recorded as NOT_AVAILABLE.
    """
    result = {
        "brier": None,
        "log_loss": None,
        "roc_auc": None,
        "pr_auc": None,
    }

    if len(y_true) == 0:
        return result

    try:
        result["brier"] = safe_float(
            brier_score_loss(y_true, y_prob)
        )
    except Exception:
        pass

    try:
        result["log_loss"] = safe_float(
            log_loss(
                y_true,
                y_prob,
                labels=[0, 1],
            )
        )
    except Exception:
        pass

    if len(np.unique(y_true)) >= 2:
        try:
            result["roc_auc"] = safe_float(
                roc_auc_score(y_true, y_prob)
            )
        except Exception:
            pass

        try:
            result["pr_auc"] = safe_float(
                average_precision_score(y_true, y_prob)
            )
        except Exception:
            pass

    return result


# ============================================================
# BOOTSTRAP UNCERTAINTY
# ============================================================

def bootstrap_metric(
    y_true,
    y_prob,
    metric_name,
    n_bootstrap=BOOTSTRAP_N,
    seed=RANDOM_SEED,
):
    """
    Nonparametric bootstrap CI for a metric.

    This is sampling uncertainty of the evaluation metric.
    It is NOT model confidence and NOT probability uncertainty.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n = len(y_true)

    if n < 30:
        return None

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        try:
            if metric_name == "roc_auc":
                if len(np.unique(y_b)) < 2:
                    continue
                value = roc_auc_score(y_b, p_b)

            elif metric_name == "pr_auc":
                if len(np.unique(y_b)) < 2:
                    continue
                value = average_precision_score(y_b, p_b)

            elif metric_name == "brier":
                value = brier_score_loss(y_b, p_b)

            elif metric_name == "log_loss":
                value = log_loss(
                    y_b,
                    p_b,
                    labels=[0, 1],
                )

            else:
                raise ValueError(
                    f"Unknown metric: {metric_name}"
                )

            if np.isfinite(value):
                values.append(float(value))

        except Exception:
            continue

    if len(values) < MIN_BOOTSTRAP_VALID:
        return None

    values = np.asarray(values, dtype=float)

    return {
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
        "n_valid": int(len(values)),
    }


def bootstrap_all(y_true, y_prob):
    result = {}

    for metric in (
        "brier",
        "log_loss",
        "roc_auc",
        "pr_auc",
    ):
        result[metric] = bootstrap_metric(
            y_true,
            y_prob,
            metric,
            n_bootstrap=BOOTSTRAP_N,
            seed=RANDOM_SEED,
        )

    return result


# ============================================================
# AUDIT ONE MODEL
# ============================================================

def audit_model(df, experiment, experiment_label, horizon):
    target_col = FUTURE_TARGETS[horizon]

    result = {
        "experiment": experiment_label,
        "horizon": horizon,
        "status": "NOT_AVAILABLE",
        "samples": None,
        "positive": None,
        "negative": None,
        "positive_rate": None,
        "calibrated": "UNKNOWN",
        "calibration_method": "NOT_AVAILABLE",
        "probability_available": "NO",
        "confidence": "NOT_AVAILABLE",
        "brier": None,
        "brier_ci_low": None,
        "brier_ci_high": None,
        "log_loss": None,
        "log_loss_ci_low": None,
        "log_loss_ci_high": None,
        "roc_auc": None,
        "roc_auc_ci_low": None,
        "roc_auc_ci_high": None,
        "pr_auc": None,
        "pr_auc_ci_low": None,
        "pr_auc_ci_high": None,
        "uncertainty_status": "NOT_AVAILABLE",
        "model_path": None,
        "required_features_status": "NOT_AVAILABLE",
        "feature_order_status": "NOT_AVAILABLE",
        "reason": None,
    }

    print("=" * 70)
    print(f"{experiment_label} / {horizon}h")
    print("=" * 70)

    artifact, model_path = load_model(
        experiment,
        horizon,
    )

    if artifact is None:
        result["status"] = "MODEL_NOT_FOUND"
        result["reason"] = "Existing V4 model was not found."
        print("Model: NOT FOUND")
        return result

    result["model_path"] = str(model_path)

    print(f"Model: {model_path}")

    inspect_artifact(artifact)

    calibrated, method = detect_calibration(artifact)
    result["calibrated"] = "YES" if calibrated else "NO"
    result["calibration_method"] = method

    print(
        f"Calibration: {result['calibrated']}"
    )
    print(
        f"Calibration method: {method}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print(f"Target: {target_col}")

    if target_col not in df.columns:
        result["status"] = "TARGET_UNAVAILABLE"
        result["reason"] = (
            f"Required future target '{target_col}' is absent."
        )
        print("Target: NOT FOUND")
        return result

    try:
        target_valid_mask, y_all = prepare_target(
            df,
            target_col,
        )
    except Exception as exc:
        result["status"] = "TARGET_INVALID"
        result["reason"] = str(exc)
        print(f"Target validation failed: {exc}")
        return result

    # --------------------------------------------------------
    # FEATURE SCHEMA
    # --------------------------------------------------------

    required_features = extract_required_features(artifact)

    if required_features is None:
        result["status"] = "FEATURE_SCHEMA_UNAVAILABLE"
        result["reason"] = (
            "Model does not expose a readable feature schema. "
            "Inference refused; feature order was not guessed."
        )
        print("Required features: NOT_AVAILABLE")
        print("Inference: BLOCKED")
        print(result["reason"])
        return result

    print(f"Required features ({len(required_features)}):")
    print("  " + ", ".join(required_features))

    schema, X, _ = check_input_data(
        df,
        required_features,
        target_col,
    )

    if schema.get("missing"):
        result["status"] = "MISSING_FEATURES"
        result["reason"] = (
            "Required model features are missing: "
            + ", ".join(schema["missing"])
        )
        result["required_features_status"] = "FAIL"
        result["feature_order_status"] = "NOT_AVAILABLE"

        print("Required features: FAIL")
        print("Missing:")
        for feature in schema["missing"]:
            print(f"  - {feature}")
        print("Inference: BLOCKED")
        return result

    if schema.get("leakage"):
        result["status"] = "LEAKAGE_DETECTED"
        result["reason"] = " | ".join(schema["leakage"])
        result["required_features_status"] = "FAIL"
        result["feature_order_status"] = "FAIL"

        print("DATA LEAKAGE: DETECTED")
        for item in schema["leakage"]:
            print(f"  - {item}")
        print("Inference: BLOCKED")
        return result

    if schema.get("invalid_values"):
        result["status"] = "INVALID_FEATURE_VALUES"
        result["reason"] = " | ".join(schema["invalid_values"])
        result["required_features_status"] = "FAIL"
        result["feature_order_status"] = "PASS"

        print("Required features: FAIL")
        for item in schema["invalid_values"]:
            print(f"  - {item}")
        print("Inference: BLOCKED")
        return result

    result["required_features_status"] = "PASS"
    result["feature_order_status"] = "PASS"

    print("Required features: PASS")
    print("Feature order: PASS")
    print("Inference: running existing model only...")

    # --------------------------------------------------------
    # REAL INFERENCE
    # --------------------------------------------------------

    try:
        probabilities, predictor_type = get_probability(
            artifact,
            X,
        )
    except Exception as exc:
        result["status"] = "PROBABILITY_INFERENCE_FAILED"
        result["reason"] = str(exc)
        print(f"Inference: FAILED")
        print(f"Reason: {exc}")
        return result

    # No probability post-processing is performed.
    # The model output itself is the probability.
    if len(probabilities) != len(df):
        result["status"] = "PROBABILITY_LENGTH_MISMATCH"
        result["reason"] = (
            f"predict_proba returned {len(probabilities)} rows for "
            f"{len(df)} dataset rows."
        )
        print("Inference: FAILED")
        print(result["reason"])
        return result

    result["probability_available"] = "YES"

    print("Inference: OK")
    print(f"Probability source: {predictor_type}")
    print(
        f"Probability MIN:  {np.min(probabilities):.6f}"
    )
    print(
        f"Probability MAX:  {np.max(probabilities):.6f}"
    )
    print(
        f"Probability MEAN: {np.mean(probabilities):.6f}"
    )

    # --------------------------------------------------------
    # ALIGN TARGET AND PROBABILITY BY THE SAME ROWS
    # --------------------------------------------------------

    target_mask = target_valid_mask
    y_true = y_all

    y_prob = probabilities[target_mask]

    if len(y_true) != len(y_prob):
        result["status"] = "ALIGNMENT_FAILED"
        result["reason"] = (
            "Target and probability arrays could not be aligned."
        )
        print("Alignment: FAILED")
        return result

    n = len(y_true)
    positive = int(np.sum(y_true == 1))
    negative = int(np.sum(y_true == 0))

    result["samples"] = n
    result["positive"] = positive
    result["negative"] = negative
    result["positive_rate"] = (
        float(positive / n) if n else None
    )

    print()
    print(f"Samples: {n:,}")
    print(f"Positive: {positive:,}")
    print(f"Negative: {negative:,}")
    print(
        f"Positive rate: "
        f"{result['positive_rate']:.6f}"
        if n
        else "Positive rate: NOT_AVAILABLE"
    )

    if n < MIN_SAMPLES:
        result["status"] = "INSUFFICIENT_SAMPLES"
        result["reason"] = "No valid labeled samples."
        return result

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    # Probability is deliberately NOT renamed confidence.
    result["confidence"] = "NOT_AVAILABLE"

    print()
    print("Confidence: NOT_AVAILABLE")
    print("Rule: probability != confidence")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_prob,
    )

    result["brier"] = metrics["brier"]
    result["log_loss"] = metrics["log_loss"]
    result["roc_auc"] = metrics["roc_auc"]
    result["pr_auc"] = metrics["pr_auc"]

    print()
    print(
        f"Brier: "
        f"{display_value(result['brier'])}"
    )
    print(
        f"Log Loss: "
        f"{display_value(result['log_loss'])}"
    )
    print(
        f"ROC-AUC: "
        f"{display_value(result['roc_auc'])}"
    )
    print(
        f"PR-AUC: "
        f"{display_value(result['pr_auc'])}"
    )

    if len(np.unique(y_true)) < 2:
        print(
            "ROC-AUC: NOT_AVAILABLE "
            "(only one target class)"
        )
        print(
            "PR-AUC: NOT_AVAILABLE "
            "(only one target class)"
        )

    # --------------------------------------------------------
    # BOOTSTRAP
    # --------------------------------------------------------

    bootstrap = bootstrap_all(
        y_true,
        y_prob,
    )

    uncertainty_count = 0

    print()
    print("Uncertainty:")
    print(
        f"Bootstrap sampling uncertainty "
        f"({BOOTSTRAP_N} iterations)"
    )

    metric_names = [
        ("brier", "Brier"),
        ("log_loss", "Log Loss"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]

    for key, label in metric_names:
        boot = bootstrap.get(key)

        if boot is None:
            print(
                f"{label} 95% CI: NOT_AVAILABLE"
            )
            continue

        uncertainty_count += 1

        result[f"{key}_ci_low"] = boot["lower_95"]
        result[f"{key}_ci_high"] = boot["upper_95"]

        print(
            f"{label} 95% CI: "
            f"[{boot['lower_95']:.6f}, "
            f"{boot['upper_95']:.6f}]"
        )

    result["uncertainty_status"] = (
        "BOOTSTRAP_AVAILABLE"
        if uncertainty_count > 0
        else "NOT_AVAILABLE"
    )

    result["status"] = "OK"

    print()
    print("Calibration:")
    print(f"  calibrated = {result['calibrated']}")
    print(
        f"  method     = "
        f"{result['calibration_method']}"
    )

    print()
    print(
        "Uncertainty interpretation: bootstrap CI is sampling "
        "uncertainty of evaluation metrics, not model confidence."
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

SUMMARY_COLUMNS = [
    "experiment",
    "horizon",
    "status",
    "samples",
    "positive",
    "negative",
    "calibrated",
    "calibration_method",
    "probability_available",
    "confidence",
    "brier",
    "brier_ci_low",
    "brier_ci_high",
    "log_loss",
    "log_loss_ci_low",
    "log_loss_ci_high",
    "roc_auc",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
    "pr_auc",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "uncertainty_status",
]


def print_summary(results):
    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    rows = []
    for result in results:
        rows.append(
            {
                key: result.get(key)
                for key in SUMMARY_COLUMNS
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=SUMMARY_COLUMNS,
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(summary.to_string(index=False))

    return summary


# ============================================================
# FINAL CHECKPOINT
# ============================================================

def print_final_checkpoint(results):
    total = 6

    models_discovered = sum(
        1
        for r in results
        if r.get("status") != "MODEL_NOT_FOUND"
    )

    models_inferred = sum(
        1
        for r in results
        if r.get("status") == "OK"
    )

    probability_available = sum(
        1
        for r in results
        if r.get("probability_available") == "YES"
    )

    calibration_detected = sum(
        1
        for r in results
        if r.get("calibrated") == "YES"
    )

    confidence_available = sum(
        1
        for r in results
        if r.get("confidence") != "NOT_AVAILABLE"
    )

    bootstrap_available = sum(
        1
        for r in results
        if r.get("uncertainty_status") == "BOOTSTRAP_AVAILABLE"
    )

    print()
    print("=" * 70)
    print("FINAL CHECKPOINT")
    print("=" * 70)

    print(
        f"Models discovered: {models_discovered}/{total}"
    )
    print(
        f"Models inferred: {models_inferred}/{total}"
    )
    print(
        f"Probability available: "
        f"{probability_available}/{total}"
    )
    print(
        f"Calibration detected: "
        f"{calibration_detected}/{total}"
    )
    print(
        f"Confidence available: "
        f"{confidence_available}/{total}"
    )
    print(
        f"Bootstrap uncertainty: "
        f"{bootstrap_available}/{total}"
    )

    print()
    print("Models modified: NO")
    print("Dataset modified: NO")
    print("New calibration: NO")
    print("Fabricated probability: NO")
    print("Fabricated confidence: NO")
    print("Fabricated uncertainty: NO")

    print()
    print("Models that did not infer successfully:")

    failures = [
        r
        for r in results
        if r.get("status") != "OK"
    ]

    if not failures:
        print("  NONE")
    else:
        for r in failures:
            print(
                f"  - {r['experiment']} / "
                f"{r['horizon']}h: "
                f"{r.get('status')} — "
                f"{r.get('reason')}"
            )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔥 FireGuard — Confidence & Uncertainty Audit v2")
    print("=" * 70)

    print("READ-ONLY AUDIT")
    print("Models modified : NO")
    print("Dataset modified: NO")
    print("Training        : NO")
    print("Calibration     : NO NEW CALIBRATION")
    print(f"Bootstrap       : {BOOTSTRAP_N}")
    print("Fabricated data : NO")

    print()

    df = load_dataset()

    results = []

    for experiment, experiment_label in EXPERIMENTS:
        for horizon in HORIZONS:
            try:
                result = audit_model(
                    df,
                    experiment,
                    experiment_label,
                    horizon,
                )
            except Exception as exc:
                result = {
                    "experiment": experiment_label,
                    "horizon": horizon,
                    "status": "AUDIT_ERROR",
                    "samples": None,
                    "positive": None,
                    "negative": None,
                    "positive_rate": None,
                    "calibrated": "UNKNOWN",
                    "calibration_method": "NOT_AVAILABLE",
                    "probability_available": "NO",
                    "confidence": "NOT_AVAILABLE",
                    "brier": None,
                    "brier_ci_low": None,
                    "brier_ci_high": None,
                    "log_loss": None,
                    "log_loss_ci_low": None,
                    "log_loss_ci_high": None,
                    "roc_auc": None,
                    "roc_auc_ci_low": None,
                    "roc_auc_ci_high": None,
                    "pr_auc": None,
                    "pr_auc_ci_low": None,
                    "pr_auc_ci_high": None,
                    "uncertainty_status": "NOT_AVAILABLE",
                    "model_path": None,
                    "required_features_status": "NOT_AVAILABLE",
                    "feature_order_status": "NOT_AVAILABLE",
                    "reason": str(exc),
                }

                print()
                print(
                    f"AUDIT ERROR — "
                    f"{experiment_label} / {horizon}h: {exc}"
                )

            results.append(result)
            print()

    print_summary(results)
    print_final_checkpoint(results)

    print()
    print("READ-ONLY GUARANTEE:")
    print("No model.save/joblib.dump was called.")
    print("No dataset write operation was called.")
    print("No training was performed.")
    print("No new calibration was performed.")
    print("No probability was post-processed.")
    print("No confidence percentage was fabricated.")
    print("Bootstrap CI is metric sampling uncertainty only.")


if __name__ == "__main__":
    main()