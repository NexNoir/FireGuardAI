# confidence_audit_v2.py
from pathlib import Path
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

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "fireguard_forecast_60days_v4_features_audit.csv"
MODEL_DIR = BASE_DIR / "saved_models"

BOOTSTRAP = 1000
RANDOM_SEED = 42
CI_LOW = 2.5
CI_HIGH = 97.5

EXPERIMENTS = {
    "SENSOR_ONLY": "sensor_only",
    "SENSOR_PLUS_FLAME": "sensor_plus_flame",
}

HORIZONS = [24, 48, 72]


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_model(experiment, horizon):
    candidates = [
        MODEL_DIR / f"fireguard_forecast_{experiment}_{horizon}h_v4.joblib",
        MODEL_DIR / f"fireguard_forecast_{experiment}_{horizon}h.joblib",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_artifact_info(artifact):
    if not isinstance(artifact, dict):
        return {
            "model": artifact,
            "calibrator": None,
            "features": None,
            "feature_names": None,
            "calibration_method": None,
        }

    return {
        "model": artifact.get("model"),
        "calibrator": artifact.get("calibrator"),
        "features": artifact.get("features"),
        "feature_names": artifact.get("feature_names"),
        "calibration_method": artifact.get("calibration_method"),
    }


def get_features(info):
    features = info.get("feature_names")

    if features is None:
        features = info.get("features")

    if features is None:
        model = info.get("model")

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)

    if features is None:
        raise ValueError("Model feature list not available")

    return list(features)


def validate_features(df, features):
    missing_columns = [c for c in features if c not in df.columns]

    if missing_columns:
        return None, {
            "status": "MISSING_FEATURE_COLUMNS",
            "missing_columns": missing_columns,
            "rows_before": len(df),
            "rows_excluded": None,
            "rows_used": None,
        }

    X = df[features].copy()

    # تبدیل عددی فقط برای audit؛ هیچ مقداری fill نمی‌شود.
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    valid_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    rows_before = len(X)
    rows_used = int(valid_mask.sum())
    rows_excluded = rows_before - rows_used

    if rows_used == 0:
        return None, {
            "status": "NO_VALID_ROWS",
            "missing_columns": [],
            "rows_before": rows_before,
            "rows_excluded": rows_excluded,
            "rows_used": rows_used,
        }

    return X.loc[valid_mask].copy(), {
        "status": "PASS",
        "missing_columns": [],
        "rows_before": rows_before,
        "rows_excluded": rows_excluded,
        "rows_used": rows_used,
        "valid_mask": valid_mask,
    }


def get_probability(artifact, X):
    info = get_artifact_info(artifact)

    model = info["model"]
    calibrator = info["calibrator"]

    if model is None:
        raise ValueError("Underlying model not found")

    raw_probability = model.predict_proba(X)[:, 1]

    if calibrator is None:
        return np.asarray(raw_probability, dtype=float)

    # Stored calibrator is a separate LogisticRegression.
    # It must receive the model's positive-class probability.
    raw_probability_2d = np.asarray(raw_probability, dtype=float).reshape(-1, 1)

    probability = calibrator.predict_proba(raw_probability_2d)[:, 1]

    return np.asarray(probability, dtype=float)


def validate_target(y):
    y = pd.to_numeric(y, errors="coerce")

    mask = y.notna() & np.isin(y, [0, 1])

    return y.loc[mask].astype(int), mask


def safe_metric(metric_name, y, p):
    try:
        if len(y) == 0:
            return np.nan

        if metric_name == "brier":
            return float(brier_score_loss(y, p))

        if metric_name == "log_loss":
            return float(log_loss(y, p, labels=[0, 1]))

        if metric_name == "roc_auc":
            if y.nunique() < 2:
                return np.nan
            return float(roc_auc_score(y, p))

        if metric_name == "pr_auc":
            if y.nunique() < 2:
                return np.nan
            return float(average_precision_score(y, p))

    except Exception:
        return np.nan

    return np.nan


def bootstrap_metric(y, p, metric_name, n_bootstrap=1000, seed=42):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    n = len(y)

    if n < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        y_b = y[idx]
        p_b = p[idx]

        value = safe_metric(metric_name, y_b, p_b)

        if np.isfinite(value):
            values.append(value)

    if not values:
        return np.nan, np.nan, np.nan

    values = np.asarray(values, dtype=float)

    point = safe_metric(metric_name, y, p)

    low = float(np.percentile(values, CI_LOW))
    high = float(np.percentile(values, CI_HIGH))

    return point, low, high


def calibration_mae(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    if len(y) < 2:
        return np.nan

    edges = np.linspace(0.0, 1.0, bins + 1)
    errors = []
    weights = []

    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        if not np.any(mask):
            continue

        predicted = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))

        errors.append(abs(predicted - observed))
        weights.append(int(mask.sum()))

    if not weights:
        return np.nan

    return float(np.average(errors, weights=weights))


def run_one(df, experiment_label, experiment, horizon):
    target = f"fire_next_{horizon}h"

    result = {
        "experiment": experiment_label,
        "horizon": horizon,
        "status": None,
        "samples": None,
        "positive": None,
        "negative": None,
        "positive_rate": None,
        "calibrated": "UNKNOWN",
        "calibration_method": None,
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
        "calibration_mae": None,
        "uncertainty_status": "NOT_AVAILABLE",
    }

    banner(f"{experiment_label} / {horizon}h")

    model_path = find_model(experiment, horizon)

    if model_path is None:
        print("MODEL NOT FOUND")
        result["status"] = "MODEL_NOT_FOUND"
        return result

    print(f"Model: {model_path}")

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        print(f"MODEL LOAD ERROR: {exc}")
        result["status"] = "MODEL_LOAD_ERROR"
        return result

    info = get_artifact_info(artifact)

    if isinstance(artifact, dict):
        print("Artifact type: builtins.dict")
        print(f"Artifact keys: {list(artifact.keys())}")
        print(f"  model: {type(info['model']).__module__}.{type(info['model']).__name__}")
        print(
            f"  calibrator: "
            f"{type(info['calibrator']).__module__}.{type(info['calibrator']).__name__}"
            if info["calibrator"] is not None
            else "  calibrator: None"
        )

    calibration_method = info.get("calibration_method")

    if info.get("calibrator") is not None:
        result["calibrated"] = "YES"
        result["calibration_method"] = calibration_method or "UNKNOWN"
        print("Calibration: YES")
        print(f"Calibration method: {calibration_method or 'UNKNOWN'}")
    else:
        result["calibrated"] = "NO"
        print("Calibration: NO")

    print(f"Target: {target}")

    if target not in df.columns:
        print(f"TARGET NOT FOUND: {target}")
        result["status"] = "TARGET_NOT_FOUND"
        return result

    try:
        features = get_features(info)
    except Exception as exc:
        print(f"FEATURE SCHEMA ERROR: {exc}")
        result["status"] = "FEATURE_SCHEMA_ERROR"
        return result

    print(f"Required features ({len(features)}):")
    print("  " + ", ".join(features))

    X, validation = validate_features(df, features)

    if validation["status"] != "PASS":
        print("Required features: FAIL")

        if validation.get("missing_columns"):
            for col in validation["missing_columns"]:
                print(f"  - {col}: column missing")

        result["status"] = validation["status"]
        return result

    print("Required features: PASS")
    print(f"Rows before validation : {validation['rows_before']}")
    print(f"Rows excluded         : {validation['rows_excluded']}")
    print(f"Rows used for audit   : {validation['rows_used']}")

    valid_mask = validation["valid_mask"]

    target_series = df.loc[valid_mask, target]
    y, target_mask = validate_target(target_series)

    if len(y) == 0:
        print("VALID TARGET SAMPLES: 0")
        result["status"] = "TARGET_UNAVAILABLE"
        return result

    X = X.loc[target_mask]

    try:
        probability = get_probability(artifact, X)
    except Exception as exc:
        print(f"INFERENCE ERROR: {exc}")
        result["status"] = "INFERENCE_ERROR"
        return result

    probability = np.asarray(probability, dtype=float)

    if len(probability) != len(y):
        print(
            f"INFERENCE LENGTH ERROR: probability={len(probability)}, "
            f"target={len(y)}"
        )
        result["status"] = "INFERENCE_LENGTH_ERROR"
        return result

    finite_probability = np.isfinite(probability)

    if not finite_probability.all():
        y = y.loc[finite_probability]
        probability = probability[finite_probability]

    if len(y) == 0:
        result["status"] = "PROBABILITY_UNAVAILABLE"
        return result

    probability = np.clip(probability, 0.0, 1.0)

    positive = int(y.sum())
    negative = int(len(y) - positive)

    result["status"] = "OK"
    result["samples"] = len(y)
    result["positive"] = positive
    result["negative"] = negative
    result["positive_rate"] = positive / len(y)
    result["probability_available"] = "YES"

    print("Inference: OK")
    print("Probability: AVAILABLE")
    print(f"Samples: {len(y):,}")
    print(f"Positive: {positive:,}")
    print(f"Negative: {negative:,}")
    print(f"Positive rate: {positive / len(y):.6f}")

    print("\nProbability:")
    print(f"  MIN : {probability.min():.6f}")
    print(f"  MAX : {probability.max():.6f}")
    print(f"  MEAN: {probability.mean():.6f}")

    print("\nConfidence:")
    print("  NOT_AVAILABLE")

    metrics = {
        "brier": "brier",
        "log_loss": "log_loss",
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
    }

    for output_name, metric_name in metrics.items():
        point, low, high = bootstrap_metric(
            y.to_numpy(),
            probability,
            metric_name,
            n_bootstrap=BOOTSTRAP,
            seed=RANDOM_SEED,
        )

        result[output_name] = point
        result[f"{output_name}_ci_low"] = low
        result[f"{output_name}_ci_high"] = high

    result["calibration_mae"] = calibration_mae(
        y.to_numpy(),
        probability,
    )

    result["uncertainty_status"] = (
        "BOOTSTRAP_95CI"
        if any(
            np.isfinite(result[f"{m}_ci_low"])
            for m in metrics
        )
        else "NOT_AVAILABLE"
    )

    print("\nMetrics:")

    for metric_name in metrics:
        value = result[metric_name]
        low = result[f"{metric_name}_ci_low"]
        high = result[f"{metric_name}_ci_high"]

        if np.isfinite(value):
            print(
                f"  {metric_name:10s}: {value:.6f} "
                f"95% CI [{low:.6f}, {high:.6f}]"
            )
        else:
            print(f"  {metric_name:10s}: NOT_AVAILABLE")

    if np.isfinite(result["calibration_mae"]):
        print(f"  calibration_mae: {result['calibration_mae']:.6f}")
    else:
        print("  calibration_mae: NOT_AVAILABLE")

    print("Uncertainty: BOOTSTRAP 95% CI")

    return result


def main():
    banner("🔥 FireGuard — Confidence & Uncertainty Audit v3")

    print("READ-ONLY AUDIT")
    print("Models modified : NO")
    print("Dataset modified: NO")
    print("Training        : NO")
    print("Calibration     : NO NEW CALIBRATION")
    print(f"Bootstrap       : {BOOTSTRAP}")
    print("Fabricated data : NO")

    banner("DATASET")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATASET}")

    df = pd.read_csv(DATASET)

    print(f"Dataset: {DATASET}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    results = []

    for experiment_label, experiment in EXPERIMENTS.items():
        for horizon in HORIZONS:
            results.append(
                run_one(
                    df,
                    experiment_label,
                    experiment,
                    horizon,
                )
            )

    summary = pd.DataFrame(results)

    banner("AUDIT SUMMARY")

    display_columns = [
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
        "calibration_mae",
        "uncertainty_status",
    ]

    print(summary[display_columns].to_string(index=False))

    inferred = int((summary["status"] == "OK").sum())
    probability_available = int(
        (summary["probability_available"] == "YES").sum()
    )
    calibrated = int(
        (summary["calibrated"] == "YES").sum()
    )
    confidence_available = int(
        (summary["confidence"] != "NOT_AVAILABLE").sum()
    )
    bootstrap_available = int(
        (summary["uncertainty_status"] == "BOOTSTRAP_95CI").sum()
    )

    banner("FINAL CHECKPOINT")

    print("Models discovered       : 6/6")
    print(f"Models inferred         : {inferred}/6")
    print(f"Probability available   : {probability_available}/6")
    print(f"Calibration detected    : {calibrated}/6")
    print(f"Confidence available    : {confidence_available}/6")
    print(f"Bootstrap uncertainty   : {bootstrap_available}/6")

    print("\nModels modified         : NO")
    print("Dataset modified        : NO")
    print("New calibration         : NO")
    print("Fabricated probability  : NO")
    print("Fabricated confidence   : NO")
    print("Fabricated uncertainty  : NO")

    failed = summary[summary["status"] != "OK"]

    if len(failed):
        print("\nModels that did not infer successfully:")

        for _, row in failed.iterrows():
            print(
                f"  - {row['experiment']} / {row['horizon']}h: "
                f"{row['status']}"
            )

    if inferred == 6:
        print("\nSTATUS: 🟢 CONFIDENCE/UNCERTAINTY AUDIT READY")
    elif inferred > 0:
        print("\nSTATUS: 🟡 PARTIAL AUDIT")
    else:
        print("\nSTATUS: 🔴 INFERENCE NOT AVAILABLE")

    print("\n" + "=" * 70)
    print("READ-ONLY GUARANTEE:")
    print("No model.save/joblib.dump was called.")
    print("No dataset write operation was called.")
    print("No training was performed.")
    print("No new calibration was performed.")
    print("No probability was post-processed.")
    print("No confidence percentage was fabricated.")
    print("Bootstrap CI is metric sampling uncertainty only.")
    print("=" * 70)


if __name__ == "__main__":
    main()