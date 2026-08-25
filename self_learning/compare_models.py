from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIRMED_FILE = (
    BASE_DIR / "data" / "self_learning" / "confirmed_labels.csv"
)

ACTIVE_DIR = BASE_DIR / "saved_models" / "active"
CANDIDATE_DIR = BASE_DIR / "saved_models" / "candidates"

COMPARISON_FILE = (
    BASE_DIR / "data" / "self_learning" / "model_comparison.json"
)


def latest_file(directory, pattern):
    files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def get_probability(model_or_artifact, X):
    if isinstance(model_or_artifact, dict):
        model = model_or_artifact.get("model")
    else:
        model = model_or_artifact

    if model is None:
        raise ValueError("Model not found in artifact.")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support predict_proba."
        )

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Binary classification probabilities expected."
        )

    return probabilities[:, 1]


def metrics(y_true, probabilities):
    result = {
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.clip(probabilities, 1e-6, 1 - 1e-6),
            )
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result


def candidate_is_better(current, candidate):
    """
    Conservative promotion policy:

    Required:
    - Candidate Brier must improve.
    - Candidate Log Loss must not worsen.
    - Candidate ROC-AUC must not worsen if available.
    - Candidate PR-AUC must not worsen if available.
    """

    checks = {}

    checks["brier_improved"] = (
        candidate["brier"] < current["brier"]
    )

    checks["log_loss_not_worse"] = (
        candidate["log_loss"] <= current["log_loss"]
    )

    if (
        current["roc_auc"] is not None
        and candidate["roc_auc"] is not None
    ):
        checks["roc_auc_not_worse"] = (
            candidate["roc_auc"] >= current["roc_auc"]
        )

    if (
        current["pr_auc"] is not None
        and candidate["pr_auc"] is not None
    ):
        checks["pr_auc_not_worse"] = (
            candidate["pr_auc"] >= current["pr_auc"]
        )

    return all(checks.values()), checks


def main():
    print("=" * 70)
    print("🔥 FireGuard — Candidate vs Active Model Comparison")
    print("=" * 70)

    active_path = latest_file(
        ACTIVE_DIR,
        "*.joblib",
    )

    candidate_path = latest_file(
        CANDIDATE_DIR,
        "*.joblib",
    )

    if active_path is None:
        raise FileNotFoundError(
            f"No active model found in {ACTIVE_DIR}"
        )

    if candidate_path is None:
        raise FileNotFoundError(
            f"No candidate model found in {CANDIDATE_DIR}"
        )

    df = pd.read_csv(CONFIRMED_FILE)

    if len(df) < 100:
        raise RuntimeError(
            "Too few confirmed records for comparison."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Same chronological validation period:
    split_index = int(len(df) * 0.80)
    validation_df = df.iloc[split_index:].copy()

    active_artifact = joblib.load(active_path)
    candidate_artifact = joblib.load(candidate_path)

    active_features = (
        active_artifact.get("features")
        or active_artifact.get("feature_names")
    )

    candidate_features = (
        candidate_artifact.get("features")
        or candidate_artifact.get("feature_names")
    )

    if active_features is None:
        raise ValueError(
            "Active model features unavailable."
        )

    if candidate_features is None:
        raise ValueError(
            "Candidate model features unavailable."
        )

    # Both models must be evaluated honestly.
    active_missing = [
        col for col in active_features
        if col not in validation_df.columns
    ]

    candidate_missing = [
        col for col in candidate_features
        if col not in validation_df.columns
    ]

    if active_missing:
        raise ValueError(
            f"Active model missing validation features: "
            f"{active_missing}"
        )

    if candidate_missing:
        raise ValueError(
            f"Candidate model missing validation features: "
            f"{candidate_missing}"
        )

    X_active = validation_df[active_features].apply(
        pd.to_numeric,
        errors="raise",
    )

    X_candidate = validation_df[candidate_features].apply(
        pd.to_numeric,
        errors="raise",
    )

    y_true = validation_df["target"].astype(int)

    active_prob = get_probability(
        active_artifact,
        X_active,
    )

    candidate_prob = get_probability(
        candidate_artifact,
        X_candidate,
    )

    active_metrics = metrics(
        y_true,
        active_prob,
    )

    candidate_metrics = metrics(
        y_true,
        candidate_prob,
    )

    better, checks = candidate_is_better(
        active_metrics,
        candidate_metrics,
    )

    print(f"\nActive model    : {active_path}")
    print(f"Candidate model : {candidate_path}")
    print(f"Validation rows : {len(validation_df)}")

    print("\nCURRENT MODEL METRICS")
    for key, value in active_metrics.items():
        print(f"  {key}: {value}")

    print("\nCANDIDATE MODEL METRICS")
    for key, value in candidate_metrics.items():
        print(f"  {key}: {value}")

    print("\nPROMOTION CHECKS")
    for key, value in checks.items():
        print(
            f"  {key}: {'PASS' if value else 'FAIL'}"
        )

    result = {
        "active_model": str(active_path),
        "candidate_model": str(candidate_path),
        "validation_rows": int(len(validation_df)),
        "active_metrics": active_metrics,
        "candidate_metrics": candidate_metrics,
        "checks": checks,
        "promotion_status": (
            "APPROVED_FOR_MANUAL_PROMOTION"
            if better
            else "REJECTED"
        ),
        "automatic_promotion": False,
    }

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        COMPARISON_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)

    if better:
        print(
            "🟢 CANDIDATE APPROVED FOR MANUAL PROMOTION"
        )
        print(
            "Run promote_model.py only after reviewing this result."
        )
    else:
        print("🔴 CANDIDATE REJECTED")
        print("Active model remains unchanged.")

    print("=" * 70)


if __name__ == "__main__":
    main()