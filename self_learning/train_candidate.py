from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIRMED_FILE = (
    BASE_DIR / "data" / "self_learning" / "confirmed_labels.csv"
)

VALIDATION_REPORT = (
    BASE_DIR
    / "data"
    / "self_learning"
    / "retraining_validation_report.json"
)

CANDIDATE_DIR = BASE_DIR / "saved_models" / "candidates"

FEATURES = [
    "temperature",
    "humidity",
    "smoke",
    "flame",
]

TARGET = "target"


def load_validated_data():
    if not VALIDATION_REPORT.exists():
        raise FileNotFoundError(
            "Validation report not found. "
            "Run validate_retraining_data.py first."
        )

    with open(
        VALIDATION_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    if report.get("overall_status") != "PASS":
        raise RuntimeError(
            "Retraining is blocked because data validation is not PASS."
        )

    df = pd.read_csv(CONFIRMED_FILE)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def chronological_split(df):
    if len(df) < 100:
        raise RuntimeError(
            "Too few confirmed samples for safe candidate training. "
            "Collect more confirmed data."
        )

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    validation_df = df.iloc[split_index:].copy()

    if train_df[TARGET].nunique() < 2:
        raise RuntimeError(
            "Training period does not contain both classes."
        )

    if validation_df[TARGET].nunique() < 2:
        raise RuntimeError(
            "Validation period does not contain both classes."
        )

    return train_df, validation_df


def main():
    print("=" * 70)
    print("🔥 FireGuard — Safe Candidate Training")
    print("=" * 70)

    df = load_validated_data()

    train_df, validation_df = chronological_split(df)

    X_train = train_df[FEATURES].apply(
        pd.to_numeric,
        errors="raise",
    )

    y_train = train_df[TARGET].astype(int)

    X_validation = validation_df[FEATURES].apply(
        pd.to_numeric,
        errors="raise",
    )

    y_validation = validation_df[TARGET].astype(int)

    print(f"Total samples      : {len(df)}")
    print(f"Train samples      : {len(train_df)}")
    print(f"Validation samples : {len(validation_df)}")
    print(
        f"Train period       : "
        f"{train_df['timestamp'].min()} -> "
        f"{train_df['timestamp'].max()}"
    )
    print(
        f"Validation period  : "
        f"{validation_df['timestamp'].min()} -> "
        f"{validation_df['timestamp'].max()}"
    )

    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Calibration is performed only inside candidate training.
    # This does not alter any active model.
    try:
        model = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=5,
        )
    except TypeError:
        model = CalibratedClassifierCV(
            base_estimator=base_model,
            method="sigmoid",
            cv=5,
        )

    print("\nTraining candidate...")
    model.fit(X_train, y_train)

    candidate_id = (
        "candidate_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        CANDIDATE_DIR
        / f"fireguard_{candidate_id}.joblib"
    )

    metadata_path = (
        CANDIDATE_DIR
        / f"fireguard_{candidate_id}.json"
    )

    artifact = {
        "model": model,
        "features": FEATURES,
        "feature_names": FEATURES,
        "target": TARGET,
        "version": candidate_id,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "status": "CANDIDATE_ONLY",
        "training_policy": (
            "confirmed labels only; "
            "chronological validation; "
            "prediction is never a label"
        ),
    }

    joblib.dump(
        artifact,
        model_path,
    )

    metadata = {
        "candidate_id": candidate_id,
        "model_path": str(model_path),
        "features": FEATURES,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
        "validation_start": str(
            validation_df["timestamp"].min()
        ),
        "validation_end": str(
            validation_df["timestamp"].max()
        ),
        "status": "CANDIDATE_ONLY",
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("CANDIDATE CREATED")
    print("=" * 70)
    print(f"Model    : {model_path}")
    print(f"Metadata : {metadata_path}")
    print()
    print("IMPORTANT:")
    print("Candidate is NOT active.")
    print("Candidate must pass compare_models.py.")
    print("Active models were NOT modified.")


if __name__ == "__main__":
    main()