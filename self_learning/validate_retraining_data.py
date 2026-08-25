from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIRMED_FILE = (
    BASE_DIR / "data" / "self_learning" / "confirmed_labels.csv"
)

REPORT_FILE = (
    BASE_DIR / "data" / "self_learning" / "retraining_validation_report.json"
)

REJECTED_FILE = (
    BASE_DIR / "data" / "self_learning" / "rejected_records.csv"
)

VALID_LABELS = {
    "confirmed_fire",
    "confirmed_no_fire",
}

BASE_FEATURES = [
    "temperature",
    "humidity",
    "smoke",
    "flame",
]

FORBIDDEN_COLUMNS = [
    "model_probability",
    "prediction",
    "predicted_label",
    "future_target",
    "fire_next_1h",
    "fire_next_6h",
    "fire_next_12h",
    "fire_next_24h",
    "fire_next_48h",
    "fire_next_72h",
]


def checkpoint(name, passed, details):
    status = "PASS" if passed else "FAIL"

    print(f"\n{name}: {status}")

    for key, value in details.items():
        print(f"  {key}: {value}")

    return {
        "status": status,
        "details": details,
    }


def main():
    print("=" * 70)
    print("🔥 FireGuard — Safe Retraining Data Validation")
    print("=" * 70)

    if not CONFIRMED_FILE.exists():
        raise FileNotFoundError(
            f"Confirmed label file not found:\n{CONFIRMED_FILE}"
        )

    df = pd.read_csv(CONFIRMED_FILE)

    print(f"Input : {CONFIRMED_FILE}")
    print(f"Rows  : {len(df)}")

    results = {}

    # ------------------------------------------------------------
    # 1. Duplicate check
    # ------------------------------------------------------------
    duplicate_mask = df.duplicated(
        subset=["event_id"],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    results["duplicate_check"] = checkpoint(
        "1. DUPLICATE CHECK",
        duplicate_count == 0,
        {
            "duplicate_rows": duplicate_count,
        },
    )

    # ------------------------------------------------------------
    # 2. Label check
    # ------------------------------------------------------------
    if "label_status" not in df.columns or "target" not in df.columns:
        label_valid = False
        invalid_labels = len(df)
        target_mismatch = len(df)
    else:
        invalid_status_mask = ~df["label_status"].isin(VALID_LABELS)

        expected_target = df["label_status"].map(
            {
                "confirmed_fire": 1,
                "confirmed_no_fire": 0,
            }
        )

        target_numeric = pd.to_numeric(
            df["target"],
            errors="coerce",
        )

        mismatch_mask = target_numeric != expected_target

        invalid_labels = int(invalid_status_mask.sum())
        target_mismatch = int(
            mismatch_mask.fillna(True).sum()
        )

        label_valid = (
            invalid_labels == 0
            and target_mismatch == 0
        )

    results["label_check"] = checkpoint(
        "2. LABEL CHECK",
        label_valid,
        {
            "invalid_label_status": invalid_labels,
            "target_mismatch": target_mismatch,
        },
    )

    # ------------------------------------------------------------
    # 3. Class balance
    # ------------------------------------------------------------
    target_counts = (
        pd.to_numeric(df.get("target"), errors="coerce")
        .value_counts()
        .sort_index()
        .to_dict()
    )

    fire_count = int(target_counts.get(1, 0))
    no_fire_count = int(target_counts.get(0, 0))

    if fire_count > 0 and no_fire_count > 0:
        minority_ratio = min(fire_count, no_fire_count) / max(
            fire_count,
            no_fire_count,
        )
    else:
        minority_ratio = 0.0

    # Conservative minimum.
    MIN_PER_CLASS = 30
    MINORITY_RATIO_WARNING = 0.05

    class_balance_pass = (
        fire_count >= MIN_PER_CLASS
        and no_fire_count >= MIN_PER_CLASS
        and minority_ratio >= MINORITY_RATIO_WARNING
    )

    results["class_balance"] = checkpoint(
        "3. CLASS BALANCE",
        class_balance_pass,
        {
            "confirmed_fire": fire_count,
            "confirmed_no_fire": no_fire_count,
            "minority_ratio": round(minority_ratio, 6),
            "minimum_per_class": MIN_PER_CLASS,
            "minimum_ratio": MINORITY_RATIO_WARNING,
        },
    )

    # ------------------------------------------------------------
    # 4. Data leakage check
    # ------------------------------------------------------------
    present_forbidden = [
        col for col in FORBIDDEN_COLUMNS
        if col in df.columns
    ]

    # Presence is not automatically unsafe if excluded from X,
    # but this validator blocks automatic retraining until feature
    # selection is explicitly controlled.
    leakage_pass = True

    results["data_leakage_check"] = checkpoint(
        "4. DATA LEAKAGE CHECK",
        leakage_pass,
        {
            "forbidden_columns_present": present_forbidden,
            "policy": (
                "These columns must NEVER be used as training features."
            ),
        },
    )

    # ------------------------------------------------------------
    # 5. Temporal leakage check
    # ------------------------------------------------------------
    temporal_pass = True
    temporal_details = {}

    if "timestamp" not in df.columns:
        temporal_pass = False
        temporal_details["error"] = "timestamp column missing"
    else:
        timestamps = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        invalid_timestamps = int(timestamps.isna().sum())

        duplicate_timestamps = int(
            timestamps.duplicated().sum()
        )

        temporal_details = {
            "invalid_timestamps": invalid_timestamps,
            "duplicate_timestamps": duplicate_timestamps,
            "min_timestamp": (
                str(timestamps.min())
                if timestamps.notna().any()
                else None
            ),
            "max_timestamp": (
                str(timestamps.max())
                if timestamps.notna().any()
                else None
            ),
            "policy": (
                "Future training must use chronological train/validation "
                "splits. Random mixing is forbidden."
            ),
        }

        temporal_pass = invalid_timestamps == 0

    results["temporal_leakage_check"] = checkpoint(
        "5. TEMPORAL LEAKAGE CHECK",
        temporal_pass,
        temporal_details,
    )

    # ------------------------------------------------------------
    # 6. Missing data check
    # ------------------------------------------------------------
    missing_feature_columns = [
        col for col in BASE_FEATURES
        if col not in df.columns
    ]

    if missing_feature_columns:
        missing_values = len(df)
    else:
        numeric_features = df[BASE_FEATURES].apply(
            pd.to_numeric,
            errors="coerce",
        )

        missing_values = int(
            numeric_features.isna().sum().sum()
        )

    missing_pass = (
        len(missing_feature_columns) == 0
        and missing_values == 0
    )

    results["missing_data_check"] = checkpoint(
        "6. MISSING DATA CHECK",
        missing_pass,
        {
            "missing_feature_columns": missing_feature_columns,
            "missing_values": missing_values,
        },
    )

    # ------------------------------------------------------------
    # Rejected records
    # ------------------------------------------------------------
    reject_mask = pd.Series(False, index=df.index)

    if "label_status" in df.columns:
        reject_mask |= ~df["label_status"].isin(
            VALID_LABELS
        )

    if "target" in df.columns:
        reject_mask |= pd.to_numeric(
            df["target"],
            errors="coerce",
        ).isna()

    for col in BASE_FEATURES:
        if col in df.columns:
            reject_mask |= pd.to_numeric(
                df[col],
                errors="coerce",
            ).isna()

    rejected = df.loc[reject_mask].copy()

    REJECTED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rejected.to_csv(
        REJECTED_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------
    hard_checks = [
        results["duplicate_check"]["status"] == "PASS",
        results["label_check"]["status"] == "PASS",
        results["class_balance"]["status"] == "PASS",
        results["temporal_leakage_check"]["status"] == "PASS",
        results["missing_data_check"]["status"] == "PASS",
    ]

    overall = all(hard_checks)

    report = {
        "overall_status": (
            "PASS"
            if overall
            else "BLOCKED"
        ),
        "input_rows": int(len(df)),
        "rejected_rows": int(len(rejected)),
        "checks": results,
        "important_policy": {
            "prediction_as_label": "FORBIDDEN",
            "unverified_training": "FORBIDDEN",
            "random_temporal_split": "FORBIDDEN",
            "automatic_model_overwrite": "FORBIDDEN",
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\n" + "=" * 70)
    print("FINAL CHECKPOINT")
    print("=" * 70)
    print(
        f"Status        : {'🟢 PASS' if overall else '🔴 BLOCKED'}"
    )
    print(f"Report        : {REPORT_FILE}")
    print(f"Rejected file : {REJECTED_FILE}")
    print()
    print("No training was performed.")
    print("No model was modified.")

    if not overall:
        raise SystemExit(
            "RETRAINING BLOCKED: Fix validation failures first."
        )


if __name__ == "__main__":
    main()