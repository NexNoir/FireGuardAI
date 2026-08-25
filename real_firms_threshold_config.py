# -*- coding: utf-8 -*-
"""
FireGuard — Real FIRMS Threshold Configuration V1
==================================================

Purpose:
    Store the validated decision thresholds for the Real FIRMS V1
    forecasting models.

IMPORTANT:
    - No model retraining.
    - No model modification.
    - No dataset modification.
    - No synthetic data.
    - No fabricated labels.
    - Thresholds are kept separately from .joblib model files.

Validated on:
    Final test period: 2023-2025

Selected criterion:
    Best F1 score from the Real FIRMS Threshold Audit V1.

Thresholds:
    24H -> 0.35
    48H -> 0.35
    72H -> 0.30
"""

from __future__ import annotations

from pathlib import Path
import json


# ======================================================================
# CONFIG
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    BASE_DIR
    / "saved_models"
    / "real_firms_v1"
)

CONFIG_DIR = (
    BASE_DIR
    / "data"
    / "retraining"
)

CONFIG_FILE = (
    CONFIG_DIR
    / "real_firms_threshold_config_v1.json"
)


# ======================================================================
# VALIDATED THRESHOLDS
# ======================================================================

THRESHOLDS = {
    24: 0.35,
    48: 0.35,
    72: 0.30,
}


# ======================================================================
# MODEL FILES
# ======================================================================

MODEL_FILES = {
    24: (
        MODEL_DIR
        / "fireguard_real_firms_sensor_only_24h_v1.joblib"
    ),

    48: (
        MODEL_DIR
        / "fireguard_real_firms_sensor_only_48h_v1.joblib"
    ),

    72: (
        MODEL_DIR
        / "fireguard_real_firms_sensor_only_72h_v1.joblib"
    ),
}


# ======================================================================
# AUDIT RESULTS
# ======================================================================

VALIDATION_REFERENCE = {
    24: {
        "threshold": 0.35,
        "accuracy": 0.5310,
        "precision": 0.4192,
        "recall": 0.8223,
        "f1": 0.5553,
    },

    48: {
        "threshold": 0.35,
        "accuracy": 0.5571,
        "precision": 0.4788,
        "recall": 0.8852,
        "f1": 0.6214,
    },

    72: {
        "threshold": 0.30,
        "accuracy": 0.5298,
        "precision": 0.4817,
        "recall": 0.9294,
        "f1": 0.6345,
    },
}


# ======================================================================
# PRINT HELPERS
# ======================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ======================================================================
# VALIDATE CONFIGURATION
# ======================================================================

def validate_configuration() -> None:

    section("VALIDATING THRESHOLD CONFIGURATION")

    expected_horizons = {24, 48, 72}

    if set(THRESHOLDS.keys()) != expected_horizons:
        raise ValueError(
            "Threshold horizon configuration is invalid."
        )

    for horizon, threshold in THRESHOLDS.items():

        if not isinstance(threshold, (int, float)):
            raise TypeError(
                f"Invalid threshold type for {horizon}h."
            )

        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(
                f"Threshold for {horizon}h must be between 0 and 1."
            )

    print("24H threshold : 0.35")
    print("48H threshold : 0.35")
    print("72H threshold : 0.30")

    print()
    print("Threshold configuration: PASS")


# ======================================================================
# CHECK MODELS
# ======================================================================

def check_models() -> None:

    section("CHECKING REAL FIRMS MODELS")

    for horizon, model_path in MODEL_FILES.items():

        exists = model_path.exists()

        print(
            f"{horizon:>2}H model : "
            f"{'FOUND' if exists else 'NOT FOUND'}"
        )

    print()
    print("IMPORTANT:")
    print("Model files are NOT modified.")
    print("Model files are NOT retrained.")


# ======================================================================
# BUILD CONFIGURATION
# ======================================================================

def build_configuration() -> dict:

    return {
        "project": "FireGuard",
        "configuration": "Real FIRMS Threshold Configuration V1",

        "purpose": (
            "Decision thresholds for existing Real FIRMS V1 "
            "forecast models."
        ),

        "training": False,
        "retraining": False,
        "model_modified": False,
        "dataset_modified": False,
        "synthetic_data": False,
        "fabricated_labels": False,

        "test_period": "2023-2025",

        "selection_method": (
            "Best F1 score from Real FIRMS Threshold Audit V1"
        ),

        "thresholds": {
            "24h": THRESHOLDS[24],
            "48h": THRESHOLDS[48],
            "72h": THRESHOLDS[72],
        },

        "validation_reference": {
            "24h": VALIDATION_REFERENCE[24],
            "48h": VALIDATION_REFERENCE[48],
            "72h": VALIDATION_REFERENCE[72],
        },

        "model_directory": str(MODEL_DIR),

        "model_files": {
            "24h": str(MODEL_FILES[24]),
            "48h": str(MODEL_FILES[48]),
            "72h": str(MODEL_FILES[72]),
        },
    }


# ======================================================================
# SAVE CONFIGURATION
# ======================================================================

def save_configuration(config: dict) -> None:

    section("SAVING THRESHOLD CONFIGURATION")

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"Configuration : {CONFIG_FILE}"
    )


# ======================================================================
# FINAL CHECKPOINT
# ======================================================================

def final_checkpoint() -> None:

    section("FINAL CHECKPOINT")

    print("24H threshold : 0.35")
    print("48H threshold : 0.35")
    print("72H threshold : 0.30")

    print()
    print("Models modified       : NO")
    print("Retraining performed  : NO")
    print("Original dataset      : NO")
    print("Synthetic data        : NO")
    print("Fabricated labels     : NO")

    print()
    print("STATUS: 🟢 THRESHOLD CONFIGURATION READY")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    section(
        "🔥 FIREGUARD — REAL FIRMS THRESHOLD CONFIGURATION V1"
    )

    print("Threshold configuration only.")
    print("No retraining.")
    print("No model modification.")
    print("No dataset modification.")

    validate_configuration()

    check_models()

    config = build_configuration()

    save_configuration(config)

    final_checkpoint()


# ======================================================================
# ENTRY POINT
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