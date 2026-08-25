from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.db import FireGuardDatabase


# ============================================================
# TEST VALUES
# ============================================================

TIMESTAMP = "2026-08-18 14:30:00"

MODEL_VERSION = "stage11-test-model-v1"
FEATURE_VERSION = "v4"

EVENT_ID = "stage11-" + uuid4().hex[:16]

RUN_ID = "stage11-run-" + uuid4().hex[:12]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("🔥 FireGuard — Stage 11: Database Test")
    print("=" * 70)

    db = FireGuardDatabase()

    print()
    print(f"Database: {db.db_path}")

    try:

        # ----------------------------------------------------
        # 1. SENSOR READING
        # ----------------------------------------------------

        sensor_id = db.add_sensor_reading(
            timestamp=TIMESTAMP,
            temperature=25.0,
            humidity=45.0,
            smoke=120.0,
            flame=0.0,
            source="stage11_test",
        )

        assert sensor_id is not None

        print("PASS — sensor_reading")

        # ----------------------------------------------------
        # 2. MODEL
        # ----------------------------------------------------

        model_id = db.add_model(
            model_version=MODEL_VERSION,
            model_name="FireGuard Test Model",
            experiment="stage11_database_test",
            feature_version=FEATURE_VERSION,
            horizon=24,
            status="test",
            path="stage11_test_model.joblib",
            notes="Stage 11 database test record",
        )

        assert model_id is not None

        print("PASS — model")

        # ----------------------------------------------------
        # 3. PREDICTION
        # ----------------------------------------------------

        prediction_id = db.add_prediction(
            timestamp=TIMESTAMP,
            model_version=MODEL_VERSION,
            feature_version=FEATURE_VERSION,
            probability=0.73,
            uncertainty=0.08,
            horizon=24,
        )

        assert prediction_id is not None

        print("PASS — prediction")

        # ----------------------------------------------------
        # 4. FIRE EVENT
        # ----------------------------------------------------

        event_id = db.add_fire_event(
            event_id=EVENT_ID,
            timestamp=TIMESTAMP,
            event_type="test_event",
            status="open",
            description="Stage 11 test event",
        )

        assert event_id is not None

        print("PASS — fire_event")

        # ----------------------------------------------------
        # 5. VERIFICATION
        # ----------------------------------------------------

        verification_id = db.add_verification(
            event_id=EVENT_ID,
            label="confirmed_no_fire",
            verified_at=TIMESTAMP,
            verified_by="stage11_test_user",
            source="human",
        )

        assert verification_id is not None

        print("PASS — verification")

        # ----------------------------------------------------
        # 6. ALERT
        # ----------------------------------------------------

        alert_id = db.add_alert(
            event_id=EVENT_ID,
            alert_level="WARNING",
            reason="Stage 11 test alert",
            status="active",
        )

        assert alert_id is not None

        print("PASS — alert")

        # ----------------------------------------------------
        # 7. TRAINING RUN
        # ----------------------------------------------------

        training_id = db.add_training_run(
            run_id=RUN_ID,
            model_version=MODEL_VERSION,
            started_at=TIMESTAMP,
            completed_at=TIMESTAMP,
            dataset_version="stage11-test-dataset-v1",
            samples=100,
            validation_score=0.91,
            status="test",
            notes="No actual training performed.",
        )

        assert training_id is not None

        print("PASS — training_run")

        # ----------------------------------------------------
        # 8. EXTERNAL OBSERVATION
        # ----------------------------------------------------

        observation_id = db.add_external_observation(
            timestamp=TIMESTAMP,
            source="stage11_test_source",
            observation_type="test_observation",
            value="no_fire",
            confidence=0.95,
        )

        assert observation_id is not None

        print("PASS — external_observation")

        # ----------------------------------------------------
        # 9. VERIFY COUNTS
        # ----------------------------------------------------

        tables = [
            "sensor_reading",
            "prediction",
            "fire_event",
            "verification",
            "alert",
            "model",
            "training_run",
            "external_observation",
        ]

        print()
        print("=" * 70)
        print("RECORD COUNTS")
        print("=" * 70)

        for table in tables:

            count = db.count_records(table)

            print(
                f"{table:25s}: {count}"
            )

            assert count >= 1

        # ----------------------------------------------------
        # 10. READ TEST RECORDS
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("RECENT TEST RECORDS")
        print("=" * 70)

        prediction_rows = db.get_recent(
            "prediction",
            3,
        )

        for row in prediction_rows:

            print(
                f"Prediction: "
                f"id={row['id']} "
                f"model={row['model_version']} "
                f"probability={row['probability']} "
                f"uncertainty={row['uncertainty']} "
                f"horizon={row['horizon']}"
            )

        event_rows = db.get_recent(
            "fire_event",
            3,
        )

        for row in event_rows:

            print(
                f"Event: "
                f"id={row['event_id']} "
                f"status={row['status']}"
            )

        verification_rows = db.get_recent(
            "verification",
            3,
        )

        for row in verification_rows:

            print(
                f"Verification: "
                f"event={row['event_id']} "
                f"label={row['label']} "
                f"verified_by={row['verified_by']} "
                f"source={row['source']}"
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("CHECKPOINT")
        print("=" * 70)

        print("Database initialization : PASS")
        print("sensor_reading         : PASS")
        print("prediction             : PASS")
        print("fire_event             : PASS")
        print("verification           : PASS")
        print("alert                  : PASS")
        print("model                  : PASS")
        print("training_run           : PASS")
        print("external_observation  : PASS")
        print("Record retrieval       : PASS")

        print()
        print("Models modified        : NO")
        print("Dataset modified       : NO")
        print("Training performed     : NO")
        print("Calibration performed  : NO")

        print()
        print("STATUS: 🟢 STAGE 11 DATABASE READY")

    except Exception as exc:

        print()
        print("=" * 70)
        print("❌ STAGE 11 DATABASE TEST FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        print()
        print(
            "No model retraining was performed."
        )

        raise


if __name__ == "__main__":
    main()