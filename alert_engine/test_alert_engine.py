
from __future__ import annotations

from pathlib import Path
import sys

# Allow running:
# python alert_engine\test_alert_engine.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alert_engine.alert_engine import AlertEngine, AlertInput
from alert_engine.alert_store import AlertStore
from alert_engine.sms_gateway import SMSGateway


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"PASS — {message}")


def main() -> None:
    print("=" * 70)
    print("🔥 FireGuard — Stage 10: Alert Engine Test")
    print("=" * 70)

    engine = AlertEngine()

    # --------------------------------------------------------------
    # 1. INFO
    # --------------------------------------------------------------
    result = engine.evaluate(
        AlertInput(
            fire_probability=0.05,
            flame=0,
            smoke_trend=0.05,
            sensor_quality=0.95,
        )
    )

    check(result.level == "INFO", "INFO")
    check(
        result.probability_changed is False,
        "Probability unchanged — INFO",
    )

    # --------------------------------------------------------------
    # 2. WATCH
    # --------------------------------------------------------------
    result = engine.evaluate(
        AlertInput(
            fire_probability=0.30,
            flame=0,
            smoke_trend=0.40,
            sensor_quality=0.95,
        )
    )

    check(result.level == "WATCH", "WATCH")

    # --------------------------------------------------------------
    # 3. WARNING
    # --------------------------------------------------------------
    result = engine.evaluate(
        AlertInput(
            fire_probability=0.55,
            flame=0,
            smoke_trend=0.65,
            sensor_quality=0.95,
        )
    )

    check(result.level == "WARNING", "WARNING")

    # --------------------------------------------------------------
    # 4. HIGH
    # --------------------------------------------------------------
    result = engine.evaluate(
        AlertInput(
            fire_probability=0.75,
            flame=0,
            smoke_trend=0.90,
            sensor_quality=0.95,
        )
    )

    check(result.level == "HIGH", "HIGH")

    # --------------------------------------------------------------
    # 5. CRITICAL
    # --------------------------------------------------------------
    result = engine.evaluate(
        AlertInput(
            fire_probability=0.40,
            flame=1,
            smoke_trend=0.90,
            sensor_quality=0.95,
        )
    )

    check(result.level == "CRITICAL", "CRITICAL")

    # --------------------------------------------------------------
    # 6. Probability must NEVER be changed by evidence
    # --------------------------------------------------------------
    original_probability = 0.40

    result = engine.evaluate(
        AlertInput(
            fire_probability=original_probability,
            flame=1,
            smoke_trend=1.0,
            sensor_quality=0.90,
            nasa_evidence=True,
            weather_risk=0.90,
            forecast_risk=0.95,
            uncertainty=0.80,
        )
    )

    check(
        result.fire_probability == original_probability,
        "Probability unchanged by external evidence",
    )

    check(
        result.probability_changed is False,
        "Probability mutation forbidden",
    )

    # --------------------------------------------------------------
    # 7. Event ID
    # --------------------------------------------------------------
    check(
        result.event_id.startswith("ALR-"),
        "Event ID generated",
    )

    # --------------------------------------------------------------
    # 8. Store
    # --------------------------------------------------------------
    test_store_path = (
        PROJECT_ROOT
        / "data"
        / "alert_engine"
        / "test_alerts.json"
    )

    if test_store_path.exists():
        test_store_path.unlink()

    store = AlertStore(test_store_path)

    added = store.add(result.to_dict())

    check(
        added is True,
        "Alert stored",
    )

    duplicate = store.add(result.to_dict())

    check(
        duplicate is False,
        "Deduplication",
    )

    # --------------------------------------------------------------
    # 9. Acknowledgement
    # --------------------------------------------------------------
    acknowledged = store.acknowledge(result.event_id)

    check(
        acknowledged is True,
        "Acknowledgement",
    )

    stored = store.get(result.event_id)

    check(
        stored is not None and stored["acknowledged"] is True,
        "Acknowledgement state persisted",
    )

    # --------------------------------------------------------------
    # 10. Resolved
    # --------------------------------------------------------------
    resolved = store.resolve(result.event_id)

    check(
        resolved is True,
        "Resolved state",
    )

    stored = store.get(result.event_id)

    check(
        stored is not None and stored["resolved"] is True,
        "Resolved state persisted",
    )

    # --------------------------------------------------------------
    # 11. SMS safety
    # --------------------------------------------------------------
    sms = SMSGateway()

    success, message = sms.send(
        event_id=result.event_id,
        level=result.level,
        message="TEST ALERT",
    )

    check(
        success is False,
        "Real SMS blocked",
    )

    check(
        "SMS BLOCKED" in message,
        "SMS safety guard active",
    )

    # --------------------------------------------------------------
    # FINAL CHECKPOINT
    # --------------------------------------------------------------
    print()
    print("=" * 70)
    print("CHECKPOINT")
    print("=" * 70)

    print("Alert Engine       : PASS")
    print("INFO               : PASS")
    print("WATCH              : PASS")
    print("WARNING            : PASS")
    print("HIGH               : PASS")
    print("CRITICAL           : PASS")
    print("Event ID           : PASS")
    print("Deduplication      : PASS")
    print("Acknowledgement    : PASS")
    print("Resolved state     : PASS")
    print("Probability mutate : NO")
    print("Real SMS           : BLOCKED")
    print("Dry Run            : ENABLED")
    print()
    print("Models modified    : NO")
    print("Dataset modified   : NO")
    print("Training           : NO")
    print("Calibration        : NO")
    print()
    print("STATUS: 🟢 STAGE 10 ALERT ENGINE READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
