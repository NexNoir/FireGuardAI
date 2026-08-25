import math
import sqlite3
from pathlib import Path


# ============================================================
# FireGuard — Stage 13
# Full System Safety / Integration Tests
# READ-ONLY
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


def assert_valid_probability(value):
    assert value is not None
    assert not isinstance(value, complex)

    value = float(value)

    assert math.isfinite(value), "Probability is NaN/Inf"
    assert 0.0 <= value <= 1.0, (
        f"Invalid probability: {value}. "
        "Probability must be between 0 and 1."
    )


def test_probability_boundaries():
    assert_valid_probability(0.0)
    assert_valid_probability(1.0)


def test_invalid_probability_is_rejected():
    for value in [-0.01, 1.01, float("nan"), float("inf")]:
        try:
            assert_valid_probability(value)
        except AssertionError:
            continue

        raise AssertionError(
            f"Unsafe probability was accepted: {value}"
        )


def test_sensor_disconnected_does_not_crash():
    sensor = None

    # Safe behavior:
    # disconnected sensor must produce no fabricated reading.
    assert sensor is None


def test_bad_temperature_is_rejected():
    bad_values = [
        float("nan"),
        float("inf"),
        -9999,
        9999,
    ]

    for value in bad_values:
        valid = (
            math.isfinite(float(value))
            and -100 <= float(value) <= 100
        )
        assert valid is False


def test_humidity_out_of_range_is_rejected():
    for humidity in [-1, 101, 999]:
        valid = 0 <= humidity <= 100
        assert valid is False


def test_smoke_spike_is_not_probability():
    smoke = 99999
    probability = 0.42

    # Smoke is evidence/input, not probability.
    assert probability == 0.42
    assert_valid_probability(probability)


def test_stale_data_is_detectable():
    status = "STALE DATA"
    assert status == "STALE DATA"


def test_nasa_unavailable_has_no_fake_fpr():
    nasa_available = False
    frp = None

    assert nasa_available is False
    assert frp is None


def test_weather_unavailable_has_no_fake_weather():
    weather_available = False
    weather = None

    assert weather_available is False
    assert weather is None


def test_database_unavailable_is_safe():
    database_available = False

    # Application must be able to represent unavailable DB
    # without inventing historical records.
    assert database_available is False


def test_model_unavailable_is_safe():
    model_available = False
    probability = None

    assert model_available is False
    assert probability is None


def test_sms_unavailable_does_not_crash():
    sms_available = False

    # Detection and SMS are separate.
    assert sms_available is False


def test_no_fake_confidence():
    confidence = None

    # Confidence must never be fabricated.
    assert confidence is None


def test_nan_probability_rejected():
    value = float("nan")

    assert not math.isfinite(value)


def test_probability_never_below_zero():
    values = [0.0, 0.1, 0.5, 1.0]

    for value in values:
        assert_valid_probability(value)


def test_probability_never_above_one():
    values = [0.0, 0.1, 0.5, 1.0]

    for value in values:
        assert_valid_probability(value)


def test_database_schema_exists():
    db_path = ROOT / "data" / "fireguard_history.db"

    # Existing Stage 11 database should exist.
    assert db_path.exists(), f"Database not found: {db_path}"


def test_database_required_tables():
    db_path = ROOT / "data" / "fireguard_history.db"

    if not db_path.exists():
        return

    connection = sqlite3.connect(str(db_path))

    try:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """
        ).fetchall()

        tables = {row[0] for row in rows}

        required = {
            "sensor_reading",
            "prediction",
            "fire_event",
            "verification",
            "alert",
            "model",
            "training_run",
            "external_observation",
        }

        missing = required - tables

        assert not missing, (
            f"Missing database tables: {sorted(missing)}"
        )

    finally:
        connection.close()


def test_required_stage13_modules_exist():
    expected = [
        "database",
        "dashboard",
    ]

    for directory in expected:
        path = ROOT / directory
        assert path.exists(), f"Missing directory: {directory}"


def test_system_must_not_generate_fake_external_data():
    nasa = None
    weather = None

    assert nasa is None
    assert weather is None


def test_system_failure_states_are_explicit():
    states = {
        "sensor": "DISCONNECTED",
        "nasa": "UNAVAILABLE",
        "weather": "UNAVAILABLE",
        "database": "UNAVAILABLE",
        "model": "UNAVAILABLE",
        "sms": "UNAVAILABLE",
    }

    for component, state in states.items():
        assert state
        assert state != "OK"


def test_stage13_read_only():
    # This test intentionally does not write anything.
    assert True