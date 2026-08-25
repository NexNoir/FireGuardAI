import sqlite3


REQUIRED_TABLES = {
    "sensor_reading",
    "prediction",
    "fire_event",
    "verification",
    "alert",
    "model",
    "training_run",
    "external_observation",
}


def create_test_database():
    connection = sqlite3.connect(":memory:")

    connection.executescript("""
        CREATE TABLE sensor_reading (
            id INTEGER PRIMARY KEY,
            timestamp TEXT
        );

        CREATE TABLE prediction (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            model_version TEXT,
            feature_version TEXT,
            probability REAL,
            uncertainty REAL,
            horizon REAL
        );

        CREATE TABLE fire_event (
            id TEXT PRIMARY KEY,
            status TEXT
        );

        CREATE TABLE verification (
            id INTEGER PRIMARY KEY,
            event_id TEXT,
            label TEXT,
            verified_at TEXT,
            verified_by TEXT,
            source TEXT
        );

        CREATE TABLE alert (
            id INTEGER PRIMARY KEY,
            event_id TEXT,
            level TEXT
        );

        CREATE TABLE model (
            id INTEGER PRIMARY KEY,
            model_version TEXT
        );

        CREATE TABLE training_run (
            id INTEGER PRIMARY KEY,
            started_at TEXT
        );

        CREATE TABLE external_observation (
            id INTEGER PRIMARY KEY,
            source TEXT
        );
    """)

    return connection


def test_database_schema_exists():
    db = create_test_database()

    assert db is not None

    db.close()


def test_required_tables():
    db = create_test_database()

    rows = db.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
    """).fetchall()

    tables = {row[0] for row in rows}

    assert REQUIRED_TABLES.issubset(tables)

    db.close()


def test_prediction_history_fields():
    db = create_test_database()

    db.execute("""
        INSERT INTO prediction
        (timestamp, model_version, feature_version,
         probability, uncertainty, horizon)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "2026-08-18T14:30:00",
        "test-model-v1",
        "features-v1",
        0.73,
        0.08,
        24,
    ))

    row = db.execute("""
        SELECT model_version, feature_version,
               probability, uncertainty, horizon
        FROM prediction
    """).fetchone()

    assert row[0] == "test-model-v1"
    assert row[1] == "features-v1"
    assert row[2] == 0.73
    assert row[3] == 0.08
    assert row[4] == 24

    db.close()