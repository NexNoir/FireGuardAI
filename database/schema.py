from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sensor_reading (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    temperature REAL,
    humidity REAL,
    smoke REAL,
    flame REAL,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    probability REAL NOT NULL,
    uncertainty REAL,
    horizon INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fire_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    event_type TEXT,
    status TEXT DEFAULT 'open',
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS verification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    label TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    verified_by TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(event_id)
        REFERENCES fire_event(event_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS alert (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    alert_level TEXT NOT NULL,
    reason TEXT,
    status TEXT DEFAULT 'active',
    acknowledged_at TEXT,
    acknowledged_by TEXT,
    resolved_at TEXT,
    resolved_by TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(event_id)
        REFERENCES fire_event(event_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL UNIQUE,
    model_name TEXT,
    experiment TEXT,
    feature_version TEXT,
    horizon INTEGER,
    status TEXT,
    path TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    model_version TEXT,
    started_at TEXT,
    completed_at TEXT,
    dataset_version TEXT,
    samples INTEGER,
    validation_score REAL,
    status TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS external_observation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    observation_type TEXT,
    value TEXT,
    confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sensor_timestamp
    ON sensor_reading(timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_timestamp
    ON prediction(timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_model
    ON prediction(model_version);

CREATE INDEX IF NOT EXISTS idx_event_timestamp
    ON fire_event(timestamp);

CREATE INDEX IF NOT EXISTS idx_verification_event
    ON verification(event_id);

CREATE INDEX IF NOT EXISTS idx_alert_event
    ON alert(event_id);
"""


TABLES = [
    "sensor_reading",
    "prediction",
    "fire_event",
    "verification",
    "alert",
    "model",
    "training_run",
    "external_observation",
]