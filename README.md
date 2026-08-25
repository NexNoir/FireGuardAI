# 🔥 FireGuard 

## Production-Oriented Wildfire Monitoring, Detection, Forecasting, and Alerting Platform

**FireGuard** is a Python-based wildfire monitoring and prediction platform designed to combine **live ESP32 sensor telemetry**, **NASA FIRMS satellite-derived fire observations**, **forecasting models**, **historical fire evidence**, **weather information**, **alerts**, and a **Streamlit operational dashboard** into a single modular system.

The project is designed around a clear separation between:

* live sensor monitoring,
* production ML inference,
* forecast persistence,
* alert evaluation,
* SMS delivery,
* dashboard presentation,
* and historical/remote-sensing evidence.

The current production integration includes a validated **Real FIRMS V1** inference pipeline with separate models for **24-hour, 48-hour, and 72-hour** wildfire-risk prediction.

---

## Project Overview

FireGuard consists of two major operational data paths.

### 1. Live Sensor Monitoring

The ESP32 sensor pipeline provides real-time environmental readings including:

* Temperature
* Humidity
* Smoke
* Flame

These readings are persisted in the FireGuard SQLite database under the `sensor_reading` table and are consumed by dashboard monitoring components.

### 2. Real FIRMS Forecasting

The Real FIRMS production pipeline uses NASA FIRMS-derived features and three approved production models:

* 24-hour forecast
* 48-hour forecast
* 72-hour forecast

The resulting probabilities and horizon-specific predictions are persisted in the `prediction` table and exposed to the Streamlit dashboard.

> **Important:** The current Real FIRMS production model schema contains FIRMS-derived and temporal/encoded features. ESP32 fields such as `temperature`, `humidity`, `smoke`, and `flame` are currently stored and displayed as live sensor evidence but are not part of the current 15-feature Real FIRMS model input schema.

---

# Architecture

```text
                         FIREGUARD v2.0
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
      ESP32 / Live Sensors                 NASA FIRMS
             │                                   │
             ▼                                   ▼
      sensor_reading DB                 FIRMS Feature Pipeline
             │                                   │
             │                                   ▼
             │                        Real FIRMS Production
             │                              Service
             │                                   │
             │                    ┌──────────────┼──────────────┐
             │                    │              │              │
             │                    ▼              ▼              ▼
             │                  24H            48H            72H
             │                 Model          Model          Model
             │                    │              │              │
             │                    └──────────────┼──────────────┘
             │                                   │
             │                                   ▼
             │                           prediction DB
             │                                   │
             └───────────────────┬───────────────┘
                                 │
                                 ▼
                         Streamlit Dashboard
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
   Live Monitoring          Forecast                  Alerts
        │                        │                        │
        ▼                        ▼                        ▼
 Sensor Evidence         ML Probabilities         Alert Engine
                                                     │
                                                     ▼
                                                  SMS Service
                                                     │
                                                     ▼
                                                  Kavenegar
```

---

# Core Technology Stack

| Component        | Technology                                 |
| ---------------- | ------------------------------------------ |
| Language         | Python                                     |
| Dashboard        | Streamlit                                  |
| Data Processing  | Pandas / NumPy                             |
| ML Model Runtime | scikit-learn / joblib                      |
| Database         | SQLite                                     |
| Remote Fire Data | NASA FIRMS                                 |
| Live Sensors     | ESP32                                      |
| Visualization    | Streamlit charts / Plotly where used       |
| SMS Provider     | Kavenegar                                  |
| Testing          | pytest + dedicated integration/smoke tests |

---

# Project Structure

A simplified production-oriented structure is:

```text
fireguard_v2.0/
│
├── app.py
├── config.py
├── production_runtime.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dashboard/
│   ├── app.py
│   ├── data.py
│   ├── components.py
│   ├── styles.py
│   └── pages/
│       ├── overview.py
│       ├── live.py
│       ├── forecast.py
│       ├── seasonal_forecast.py
│       ├── analytics.py
│       ├── events.py
│       ├── nasa.py
│       ├── weather.py
│       ├── alerts.py
│       ├── verification.py
│       ├── models.py
│       ├── health.py
│       ├── database.py
│       └── hyrcanian.py
│
├── database/
│   ├── db.py
│   ├── schema.py
│   └── test_database.py
│
├── sensors/
│   ├── live_sensor.py
│   ├── esp32_reader.py
│   └── data_validator.py
│
├── nasa/
│   ├── firms_client.py
│   └── live_firms.py
│
├── weather/
│   ├── open_meteo_client.py
│   ├── live_weather.py
│   └── ...
│
├── alerts/
│   ├── alert_service.py
│   ├── sms_service.py
│   ├── prediction_alert_bridge.py
│   └── ...
│
├── alert_engine/
│   ├── alert_engine.py
│   ├── alert_store.py
│   └── sms_gateway.py
│
├── detection/
│   └── rule_engine.py
│
├── features/
│   └── feature_builder.py
│
├── models/
│   └── predictor.py
│
├── forecast/
│   ├── real_forecast.py
│   ├── real_forecast_v2.py
│   ├── real_forecast_v3.py
│   └── real_forecast_v4.py
│
├── saved_models/
│   ├── active/
│   ├── archive/
│   ├── candidates/
│   └── real_firms_v1/
│       ├── fireguard_real_firms_sensor_only_24h_v1.joblib
│       ├── fireguard_real_firms_sensor_only_48h_v1.joblib
│       └── fireguard_real_firms_sensor_only_72h_v1.joblib
│
├── data/
│   ├── retraining/
│   ├── historical/
│   ├── weather/
│   ├── alerts/
│   └── self_learning/
│
├── training/
├── self_learning/
├── tests/
└── reports/
```

The repository may contain additional audit, validation, migration, and compatibility utilities.

---

# Main Application Entry Point

The main Streamlit entry point is:

```text
app.py
```

It delegates to:

```text
dashboard.app.run_dashboard()
```

Run the application from the project root:

```bat
cd /d C:\Users\vista\Desktop\fireguard_v2.0
streamlit run app.py
```

This is the canonical application startup path.

---

# Dashboard

The Streamlit dashboard provides a modular navigation layer over the project's operational subsystems.

Current dashboard areas include:

* Overview
* Live Monitoring
* Seasonal Forecast
* Analytics
* Fire Events
* Forecast
* NASA
* Hyrcanian Historical Evidence
* Weather
* Alerts
* Verification
* Model Status
* System Health
* Database

The dashboard is intentionally kept separate from the underlying inference and database layers.

---

# Real FIRMS Production Models

## Production Model Set

The current Real FIRMS V1 production directory is:

```text
saved_models/real_firms_v1/
```

with:

```text
fireguard_real_firms_sensor_only_24h_v1.joblib
fireguard_real_firms_sensor_only_48h_v1.joblib
fireguard_real_firms_sensor_only_72h_v1.joblib
```

These models are loaded by:

```text
real_firms_production_service.py
```

---

# Production Feature Schema

The approved production feature order is:

```text
01. latitude
02. longitude
03. brightness
04. scan
05. track
06. confidence
07. bright_t31
08. frp
09. hour
10. minute
11. daynight_encoded
12. satellite_encoded
13. instrument_encoded
14. type_encoded
15. season_encoded
```

**Feature count: 15**

The feature order is part of the production model contract and must not be changed without a corresponding model retraining and validation process.

---

# Prediction Horizons

The production system provides three independent forecast horizons:

| Horizon  | Production Model                                 | Threshold |
| -------- | ------------------------------------------------ | --------: |
| 24 hours | `fireguard_real_firms_sensor_only_24h_v1.joblib` |      0.35 |
| 48 hours | `fireguard_real_firms_sensor_only_48h_v1.joblib` |      0.35 |
| 72 hours | `fireguard_real_firms_sensor_only_72h_v1.joblib` |      0.30 |

The thresholds are loaded from:

```text
data/retraining/real_firms_threshold_config_v1.json
```

The production service validates that probabilities are within:

```text
0.0 <= probability <= 1.0
```

and converts probabilities to horizon-specific binary predictions using the approved thresholds.

---

# Production Inference Service

The primary runtime service is:

```text
real_firms_production_service.py
```

It is responsible for:

1. loading the approved model artifacts,
2. loading the approved threshold configuration,
3. preparing the production feature matrix,
4. validating model feature compatibility,
5. running `predict_proba()`,
6. validating probability ranges,
7. applying approved thresholds,
8. returning 24h / 48h / 72h outputs.

The production service does **not** retrain models.

It does **not** modify model artifacts.

It does **not** modify the source dataset.

---

# Example Production Inference Contract

A single record is converted into a prediction result containing:

```text
prob_24h
pred_24h

prob_48h
pred_48h

prob_72h
pred_72h
```

Conceptually:

```text
Input FIRMS record
        ↓
Production Feature Preparation
        ↓
24H model → probability + prediction
48H model → probability + prediction
72H model → probability + prediction
```

---

# Database

FireGuard uses SQLite for operational persistence.

The primary database file is:

```text
data/fireguard_history.db
```

The canonical database schema is defined in:

```text
database/schema.py
```

and database operations are implemented in:

```text
database/db.py
```

## Main Tables

### `sensor_reading`

Stores live ESP32 telemetry:

```text
timestamp
temperature
humidity
smoke
flame
source
created_at
```

### `prediction`

Stores model outputs:

```text
timestamp
model_version
feature_version
probability
uncertainty
horizon
created_at
```

Each forecast execution may create separate records for:

```text
24
48
72
```

### `fire_event`

Stores fire-related operational events.

### `alert`

Stores alert records and their lifecycle state.

### `verification`

Stores verification information associated with fire events.

### `model`

Stores model metadata.

### `training_run`

Stores training-run metadata where applicable.

### `external_observation`

Stores external evidence/observations.

---

# Live ESP32 Sensor Pipeline

The live sensor subsystem is implemented primarily under:

```text
sensors/
```

The live sensor runtime stores telemetry through:

```text
FireGuardDatabase.add_sensor_reading()
```

Example live readings include:

```text
Temperature
Humidity
Smoke
Flame
Source
Timestamp
```

The current project demonstrates real sensor persistence into:

```text
sensor_reading
```

with ESP32 records.

---

# Sensor and ML Separation

The current architecture intentionally distinguishes **sensor evidence** from the **Real FIRMS model feature schema**.

### Current Real FIRMS model inputs

The model currently consumes its approved 15-feature production schema described above.

### Current ESP32 sensor evidence

The ESP32 pipeline stores:

```text
temperature
humidity
smoke
flame
```

These values are available to dashboard monitoring and database history.

They are **not currently part of the 15-feature Real FIRMS production model schema**.

This distinction is important for reproducibility and for avoiding the incorrect assumption that a sensor value affects an ML prediction merely because both systems exist in the same dashboard or database.

---

# Forecast Persistence

After successful production inference, the dashboard can persist the three horizon-specific probabilities to:

```text
prediction
```

The resulting architecture is:

```text
Real FIRMS Production Service
          ↓
   24h / 48h / 72h
          ↓
     SQLite prediction
          ↓
     Dashboard Forecast
```

This enables the dashboard to display the latest persisted prediction history rather than relying solely on temporary in-memory UI state.

---

# Alerts

The alert subsystem is separated from the model layer.

Main components include:

```text
alerts/alert_service.py
alerts/sms_service.py
alerts/prediction_alert_bridge.py
alert_engine/alert_engine.py
alert_engine/alert_store.py
alert_engine/sms_gateway.py
```

The design separates:

```text
ML probability
       ↓
Alert Engine
       ↓
Alert level
       ↓
Alert Service
       ↓
SMS Service
```

The SMS system uses Kavenegar configuration when enabled and correctly configured.

---

# SMS Configuration

SMS delivery is controlled through environment configuration.

Typical configuration values include:

```text
SMS_ENABLED
KAVENEGAR_API_KEY
SMS_SENDER
SMS_RECEIVER
SMS_COOLDOWN_MINUTES
```

A production environment should provide these through `.env`, environment variables, or another secure secret-management mechanism.

**API credentials and recipient information must never be committed to GitHub.**

---

# Environment Configuration

Create a local `.env` file where required.

Example:

```env
SMS_ENABLED=false

KAVENEGAR_API_KEY=
SMS_SENDER=
SMS_RECEIVER=

SMS_COOLDOWN_MINUTES=15
```

For real SMS delivery, configure the appropriate values locally and keep the `.env` file outside version control.

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fireguard_v2.0
```

## 2. Create a virtual environment

Windows:

```bat
python -m venv .venv
```

Activate:

```bat
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If SMS functionality is enabled and the dependency is not already present:

```bash
pip install kavenegar
```

---

# Running FireGuard

From the project root:

```bat
streamlit run app.py
```

The Streamlit application should then be available locally through the address reported by Streamlit, typically:

```text
http://localhost:8501
```

---

# Production Model Verification

Before relying on Real FIRMS production inference, run the dedicated smoke test:

```bat
python real_firms_production_smoke_test.py
```

The smoke test validates:

* required files,
* production dataset availability,
* production feature preparation,
* 15-feature schema,
* threshold configuration,
* model loading,
* model schema compatibility,
* direct prediction,
* reusable production inference,
* probability ranges,
* prediction values,
* output schema.

A successful run reports:

```text
STATUS: 🟢 PRODUCTION INTEGRATION SMOKE TEST PASS
```

---

# Production Service Check

The production service can also be checked directly:

```bat
python real_firms_production_service.py
```

A successful run reports:

```text
STATUS: 🟢 PRODUCTION SERVICE READY
```

---

# Validation Philosophy

FireGuard uses a layered validation approach.

### Model-level validation

Checks include:

* artifact availability,
* feature count,
* probability range,
* binary prediction validity.

### Integration-level validation

Checks include:

* service loading,
* reusable inference,
* dashboard-facing inference,
* output structure.

### Database-level validation

Checks include:

* schema initialization,
* prediction persistence,
* sensor reading persistence,
* alert/event persistence.

### Dashboard-level validation

Checks include:

* page routing,
* service loading,
* data retrieval,
* result rendering.

---

# Data Integrity Principles

The production Real FIRMS pipeline is explicitly designed around these principles:

```text
NO RETRAINING during production inference
NO MODEL MODIFICATION
NO DATASET MODIFICATION
NO SYNTHETIC DATA
NO FABRICATED LABELS
```

Production inference uses the existing approved model artifacts.

---

# Existing Model Archive

The repository may contain historical or archived model artifacts under:

```text
saved_models/archive/
```

These are retained for traceability, rollback, comparison, or historical validation and should not be confused with the active `real_firms_v1` production model set.

---

# Historical and External Evidence

FireGuard also includes components for:

* NASA FIRMS evidence
* Hyrcanian historical fire evidence
* weather observations
* historical datasets
* verification workflows
* forecast audits

These components provide supporting operational or analytical context and are distinct from the core production inference contract unless explicitly integrated into that contract.

---

# Security

## Never commit

Do **not** commit:

```text
.env
API keys
passwords
tokens
private credentials
local database files
personal machine paths
private logs
temporary files
```

At minimum, `.gitignore` should exclude:

```gitignore
.env
.venv/
venv/
__pycache__/
*.pyc
*.log
*.db
*.sqlite
*.sqlite3
.streamlit/secrets.toml
```

---

# Database and Runtime Data Policy

Operational SQLite databases are local runtime state.

For example:

```text
data/fireguard_history.db
```

should normally remain local and should not be committed to a public repository.

The repository should contain:

* source code,
* schemas,
* model artifacts when appropriate,
* configuration templates,
* tests,
* documentation.

It should not contain live operational records unless there is a deliberate anonymized dataset policy.

---

# Testing

Run the project test suite with:

```bash
pytest
```

Specific database tests:

```bash
pytest database/test_database.py
```

Sensor tests:

```bash
pytest tests/test_sensor.py
```

Dashboard tests:

```bash
pytest tests/test_dashboard.py
```

Forecast tests:

```bash
pytest tests/test_forecast.py
```

---

# Recommended Verification Sequence

For a clean deployment or demonstration, use this order:

```text
1. Install dependencies
2. Verify database initialization
3. Verify live sensor ingestion
4. Run Real FIRMS production smoke test
5. Verify production service
6. Start Streamlit
7. Verify Forecast page
8. Verify 24h / 48h / 72h persistence
9. Verify Alerts
10. Verify SMS configuration if enabled
```

---

# Demonstration Workflow

A recommended evaluator/demo workflow is:

### Step 1 — Show live sensor evidence

Open:

```text
Live Monitoring
```

and demonstrate:

```text
Temperature
Humidity
Smoke
Flame
```

coming from the ESP32 pipeline.

### Step 2 — Show Real FIRMS forecast

Open:

```text
Seasonal Forecast
```

and run the production inference.

Demonstrate:

```text
24H probability
48H probability
72H probability
```

### Step 3 — Show persistence

Open:

```text
Forecast
```

and demonstrate that the probabilities are persisted in the FireGuard database.

### Step 4 — Show architecture separation

Explain that:

```text
Sensor Evidence
```

and:

```text
Real FIRMS ML Forecast
```

are currently separate production data paths.

This distinction is deliberate and reproducible.

---

# Example Database Verification

To inspect recent predictions locally:

```bat
python -c "import sqlite3; c=sqlite3.connect('data/fireguard_history.db'); print(*c.execute('SELECT id,timestamp,model_version,feature_version,probability,uncertainty,horizon FROM prediction ORDER BY id DESC LIMIT 10').fetchall(), sep='\n')"
```

To inspect recent sensor readings:

```bat
python -c "import sqlite3; c=sqlite3.connect('data/fireguard_history.db'); print(*c.execute('SELECT id,timestamp,temperature,humidity,smoke,flame,source FROM sensor_reading ORDER BY id DESC LIMIT 10').fetchall(), sep='\n')"
```

---

# Research and Model Scope

The Real FIRMS V1 pipeline is designed around real FIRMS-derived records used for production inference.

The project distinguishes between:

* remote-sensing fire observations,
* live sensor telemetry,
* model inference,
* alert evaluation,
* and operational dashboard presentation.

This separation makes it easier to audit what each component actually contributes.

---

# Current Limitations

The current Real FIRMS model uses a fixed 15-feature production schema.

The following ESP32 fields are currently available as live sensor evidence but are not part of the active Real FIRMS 15-feature model input:

```text
temperature
humidity
smoke
flame
```

Creating a genuinely sensor-fused ML model would require:

1. defining a new feature schema,
2. preparing an appropriate training dataset,
3. retraining the models,
4. evaluating them independently,
5. validating thresholds,
6. producing new production artifacts,
7. and updating the production service contract.

Simply adding sensor fields to a Streamlit form would not constitute sensor-aware model inference.

---

# Reproducibility

For reproducible results, record:

```text
Model artifact
Feature schema
Threshold configuration
Input record
Timestamp
Prediction horizon
Probability
Prediction
```

Production artifacts and configuration should be versioned together.

---

# Repository Hygiene

A clean GitHub repository should prioritize:

```text
Source Code
Production Models
Schemas
Tests
Configuration Templates
Documentation
```

and exclude:

```text
Caches
Virtual Environments
Live Databases
Secrets
Temporary Outputs
Large Unnecessary Raw Data
Local Logs
Machine-Specific Files
```

---

# Project Status

Current demonstrated production components include:

```text
Real FIRMS 24H model      ✅
Real FIRMS 48H model      ✅
Real FIRMS 72H model      ✅
15-feature schema         ✅
Threshold configuration   ✅
Production smoke test     ✅
Production service        ✅
Prediction persistence    ✅
ESP32 sensor persistence  ✅
Streamlit dashboard      ✅
Forecast dashboard       ✅
Alert subsystem           ✅
SMS service integration   ✅
```

The exact production state should always be verified using the project's current smoke tests and runtime configuration rather than inferred solely from this document.

---

# Development Guidelines

When extending FireGuard:

### Do

* preserve the existing production feature contract,
* isolate experimental code from production inference,
* add tests for new behavior,
* keep database writes explicit,
* keep model artifacts versioned,
* document schema changes,
* preserve reproducibility.

### Do not

* silently alter model features,
* change thresholds without validation,
* overwrite approved production models,
* inject synthetic observations into production inference,
* commit secrets,
* use test data as production evidence,
* claim sensor influence that is not part of the actual model schema.

---

# License

Add the project's chosen license here.

Example:

```text
MIT License
```

or replace it with the appropriate institutional/research license.

---

# Acknowledgements

FireGuard integrates concepts and data workflows associated with:

* NASA FIRMS
* ESP32-based environmental sensing
* Python scientific computing
* scikit-learn
* Streamlit
* SQLite
* Kavenegar

---

# Contact

**Project:** FireGuard v2.0

**Repository:** `<YOUR_GITHUB_REPOSITORY_URL>`

**Maintainer:** `<YOUR_NAME_OR_ORGANIZATION>`

**Primary Purpose:** Wildfire monitoring, prediction, evidence integration, and operational alerting for the Hyrcanian ecosystem.

---

# Final Architecture Summary

```text
┌─────────────────────────────────────────────────────────────┐
│                       FIREGUARD v2.0                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ESP32 SENSOR PIPELINE                                      │
│  ├── Temperature                                            │
│  ├── Humidity                                               │
│  ├── Smoke                                                  │
│  └── Flame                                                  │
│          │                                                  │
│          ▼                                                  │
│    sensor_reading                                           │
│                                                             │
│                         +                                   │
│                                                             │
│  NASA FIRMS PRODUCTION PIPELINE                             │
│  ├── FIRMS-derived features                                │
│  ├── 24H production model                                  │
│  ├── 48H production model                                  │
│  └── 72H production model                                  │
│          │                                                  │
│          ▼                                                  │
│    real_firms_production_service.py                        │
│          │                                                  │
│          ▼                                                  │
│      prediction                                             │
│                                                             │
│                         +                                   │
│                                                             │
│  ALERT PIPELINE                                             │
│  ├── AlertEngine                                            │
│  ├── AlertService                                           │
│  ├── Cooldown                                               │
│  └── SMS / Kavenegar                                        │
│                                                             │
│                         │                                   │
│                         ▼                                   │
│                 STREAMLIT DASHBOARD                         │
│                                                             │
│  Overview | Live | Forecast | Seasonal | Alerts | Analytics │
│                                                             │
└─────────────────────────────────────────────────────────────┘
**FireGuard v2.0 is structured as a modular wildfire intelligence platform in which live sensor evidence, remote-sensing inference, database persistence, dashboard visualization, and alert delivery remain separately identifiable and auditable components.**
