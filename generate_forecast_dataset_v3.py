import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FireGuard - Synthetic Time-Series Forecast Dataset v3
# 60 days / minute-level / varied fire scenarios
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "data" / "fireguard_forecast_60days.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

START = "2026-06-01 00:00:00"
MINUTES = 60 * 24 * 60

timestamps = pd.date_range(
    start=START,
    periods=MINUTES,
    freq="min"
)

df = pd.DataFrame({"timestamp": timestamps})
n = len(df)

# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day

minute_of_day = df["hour"].to_numpy() * 60 + df["minute"].to_numpy()
time_index = np.arange(n)

# ------------------------------------------------------------
# Environmental baseline
# ------------------------------------------------------------

daily_temperature = (
    27
    + 7 * np.sin(2 * np.pi * (minute_of_day - 8 * 60) / 1440)
)

slow_temperature = (
    1.8 * np.sin(2 * np.pi * time_index / (9 * 1440))
)

temperature = (
    daily_temperature
    + slow_temperature
    + rng.normal(0, 0.8, n)
)

humidity = (
    66
    - 1.05 * (temperature - 25)
    + 3.5 * np.sin(2 * np.pi * time_index / (4 * 1440))
    + rng.normal(0, 2.0, n)
)

smoke = (
    30
    + 5 * np.sin(2 * np.pi * time_index / 1440)
    + 2 * np.sin(2 * np.pi * time_index / (5 * 1440))
    + rng.normal(0, 3, n)
)

flame = np.zeros(n, dtype=int)

# ============================================================
# Fire events
# 12 events across 60 days with irregular spacing.
#
# start, prefire, ignition, peak, recovery,
# smoke_scale, temp_scale, humidity_scale, flame_delay
# ============================================================

fire_events = [
    ("2026-06-04 17:23", 90, 25, 70, 180, 0.75, 0.70, 0.75, 8),
    ("2026-06-09 04:41", 180, 40, 125, 230, 1.10, 1.00, 1.00, 18),
    ("2026-06-14 13:17", 50, 18, 45, 140, 0.55, 0.50, 0.55, 4),
    ("2026-06-20 22:08", 135, 35, 170, 260, 1.30, 1.15, 1.10, 25),
    ("2026-06-26 08:52", 75, 28, 90, 170, 0.90, 0.80, 0.85, 10),
    ("2026-07-02 15:36", 220, 45, 75, 200, 0.65, 0.70, 0.70, 14),
    ("2026-07-08 02:14", 35, 20, 110, 210, 1.05, 0.95, 0.95, 6),
    ("2026-07-14 19:47", 165, 42, 190, 290, 1.40, 1.25, 1.20, 28),
    ("2026-07-21 06:31", 60, 22, 55, 145, 0.60, 0.60, 0.60, 5),
    ("2026-07-27 16:19", 115, 30, 145, 220, 1.05, 0.90, 0.95, 17),
    ("2026-08-03 11:06", 240, 48, 100, 210, 0.80, 0.75, 0.80, 20),
    ("2026-08-09 23:38", 85, 26, 160, 250, 1.20, 1.05, 1.05, 12),
]

for (
    start_time,
    prefire,
    ignition,
    peak,
    recovery,
    smoke_scale,
    temp_scale,
    humidity_scale,
    flame_delay,
) in fire_events:

    fire_start = pd.Timestamp(start_time)
    pre_start = fire_start - pd.Timedelta(minutes=prefire)
    peak_start = fire_start + pd.Timedelta(minutes=ignition)
    fire_end = peak_start + pd.Timedelta(minutes=peak)
    recovery_end = fire_end + pd.Timedelta(minutes=recovery)

    # PRE-FIRE
    mask = (
        (df["timestamp"] >= pre_start)
        & (df["timestamp"] < fire_start)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)
        smoke[mask] += 65 * smoke_scale * progress
        temperature[mask] += 3.0 * temp_scale * progress
        humidity[mask] -= 8.0 * humidity_scale * progress

    # IGNITION
    mask = (
        (df["timestamp"] >= fire_start)
        & (df["timestamp"] < peak_start)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)
        smoke[mask] += (
            70 * smoke_scale
            + 180 * smoke_scale * progress
        )
        temperature[mask] += (
            4.0 * temp_scale
            + 7.0 * temp_scale * progress
        )
        humidity[mask] -= (
            12.0 * humidity_scale
            + 12.0 * humidity_scale * progress
        )

    # FIRE PEAK
    mask = (
        (df["timestamp"] >= peak_start)
        & (df["timestamp"] < fire_end)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)

        smoke[mask] += (
            225 * smoke_scale
            + 135 * smoke_scale * np.sin(np.pi * progress)
        )

        temperature[mask] += (
            11 * temp_scale
            + 5 * temp_scale * np.sin(np.pi * progress)
        )

        humidity[mask] -= (
            27 * humidity_scale
            + 8 * humidity_scale * np.sin(np.pi * progress)
        )

        flame_start = peak_start + pd.Timedelta(minutes=flame_delay)

        flame_mask = (
            (df["timestamp"] >= flame_start)
            & (df["timestamp"] < fire_end)
        )

        flame[flame_mask] = 1

    # RECOVERY
    mask = (
        (df["timestamp"] >= fire_end)
        & (df["timestamp"] < recovery_end)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(1, 0, count)
        smoke[mask] += 80 * smoke_scale * progress
        temperature[mask] += 4.0 * temp_scale * progress
        humidity[mask] -= 10 * humidity_scale * progress

# ============================================================
# False alarms: 12 varied non-fire disturbances
# flame always remains 0
# ============================================================

false_events = [
    ("2026-06-06 08:11", 40, 120, 0.0, 0.0),
    ("2026-06-11 20:26", 105, 75, 0.0, 0.0),
    ("2026-06-16 09:43", 55, 135, 4.0, 0.0),
    ("2026-06-22 18:37", 80, 105, 0.0, -5.0),
    ("2026-06-29 03:52", 24, 190, 0.0, 0.0),
    ("2026-07-05 11:18", 125, 95, 2.5, -3.0),
    ("2026-07-11 21:05", 65, 145, 3.0, 0.0),
    ("2026-07-18 07:44", 90, 125, 1.5, -2.0),
    ("2026-07-24 14:12", 35, 170, 0.0, 0.0),
    ("2026-07-31 00:48", 150, 85, 2.0, -4.0),
    ("2026-08-06 17:29", 50, 155, 3.5, 0.0),
    ("2026-08-12 10:16", 115, 110, 1.0, -2.5),
]

for start_time, duration, intensity, temp_shift, humidity_shift in false_events:
    start = pd.Timestamp(start_time)
    end = start + pd.Timedelta(minutes=duration)

    mask = (
        (df["timestamp"] >= start)
        & (df["timestamp"] < end)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)
        pulse = np.sin(np.pi * progress)

        smoke[mask] += intensity * pulse
        temperature[mask] += temp_shift * pulse
        humidity[mask] += humidity_shift * pulse

# ------------------------------------------------------------
# Noise / bounds
# ------------------------------------------------------------

temperature += 0.12 * np.sin(2 * np.pi * time_index / 480)
humidity += rng.normal(0, 0.6, n)
smoke += rng.normal(0, 1.2, n)

temperature = np.clip(temperature, 5, 55)
humidity = np.clip(humidity, 5, 95)
smoke = np.clip(smoke, 5, 600)

df["temperature"] = np.round(temperature, 2)
df["humidity"] = np.round(humidity, 2)
df["smoke"] = np.round(smoke, 2)
df["flame"] = flame

# ============================================================
# Targets
# ============================================================

df["fire_now"] = df["flame"].astype(int)

HORIZONS = {
    "1h": 60,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "72h": 4320,
}

fire = df["fire_now"].to_numpy()

for name, minutes in HORIZONS.items():
    target = np.full(n, np.nan)

    for i in range(n - minutes):
        target[i] = int(
            np.any(
                fire[i + 1:i + minutes + 1] == 1
            )
        )

    df[f"fire_next_{name}"] = target

# Remove final 72 hours without complete labels
df = df.iloc[:-4320].copy()

# ============================================================
# State
# ============================================================

df["state"] = np.select(
    [
        df["flame"] == 1,
        (
            (df["smoke"] >= 180)
            & (df["humidity"] <= 45)
        ),
        df["smoke"] >= 90,
    ],
    [
        "fire",
        "pre_fire",
        "smoke_alert",
    ],
    default="normal",
)

# ============================================================
# Reports
# ============================================================

def binary_report(column):
    counts = df[column].value_counts().sort_index()
    percentages = (
        df[column]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    return pd.DataFrame({
        "count": counts,
        "percent": percentages,
    })

# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 70)
print("🔥 FireGuard Synthetic Forecast Dataset v3")
print("=" * 70)
print(f"File    : {OUTPUT_FILE}")
print(f"Records : {len(df):,}")
print(f"Start   : {df['timestamp'].min()}")
print(f"End     : {df['timestamp'].max()}")

print()
print("State distribution:")
print(df["state"].value_counts().to_string())

print()
print("State percentage:")
print(
    df["state"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)

print()
print("Fire Now:")
print(binary_report("fire_now").to_string())

for horizon in ("24h", "48h", "72h"):
    print()
    print(f"{horizon} Target:")
    print(binary_report(f"fire_next_{horizon}").to_string())

print()
print(f"Fire event count  : {len(fire_events)}")
print(f"False alarm count : {len(false_events)}")

print()
print("Columns:")
print(list(df.columns))

print()
print("✅ Dataset v3 آماده است.")
