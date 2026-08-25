import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FireGuard - Synthetic Time-Series Forecast Dataset v2
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "data" / "fireguard_forecast_30days.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

START = "2026-07-01 00:00:00"
MINUTES = 30 * 24 * 60

timestamps = pd.date_range(
    start=START,
    periods=MINUTES,
    freq="min"
)

df = pd.DataFrame({"timestamp": timestamps})
n = len(df)

# ------------------------------------------------------------
# Base environment
# ------------------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day

minute_of_day = (
    df["hour"].to_numpy() * 60 + df["minute"].to_numpy()
)
time_index = np.arange(n)

daily_temperature = (
    27
    + 7 * np.sin(2 * np.pi * (minute_of_day - 8 * 60) / 1440)
)

weekly_temperature = (
    1.5 * np.sin(2 * np.pi * time_index / (7 * 1440))
)

temperature = (
    daily_temperature
    + weekly_temperature
    + rng.normal(0, 0.8, n)
)

humidity = (
    65
    - 1.0 * (temperature - 25)
    + 4 * np.sin(2 * np.pi * time_index / (3 * 1440))
    + rng.normal(0, 2, n)
)

smoke = (
    30
    + 5 * np.sin(2 * np.pi * time_index / 1440)
    + rng.normal(0, 3, n)
)

flame = np.zeros(n, dtype=int)

# ============================================================
# Fire scenarios
# ============================================================
# start, prefire_min, ignition_min, peak_min, recovery_min,
# smoke_scale, temp_scale, humidity_drop, flame_delay_min
#
# Different timings/intensities/durations prevent day-based
# shortcuts and create multiple sensor-behavior patterns.
# ============================================================

fire_events = [
    ("2026-07-03 17:23", 95, 28, 72, 190, 0.75, 0.70, 0.75, 8),
    ("2026-07-06 04:41", 210, 42, 135, 240, 1.15, 1.00, 1.00, 20),
    ("2026-07-09 13:17", 55, 18, 48, 150, 0.55, 0.50, 0.55, 4),
    ("2026-07-12 22:08", 145, 35, 180, 270, 1.35, 1.20, 1.15, 25),
    ("2026-07-16 08:52", 75, 30, 95, 180, 0.90, 0.80, 0.85, 10),
    ("2026-07-19 15:36", 260, 50, 70, 210, 0.65, 0.70, 0.70, 15),
    ("2026-07-22 02:14", 35, 22, 120, 220, 1.05, 0.95, 0.95, 6),
    ("2026-07-25 19:47", 180, 45, 210, 300, 1.45, 1.30, 1.25, 30),
    ("2026-07-28 06:31", 65, 20, 55, 150, 0.60, 0.60, 0.60, 5),
    ("2026-07-29 16:19", 120, 32, 150, 240, 1.10, 0.90, 1.00, 18),
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

    # PRE-FIRE: gradual sensor drift
    mask = (
        (df["timestamp"] >= pre_start)
        & (df["timestamp"] < fire_start)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)
        smoke[mask] += 70 * smoke_scale * progress
        temperature[mask] += 3.0 * temp_scale * progress
        humidity[mask] -= 8.0 * humidity_scale * progress

    # IGNITION: rapidly increasing signal
    mask = (
        (df["timestamp"] >= fire_start)
        & (df["timestamp"] < peak_start)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)
        smoke[mask] += (
            75 * smoke_scale + 175 * smoke_scale * progress
        )
        temperature[mask] += (
            4.0 * temp_scale + 7.0 * temp_scale * progress
        )
        humidity[mask] -= (
            12.0 * humidity_scale + 12.0 * humidity_scale * progress
        )

    # FIRE PEAK: flame appears only after a scenario-specific delay
    mask = (
        (df["timestamp"] >= peak_start)
        & (df["timestamp"] < fire_end)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(0, 1, count)

        smoke[mask] += (
            230 * smoke_scale
            + 130 * smoke_scale * np.sin(np.pi * progress)
        )
        temperature[mask] += (
            11 * temp_scale
            + 5 * temp_scale * np.sin(np.pi * progress)
        )
        humidity[mask] -= (
            27 * humidity_scale
            + 8 * humidity_scale * np.sin(np.pi * progress)
        )

        peak_times = df.loc[mask, "timestamp"].to_numpy()
        flame_start = peak_start + pd.Timedelta(minutes=flame_delay)
        flame_mask = (
            (df["timestamp"] >= flame_start)
            & (df["timestamp"] < fire_end)
        )
        flame[flame_mask] = 1

    # RECOVERY: fading but still abnormal
    mask = (
        (df["timestamp"] >= fire_end)
        & (df["timestamp"] < recovery_end)
    )
    count = int(mask.sum())

    if count:
        progress = np.linspace(1, 0, count)
        smoke[mask] += 85 * smoke_scale * progress
        temperature[mask] += 4.5 * temp_scale * progress
        humidity[mask] -= 10 * humidity_scale * progress

# ============================================================
# False alarms - multiple sensor patterns, flame stays zero
# ============================================================

false_events = [
    # smoke-only spike
    ("2026-07-04 08:11", 38, 120, 0, 0),

    # long low smoke event
    ("2026-07-08 20:26", 105, 75, 0, 0),

    # smoke + temperature disturbance
    ("2026-07-11 09:43", 52, 135, 4.0, 0),

    # humid-condition smoke disturbance
    ("2026-07-15 18:37", 80, 105, 0, -5.0),

    # sharp short spike
    ("2026-07-18 03:52", 24, 190, 0, 0),

    # mixed gradual disturbance
    ("2026-07-23 11:18", 125, 95, 2.5, -3.0),

    # evening spike
    ("2026-07-27 21:05", 65, 145, 3.0, 0),

    # post-fire-like smoke without flame
    ("2026-07-29 09:12", 90, 125, 1.5, -2.0),
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

# ============================================================
# Environmental sensor noise / small drift
# ============================================================

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

# Last 72 hours have no complete future label.
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
# Distribution report
# ============================================================

def report_binary(column):
    counts = df[column].value_counts().sort_index()
    percentages = (
        df[column].value_counts(normalize=True)
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
print("🔥 FireGuard Synthetic Forecast Dataset v2")
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
for horizon in ("24h", "48h", "72h"):
    print(f"{horizon} Target:")
    print(report_binary(f"fire_next_{horizon}").to_string())
    print()

print("Fire Now:")
print(report_binary("fire_now").to_string())

print()
print("Fire event count:", len(fire_events))
print("False alarm count:", len(false_events))

print()
print("Columns:")
print(list(df.columns))

print()
print("✅ Dataset v2 آماده است.")
