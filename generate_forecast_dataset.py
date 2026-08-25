import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FireGuard - Synthetic Time-Series Forecast Dataset
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "data" / "fireguard_forecast_30days.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. ساخت سری زمانی 30 روزه
# ------------------------------------------------------------

START = "2026-07-01 00:00:00"
MINUTES = 30 * 24 * 60

timestamps = pd.date_range(
    start=START,
    periods=MINUTES,
    freq="min"
)

df = pd.DataFrame({
    "timestamp": timestamps
})

n = len(df)

# ------------------------------------------------------------
# 2. ویژگی‌های زمانی
# ------------------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day"] = df["timestamp"].dt.day

minute_of_day = (
    df["hour"].to_numpy() * 60
    + df["minute"].to_numpy()
)

time_index = np.arange(n)

# ------------------------------------------------------------
# 3. دمای پایه
# ------------------------------------------------------------

daily_temperature = (
    27
    + 7 * np.sin(
        2 * np.pi * (minute_of_day - 8 * 60) / 1440
    )
)

weekly_temperature = (
    1.5
    * np.sin(
        2 * np.pi * time_index / (7 * 1440)
    )
)

temperature = (
    daily_temperature
    + weekly_temperature
    + rng.normal(0, 0.8, n)
)

# ------------------------------------------------------------
# 4. رطوبت
# ------------------------------------------------------------

humidity = (
    65
    - 1.0 * (temperature - 25)
    + 4 * np.sin(
        2 * np.pi * time_index / (3 * 1440)
    )
    + rng.normal(0, 2, n)
)

humidity = np.clip(
    humidity,
    10,
    95
)

# ------------------------------------------------------------
# 5. دود پایه
# ------------------------------------------------------------

smoke = (
    30
    + 5 * np.sin(
        2 * np.pi * time_index / 1440
    )
    + rng.normal(0, 3, n)
)

smoke = np.clip(
    smoke,
    5,
    None
)

# ------------------------------------------------------------
# 6. شعله
# ------------------------------------------------------------

flame = np.zeros(
    n,
    dtype=int
)

# ============================================================
# 7. سناریوهای آتش
# ============================================================

fire_events = [
    ("2026-07-04 14:00", 120, 30, 90),
    ("2026-07-09 11:30", 180, 35, 120),
    ("2026-07-15 15:00", 150, 30, 100),
    ("2026-07-21 13:00", 210, 40, 140),
    ("2026-07-26 16:30", 120, 25, 80),
    ("2026-07-29 12:00", 180, 35, 110),
]

for start_time, prefire, ignition, peak in fire_events:

    fire_start = pd.Timestamp(start_time)

    pre_start = (
        fire_start
        - pd.Timedelta(minutes=prefire)
    )

    ignition_start = fire_start

    peak_start = (
        ignition_start
        + pd.Timedelta(minutes=ignition)
    )

    fire_end = (
        peak_start
        + pd.Timedelta(minutes=peak)
    )

    recovery_end = (
        fire_end
        + pd.Timedelta(minutes=180)
    )

    # --------------------------------------------------------
    # PRE-FIRE
    # --------------------------------------------------------

    mask = (
        (df["timestamp"] >= pre_start)
        &
        (df["timestamp"] < ignition_start)
    )

    count = mask.sum()

    if count > 0:

        progress = np.linspace(
            0,
            1,
            count
        )

        smoke[mask] += (
            80 * progress
        )

        temperature[mask] += (
            3.5 * progress
        )

        humidity[mask] -= (
            10 * progress
        )

    # --------------------------------------------------------
    # IGNITION
    # --------------------------------------------------------

    mask = (
        (df["timestamp"] >= ignition_start)
        &
        (df["timestamp"] < peak_start)
    )

    count = mask.sum()

    if count > 0:

        progress = np.linspace(
            0,
            1,
            count
        )

        smoke[mask] += (
            100
            + 180 * progress
        )

        temperature[mask] += (
            6
            + 8 * progress
        )

        humidity[mask] -= (
            18
            + 12 * progress
        )

    # --------------------------------------------------------
    # FIRE PEAK
    # --------------------------------------------------------

    mask = (
        (df["timestamp"] >= peak_start)
        &
        (df["timestamp"] < fire_end)
    )

    count = mask.sum()

    if count > 0:

        progress = np.linspace(
            0,
            1,
            count
        )

        smoke[mask] += (
            280
            + 100
            * np.sin(np.pi * progress)
        )

        temperature[mask] += (
            14
            + 4
            * np.sin(np.pi * progress)
        )

        humidity[mask] -= (
            32
            + 6
            * np.sin(np.pi * progress)
        )

        flame[mask] = 1

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    mask = (
        (df["timestamp"] >= fire_end)
        &
        (df["timestamp"] < recovery_end)
    )

    count = mask.sum()

    if count > 0:

        progress = np.linspace(
            1,
            0,
            count
        )

        smoke[mask] += (
            80 * progress
        )

        temperature[mask] += (
            4 * progress
        )

        humidity[mask] -= (
            10 * progress
        )

# ============================================================
# 8. False Alarm
# ============================================================

false_events = [
    ("2026-07-06 09:00", 45, 110),
    ("2026-07-12 18:30", 60, 150),
    ("2026-07-18 10:00", 30, 90),
    ("2026-07-24 08:30", 70, 120),
    ("2026-07-28 19:00", 50, 100),
]

for start_time, duration, intensity in false_events:

    start = pd.Timestamp(start_time)

    end = (
        start
        + pd.Timedelta(minutes=duration)
    )

    mask = (
        (df["timestamp"] >= start)
        &
        (df["timestamp"] < end)
    )

    count = mask.sum()

    if count > 0:

        progress = np.linspace(
            0,
            1,
            count
        )

        smoke[mask] += (
            intensity
            * np.sin(np.pi * progress)
        )

        # عمداً flame صفر باقی می‌ماند.
        # این بخش برای False Alarm مهم است.

# ============================================================
# 9. محدود کردن مقادیر
# ============================================================

temperature = np.clip(
    temperature,
    5,
    55
)

humidity = np.clip(
    humidity,
    5,
    95
)

smoke = np.clip(
    smoke,
    5,
    600
)

df["temperature"] = np.round(
    temperature,
    2
)

df["humidity"] = np.round(
    humidity,
    2
)

df["smoke"] = np.round(
    smoke,
    2
)

df["flame"] = flame

# ============================================================
# 10. Current Fire Target
# ============================================================

df["fire_now"] = (
    df["flame"]
    .astype(int)
)

# ============================================================
# 11. Future Targets
#
# آیا در آینده آتش رخ خواهد داد؟
# ============================================================

fire = (
    df["fire_now"]
    .to_numpy()
)

HORIZONS = {
    "1h": 60,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "72h": 4320,
}

for name, minutes in HORIZONS.items():

    target = np.full(
        n,
        np.nan
    )

    for i in range(
        0,
        n - minutes
    ):

        future_window = fire[
            i + 1:
            i + minutes + 1
        ]

        target[i] = int(
            np.any(
                future_window == 1
            )
        )

    df[
        f"fire_next_{name}"
    ] = target

# ============================================================
# 12. حذف انتهای Dataset
#
# چون برای آخرین 72 ساعت Future Label نداریم.
# ============================================================

df = df.iloc[
    :-4320
].copy()

# ============================================================
# 13. State
# ============================================================

df["state"] = np.select(
    [
        df["flame"] == 1,

        (
            (df["smoke"] >= 180)
            &
            (df["humidity"] <= 45)
        ),

        df["smoke"] >= 90
    ],
    [
        "fire",
        "pre_fire",
        "smoke_alert"
    ],
    default="normal"
)

# ============================================================
# 14. ذخیره
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 15. گزارش
# ============================================================

print()
print("=" * 60)
print("🔥 FireGuard Synthetic Forecast Dataset")
print("=" * 60)

print(
    f"File: {OUTPUT_FILE}"
)

print(
    f"Records: {len(df):,}"
)

print(
    f"Start: {df['timestamp'].min()}"
)

print(
    f"End: {df['timestamp'].max()}"
)

print()
print("State distribution:")
print(
    df["state"].value_counts()
)

print()
print("Fire Now:")
print(
    df["fire_now"].value_counts()
)

print()
print("24h Target:")
print(
    df["fire_next_24h"].value_counts()
)

print()
print("48h Target:")
print(
    df["fire_next_48h"].value_counts()
)

print()
print("72h Target:")
print(
    df["fire_next_72h"].value_counts()
)

print()
print("Columns:")
print(
    list(df.columns)
)

print()
print("✅ Dataset آماده است.")