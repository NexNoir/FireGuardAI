# kj.py

from database import rebuild_database
from forecast.real_forecast_v3 import FireGuardForecast


def main():

    print("=" * 60)
    print("🔥 FireGuard - Forecast System v3")
    print("=" * 60)

    # --------------------------------------------------
    # 1. بررسی / آماده‌سازی دیتابیس
    # --------------------------------------------------

    print("\n[1] بررسی دیتابیس...")

    try:
        rebuild_database()

    except Exception as e:

        print(
            f"❌ خطا در آماده‌سازی دیتابیس: {e}"
        )

        return

    # --------------------------------------------------
    # 2. ساخت Forecast
    # --------------------------------------------------

    print("\n[2] بارگذاری Forecast v3...")

    try:

        forecast = FireGuardForecast()

    except Exception as e:

        print(
            f"❌ خطا در ساخت Forecast: {e}"
        )

        return

    # --------------------------------------------------
    # 3. بارگذاری مدل‌های ذخیره‌شده
    # --------------------------------------------------

    print(
        "\n[3] بارگذاری مدل‌های آموزش‌دیده..."
    )

    try:

        loaded = (
            forecast._load_saved_models(
                "sensor_only"
            )
        )

        if not loaded:

            print(
                "❌ مدل‌های v3 پیدا نشدند."
            )

            print(
                "ابتدا real_forecast_v3.py را اجرا کن."
            )

            return

        print(
            "✅ مدل‌های Sensor-only بارگذاری شدند."
        )

    except Exception as e:

        print(
            f"❌ خطا هنگام بارگذاری مدل‌ها: {e}"
        )

        return

    # --------------------------------------------------
    # 4. تست Forecast با داده نمونه
    # --------------------------------------------------

    print(
        "\n[4] تست get_forecast()..."
    )

    current_reading = {

        "timestamp":
            "2026-07-27 12:00:00",

        "temperature":
            32.5,

        "humidity":
            42.0,

        "smoke":
            85.0,

        "flame":
            0,

        "smoke_change_1m":
            2.0,

        "smoke_change_5m":
            8.0,

        "smoke_change_15m":
            18.0,

        "smoke_change_30m":
            30.0,

        "smoke_change_60m":
            45.0,

        "temperature_change_5m":
            0.8,

        "temperature_change_15m":
            1.5,

        "temperature_change_30m":
            2.5,

        "temperature_change_60m":
            4.0,

        "humidity_change_5m":
            -1.0,

        "humidity_change_15m":
            -3.0,

        "humidity_change_30m":
            -5.0,

        "smoke_mean_5m":
            82.0,

        "smoke_mean_15m":
            76.0,

        "smoke_mean_30m":
            68.0,

        "smoke_mean_60m":
            55.0,

        "smoke_std_15m":
            8.0,

        "smoke_std_30m":
            12.0,

        "smoke_max_30m":
            90.0,

        "temperature_mean_15m":
            31.8,

        "temperature_mean_30m":
            30.9,

        "humidity_mean_15m":
            44.0,

        "humidity_mean_30m":
            47.0,
    }

    try:

        result = forecast.get_forecast(
            current_reading
        )

    except Exception as e:

        print(
            f"❌ خطا در Forecast: {e}"
        )

        return

    # --------------------------------------------------
    # 5. نمایش نتیجه
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("🔥 FIREGUARD FORECAST")
    print("=" * 60)

    if not result.get("success"):

        print(
            "❌ Forecast ناموفق بود."
        )

        print(
            result
        )

        return

    forecasts = result["forecast"]

    for horizon in (
        "24h",
        "48h",
        "72h"
    ):

        data = forecasts.get(
            horizon,
            {}
        )

        print()
        print(
            f"--- {horizon} ---"
        )

        print(
            f"Probability : "
            f"{data.get('fire_probability_percent', 0):.2f}%"
        )

        print(
            f"Threshold   : "
            f"{data.get('threshold', 0):.2f}"
        )

        print(
            f"Prediction  : "
            f"{data.get('prediction', 0)}"
        )

        print(
            f"Risk        : "
            f"{data.get('risk_level', 'UNKNOWN')}"
        )

    # --------------------------------------------------
    # 6. پایان
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("✅ FireGuard Forecast v3 آماده است.")
    print("=" * 60)


if __name__ == "__main__":
    main()