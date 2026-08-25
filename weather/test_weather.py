from open_meteo_client import (
    fetch_current_weather,
    fetch_forecast_weather,
    fetch_historical_weather,
)


print("=" * 70)
print("🌦️ FireGuard Weather REAL DATA TEST")
print("=" * 70)

print()
print("CURRENT WEATHER")
print("-" * 70)

current = fetch_current_weather()

if current is None or current.empty:
    print("STATUS: Current Weather unavailable")
else:
    print("STATUS: Current Weather OK")
    print()
    print(current.to_string(index=False))

print()
print("FORECAST WEATHER")
print("-" * 70)

forecast = fetch_forecast_weather(72)

if forecast is None or forecast.empty:
    print("STATUS: Forecast Weather unavailable")
else:
    print(f"STATUS: Forecast Weather OK")
    print(f"Rows: {len(forecast)}")
    print()
    print(forecast.head(5).to_string(index=False))
    print("...")
    print(forecast.tail(3).to_string(index=False))

print()
print("HISTORICAL WEATHER")
print("-" * 70)

historical = fetch_historical_weather(
    "2026-06-01",
    "2026-07-27",
)

if historical is None or historical.empty:
    print("STATUS: Historical Weather unavailable")
else:
    print("STATUS: Historical Weather OK")
    print(f"Rows: {len(historical)}")
    print()
    print(historical.head(5).to_string(index=False))

print()
print("=" * 70)
print("REQUIRED FIELD CHECK")
print("=" * 70)

required = [
    "timestamp",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "pressure",
    "cloud_cover",
    "dew_point",
]

if forecast is not None and not forecast.empty:
    missing = [x for x in required if x not in forecast.columns]

    if not missing:
        print("All required fields: PASS")
    else:
        print("Missing fields:", missing)
else:
    print("Field check: SKIPPED — forecast unavailable")

print()
print("=" * 70)
print("FINAL STATUS")
print("=" * 70)

if (
    current is not None
    and not current.empty
    and forecast is not None
    and not forecast.empty
    and historical is not None
    and not historical.empty
):
    print("🟢 WEATHER API READY")
else:
    print("🟡 WEATHER PARTIALLY AVAILABLE")
    print("FireGuard must continue without fake weather values.")

print("=" * 70)