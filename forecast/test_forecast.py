from real_forecast_v3 import FireGuardForecast

engine = FireGuardForecast()

engine.set_experiment("sensor_only")

reading = {
    "timestamp": "2026-07-27 12:00:00",

    "temperature": 32.5,
    "humidity": 42.0,
    "smoke": 85.0,
    "flame": 0,

    "smoke_change_1m": 2.0,
    "smoke_change_5m": 8.0,
    "smoke_change_15m": 18.0,
    "smoke_change_30m": 30.0,
    "smoke_change_60m": 45.0,

    "temperature_change_5m": 0.8,
    "temperature_change_15m": 1.5,
    "temperature_change_30m": 2.5,
    "temperature_change_60m": 4.0,

    "humidity_change_5m": -1.0,
    "humidity_change_15m": -3.0,
    "humidity_change_30m": -5.0,

    "smoke_mean_5m": 82.0,
    "smoke_mean_15m": 76.0,
    "smoke_mean_30m": 68.0,
    "smoke_mean_60m": 55.0,

    "smoke_std_15m": 8.0,
    "smoke_std_30m": 12.0,

    "smoke_max_30m": 90.0,

    "temperature_mean_15m": 31.8,
    "temperature_mean_30m": 30.9,

    "humidity_mean_15m": 44.0,
    "humidity_mean_30m": 47.0,
}

result = engine.get_forecast(reading)

print()
print("=" * 60)
print("FIREGUARD FORECAST TEST")
print("=" * 60)

print(result)