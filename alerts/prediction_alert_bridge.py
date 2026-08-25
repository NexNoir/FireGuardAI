from __future__ import annotations

from typing import Any

from alert_engine.alert_engine import AlertInput
from alerts.alert_service import AlertService
from forecast.real_forecast_v4 import FireGuardForecastV4


class PredictionAlertBridge:
    """
    Connects real FireGuard V4 forecast output
    to the existing AlertEngine + SMS pipeline.

    Model probability is read-only.
    NASA / Weather never modify model probability.
    """

    def __init__(
        self,
        forecast_engine: FireGuardForecastV4 | None = None,
        alert_service: AlertService | None = None,
    ):
        self.forecast_engine = (
            forecast_engine
            or FireGuardForecastV4(
                experiment="sensor_plus_flame"
            )
        )

        self.alert_service = (
            alert_service
            or AlertService()
        )

    def evaluate(
        self,
        history: list[dict[str, Any]],
        *,
        horizon: str = "72h",
        sensor_quality: float | None = 1.0,
        nasa_evidence: bool | None = None,
        weather_risk: float | None = None,
        forecast_risk: float | None = None,
        uncertainty: float | None = None,
        source: str = "ESP32 + FireGuard V4",
        persist_alert: bool = True,
    ) -> dict[str, Any]:

        if not history:
            return {
                "success": False,
                "error": "No real sensor history available",
                "forecast": None,
                "alert": None,
            }

        # ----------------------------------------------------
        # Real V4 forecast
        # ----------------------------------------------------

        forecast_result = (
            self.forecast_engine
            .get_forecast_from_history(
                history
            )
        )
        print("\n" + "=" * 70)
        print("FIREGUARD V4 - ALL FORECAST HORIZONS")
        print("=" * 70)
        print("SUCCESS:", forecast_result.get("success"))
        print("EXPERIMENT:", forecast_result.get("experiment"))
        print("FORECAST KEYS:", list(
            forecast_result.get("forecast", {}).keys()
        ))
        
        for horizon_name, result in forecast_result.get(
            "forecast", {}
        ).items():
            print(f"\n[{horizon_name}]")
            print(result)
        
        print("=" * 70 + "\n")

        if not forecast_result.get(
            "success",
            False,
        ):
            return {
                "success": False,
                "error": forecast_result.get(
                    "error",
                    "Forecast failed",
                ),
                "forecast": forecast_result,
                "alert": None,
            }

        forecasts = forecast_result.get(
            "forecast",
            {},
        )

        selected = forecasts.get(
            horizon
        )

        if selected is None:
            return {
                "success": False,
                "error": (
                    f"Forecast horizon unavailable: "
                    f"{horizon}"
                ),
                "forecast": forecast_result,
                "alert": None,
            }

        # ----------------------------------------------------
        # Latest REAL sensor record
        # ----------------------------------------------------

        latest = history[-1]

        flame = latest.get(
            "flame",
            0,
        )

        temperature = latest.get(
            "temperature"
        )

        humidity = latest.get(
            "humidity"
        )

        smoke = latest.get(
            "smoke"
        )

        timestamp = latest.get(
            "timestamp"
        )

        # ----------------------------------------------------
        # Alert input
        # ----------------------------------------------------

        alert_input = AlertInput(
            fire_probability=selected.get(
                "fire_probability"
            ),
            flame=flame,
            smoke_trend=None,
            sensor_quality=sensor_quality,
            nasa_evidence=nasa_evidence,
            weather_risk=weather_risk,
            forecast_risk=forecast_risk,
            uncertainty=uncertainty,

            # context only
            temperature=temperature,
            humidity=humidity,
            smoke=smoke,

            timestamp=str(
                timestamp
            ),

            source=source,
        )

        # ----------------------------------------------------
        # Alert + database + cooldown + SMS
        # ----------------------------------------------------

        alert_result = (
            self.alert_service
            .evaluate_and_notify(
                alert_input,
                persist_alert=persist_alert,
            )
        )

        return {
            "success": True,
            "horizon": horizon,
            "forecast": selected,
            "alert": alert_result,
        }