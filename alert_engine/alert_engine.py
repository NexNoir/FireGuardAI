
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import IntEnum
from hashlib import sha256
from typing import Any, Optional


class AlertLevel(IntEnum):
    INFO = 0
    WATCH = 1
    WARNING = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AlertInput:
    fire_probability: Optional[float] = None
    flame: Optional[float] = None
    smoke_trend: Optional[float] = None
    sensor_quality: Optional[float] = None
    nasa_evidence: Optional[bool] = None
    weather_risk: Optional[float] = None
    forecast_risk: Optional[float] = None
    uncertainty: Optional[float] = None

    # Context only:
    # These fields are for SMS/logging/UI.
    # They MUST NOT modify fire_probability.
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    smoke: Optional[float] = None

    timestamp: Optional[str] = None
    source: str = "system"


@dataclass
class AlertResult:
    event_id: str
    level: str
    level_value: int
    reasons: list[str]
    fire_probability: Optional[float]
    probability_changed: bool
    created_at: str
    deduplication_key: str
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertEngine:
    """
    Safe Alert Engine.

    IMPORTANT:
    - fire_probability is read-only.
    - Evidence never modifies fire_probability.
    - No SMS is sent here.
    - Alert decision is separate from ML prediction.
    """

    def __init__(
        self,
        warning_probability: float = 0.50,
        high_probability: float = 0.70,
        critical_probability: float = 0.85,
    ):
        self.warning_probability = warning_probability
        self.high_probability = high_probability
        self.critical_probability = critical_probability

    @staticmethod
    def _clamp_probability(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None

        value = float(value)

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"fire_probability must be between 0 and 1, got {value}"
            )

        return value

    @staticmethod
    def _normalise_score(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None

        value = float(value)

        # Scores are evidence inputs, not probabilities.
        return max(0.0, min(1.0, value))

    @staticmethod
    def _make_deduplication_key(
        data: AlertInput,
        level: AlertLevel,
    ) -> str:
        timestamp = data.timestamp or ""

        raw = "|".join(
            [
                timestamp,
                str(level.name),
                str(data.flame),
                str(data.smoke_trend),
                str(data.sensor_quality),
                str(data.nasa_evidence),
                str(data.weather_risk),
                str(data.forecast_risk),
            ]
        )

        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _make_event_id(deduplication_key: str) -> str:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%d%H%M%S")

        return f"ALR-{stamp}-{deduplication_key[:8]}"

    def evaluate(self, data: AlertInput) -> AlertResult:
        probability = self._clamp_probability(data.fire_probability)

        smoke_trend = self._normalise_score(data.smoke_trend)
        sensor_quality = self._normalise_score(data.sensor_quality)
        weather_risk = self._normalise_score(data.weather_risk)
        forecast_risk = self._normalise_score(data.forecast_risk)
        uncertainty = self._normalise_score(data.uncertainty)

        reasons: list[str] = []

        # ------------------------------------------------------------
        # 1. BASE LEVEL FROM ML PROBABILITY
        # ------------------------------------------------------------
        #
        # This is the ONLY place where model probability itself is used.
        # Evidence below NEVER changes probability.
        #
        level = AlertLevel.INFO

        if probability is not None:
            if probability >= self.critical_probability:
                level = max(level, AlertLevel.CRITICAL)
                reasons.append(
                    f"fire_probability >= {self.critical_probability:.2f}"
                )
            elif probability >= self.high_probability:
                level = max(level, AlertLevel.HIGH)
                reasons.append(
                    f"fire_probability >= {self.high_probability:.2f}"
                )
            elif probability >= self.warning_probability:
                level = max(level, AlertLevel.WARNING)
                reasons.append(
                    f"fire_probability >= {self.warning_probability:.2f}"
                )
            elif probability >= 0.25:
                level = max(level, AlertLevel.WATCH)
                reasons.append("elevated fire_probability")

        # ------------------------------------------------------------
        # 2. FLAME EVIDENCE
        # ------------------------------------------------------------
        #
        # Flame can increase ALERT LEVEL.
        # It never modifies probability.
        #
        if data.flame is not None and float(data.flame) >= 1:
            level = max(level, AlertLevel.CRITICAL)
            reasons.append("flame detected")

        # ------------------------------------------------------------
        # 3. SMOKE TREND
        # ------------------------------------------------------------
        if smoke_trend is not None:
            if smoke_trend >= 0.85:
                level = max(level, AlertLevel.HIGH)
                reasons.append("very strong smoke trend")
            elif smoke_trend >= 0.60:
                level = max(level, AlertLevel.WARNING)
                reasons.append("elevated smoke trend")
            elif smoke_trend >= 0.35:
                level = max(level, AlertLevel.WATCH)
                reasons.append("increasing smoke trend")

        # ------------------------------------------------------------
        # 4. SENSOR QUALITY
        # ------------------------------------------------------------
        #
        # Poor sensor quality should not create a fire probability.
        # It can, however, reduce alert severity because evidence is weak.
        #
        if sensor_quality is not None:
            if sensor_quality < 0.30:
                if level > AlertLevel.WATCH:
                    level = AlertLevel.WATCH

                reasons.append("low sensor quality")

        # ------------------------------------------------------------
        # 5. EXTERNAL EVIDENCE
        # ------------------------------------------------------------
        #
        # These values affect alert level only.
        #
        if data.nasa_evidence is True:
            level = max(level, AlertLevel.WATCH)
            reasons.append("NASA evidence present")

        if weather_risk is not None:
            if weather_risk >= 0.85:
                level = max(level, AlertLevel.HIGH)
                reasons.append("high weather risk")
            elif weather_risk >= 0.60:
                level = max(level, AlertLevel.WATCH)
                reasons.append("elevated weather risk")

        if forecast_risk is not None:
            if forecast_risk >= 0.90:
                level = max(level, AlertLevel.HIGH)
                reasons.append("high forecast risk")
            elif forecast_risk >= 0.70:
                level = max(level, AlertLevel.WARNING)
                reasons.append("elevated forecast risk")

        # ------------------------------------------------------------
        # 6. UNCERTAINTY
        # ------------------------------------------------------------
        #
        # Here uncertainty is treated as a warning/context signal.
        # It does NOT modify probability.
        #
        if uncertainty is not None and uncertainty >= 0.70:
            reasons.append("high uncertainty")

            if level >= AlertLevel.HIGH:
                reasons.append("high uncertainty requires review")

        # ------------------------------------------------------------
        # 7. NO EVIDENCE
        # ------------------------------------------------------------
        if not reasons:
            reasons.append("no alert condition detected")

        dedup_key = self._make_deduplication_key(data, level)
        event_id = self._make_event_id(dedup_key)

        created_at = datetime.now(timezone.utc).isoformat()

        return AlertResult(
            event_id=event_id,
            level=level.name,
            level_value=int(level),
            reasons=reasons,
            fire_probability=probability,
            probability_changed=False,
            created_at=created_at,
            deduplication_key=dedup_key,
            acknowledged=False,
            resolved=False,
        )

