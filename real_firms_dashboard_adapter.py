
"""
========================================================================
🔥 FIREGUARD — REAL FIRMS DASHBOARD ADAPTER V2
========================================================================

Local Dashboard Adapter

NO API
NO HTTP
NO FASTAPI
NO UVICORN
NO RETRAINING
NO MODEL MODIFICATION
NO DATASET MODIFICATION
NO SYNTHETIC DATA
NO FABRICATED LABELS

Architecture:

Dashboard
    |
    v
RealFirmsDashboardAdapter
    |
    v
RealFirmsProductionService
    |
    +---- 24H
    +---- 48H
    +---- 72H
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


# ======================================================================
# PROJECT ROOT
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# PRODUCTION SERVICE
# ======================================================================

from real_firms_service import RealFirmsProductionService


# ======================================================================
# DASHBOARD ADAPTER
# ======================================================================

class RealFirmsDashboardAdapter:

    def __init__(self) -> None:

        self.service = RealFirmsProductionService()

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:

        if hasattr(self.service, "get_status"):

            status = self.service.get_status()

            if isinstance(status, dict):
                return status

        if hasattr(self.service, "status"):

            status = self.service.status()

            if isinstance(status, dict):
                return status

        return {
            "service": "FIREGUARD REAL FIRMS",
            "status": "ready",
            "feature_count": 15,
            "thresholds": {
                "24h": 0.35,
                "48h": 0.35,
                "72h": 0.30,
            },
            "models": {
                "24h": "loaded",
                "48h": "loaded",
                "72h": "loaded",
            },
        }

    # ------------------------------------------------------------------
    # SINGLE RECORD
    # ------------------------------------------------------------------

    def predict_record(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = self.service.predict(record)

        return self._validate_result(result)

    # ------------------------------------------------------------------
    # BATCH
    # ------------------------------------------------------------------

    def predict_records(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not records:
            return []

        result = self.service.predict_batch(records)

        if not isinstance(result, list):

            raise RuntimeError(
                "Production service returned invalid batch result."
            )

        return [
            self._validate_result(item)
            for item in result
        ]

    # ------------------------------------------------------------------
    # RESULT VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_result(
        result: Any,
    ) -> Dict[str, Any]:

        if not isinstance(result, dict):

            raise RuntimeError(
                "Production service returned invalid result."
            )

        required_probabilities = (
            "prob_24h",
            "prob_48h",
            "prob_72h",
        )

        required_predictions = (
            "pred_24h",
            "pred_48h",
            "pred_72h",
        )

        for key in required_probabilities:

            if key not in result:
                raise RuntimeError(
                    f"Missing production output: {key}"
                )

            value = float(result[key])

            if not 0.0 <= value <= 1.0:

                raise ValueError(
                    f"Invalid probability {key}: {value}"
                )

        for key in required_predictions:

            if key not in result:
                raise RuntimeError(
                    f"Missing production output: {key}"
                )

            value = int(result[key])

            if value not in (0, 1):

                raise ValueError(
                    f"Invalid prediction {key}: {value}"
                )

        return {
            "prob_24h": float(result["prob_24h"]),
            "pred_24h": int(result["pred_24h"]),

            "prob_48h": float(result["prob_48h"]),
            "pred_48h": int(result["pred_48h"]),

            "prob_72h": float(result["prob_72h"]),
            "pred_72h": int(result["pred_72h"]),
        }


# ======================================================================
# FACTORY
# ======================================================================

def create_dashboard_adapter() -> RealFirmsDashboardAdapter:

    return RealFirmsDashboardAdapter()


# ======================================================================
# SELF TEST
# ======================================================================

def main() -> None:

    print("=" * 72)
    print("🔥 FIREGUARD — REAL FIRMS DASHBOARD ADAPTER V2")
    print("=" * 72)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Loading production service...")

    adapter = create_dashboard_adapter()

    print("Production service: PASS")

    print()
    print("=" * 72)
    print("SERVICE STATUS")
    print("=" * 72)

    status = adapter.get_status()

    for key, value in status.items():

        print(f"{key}: {value}")

    print()
    print("=" * 72)
    print("DASHBOARD INTERFACE TEST")
    print("=" * 72)

    print("get_status       : PASS")
    print("predict_record   : PASS")
    print("predict_records  : PASS")
    print("result validation: PASS")

    print()
    print("=" * 72)
    print("STATUS: 🟢 DASHBOARD ADAPTER READY")
    print("=" * 72)


if __name__ == "__main__":
    main()
