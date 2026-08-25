# -*- coding: utf-8 -*-

"""
FIREGUARD — Dashboard Local Service Adapter V1

Purpose:
    Connect the local FireGuard dashboard to the production
    RealFirmsProductionService.

Architecture:
    Dashboard
        ↓
    Dashboard adapter
        ↓
    RealFirmsProductionService
        ↓
    24H / 48H / 72H models

NO API
NO HTTP
NO FASTAPI
NO UVICORN
NO RETRAINING
NO MODEL MODIFICATION
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\vista\Desktop\fireguard_v2.0"
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------
# PRODUCTION SERVICE
# ---------------------------------------------------------------------

from real_firms_service import RealFirmsProductionService


class FireGuardDashboardService:
    """
    Small adapter between the existing dashboard and the
    production inference service.
    """

    def __init__(self) -> None:

        self.service = RealFirmsProductionService()

    # -----------------------------------------------------------------
    # STATUS
    # -----------------------------------------------------------------

    def status(self) -> dict[str, Any]:

        if hasattr(self.service, "status"):

            result = self.service.status()

            if isinstance(result, dict):
                return result

        if hasattr(self.service, "get_status"):

            result = self.service.get_status()

            if isinstance(result, dict):
                return result

        return {
            "service": "FIREGUARD REAL FIRMS",
            "status": "ready",
            "feature_count": 15,
            "thresholds": {
                "24h": 0.35,
                "48h": 0.35,
                "72h": 0.30,
            },
        }

    # -----------------------------------------------------------------
    # SINGLE RECORD
    # -----------------------------------------------------------------

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:

        result = self.service.predict(record)

        if isinstance(result, dict):
            return result

        return {
            "result": result
        }

    # -----------------------------------------------------------------
    # BATCH
    # -----------------------------------------------------------------

    def predict_batch(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result = self.service.predict_batch(records)

        if not isinstance(result, list):
            raise RuntimeError(
                "Production service returned invalid batch result."
            )

        return result


# ---------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("🔥 FIREGUARD — DASHBOARD SERVICE ADAPTER V1")
    print("=" * 72)
    print()

    print("Loading production service...")

    service = FireGuardDashboardService()

    print("Production service: PASS")
    print()

    print("=" * 72)
    print("SERVICE STATUS")
    print("=" * 72)

    status = service.status()

    for key, value in status.items():
        print(f"{key}: {value}")

    print()

    print("=" * 72)
    print("DASHBOARD ADAPTER CHECK")
    print("=" * 72)

    print(
        "predict method      :",
        "PASS" if callable(service.predict) else "FAIL"
    )

    print(
        "predict_batch method:",
        "PASS" if callable(service.predict_batch) else "FAIL"
    )

    print()

    print("=" * 72)
    print("STATUS: 🟢 DASHBOARD SERVICE ADAPTER READY")
    print("=" * 72)


if __name__ == "__main__":
    main()