
from __future__ import annotations

import sys
from pathlib import Path
from .pages.alerts import render as render_alerts
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# AUTO REFRESH
# ============================================================

REFRESH_SECONDS = 5

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# ============================================================
# PAGE IMPORTS
# Supports both package-relative and absolute imports
# ============================================================

def _load_page(module_name: str):
    try:
        module = __import__(
            f"dashboard.pages.{module_name}",
            fromlist=["render"],
        )
        return getattr(module, "render", None)
    except Exception:
        try:
            module = __import__(
                f"dashboard.pages.{module_name}",
                fromlist=["render"],
            )
            return getattr(module, "render", None)
        except Exception:
            return None


render_overview = _load_page("overview")
render_live = _load_page("live")
render_seasonal_forecast = _load_page("seasonal_forecast")
render_alerts = _load_page("alerts")
render_analytics = _load_page("analytics")

# Existing pages
render_events = _load_page("events")
render_forecast = _load_page("forecast")
render_nasa = _load_page("nasa")
render_weather = _load_page("weather")
render_verification = _load_page("verification")
render_models = _load_page("models")
render_health = _load_page("health")
render_database = _load_page("database")
render_hyrcanian = _load_page("hyrcanian")


# ============================================================
# CUSTOM UI
# ============================================================

def apply_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        div[data-testid="stMetric"] {
            border-radius: 10px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE FALLBACK
# ============================================================

def render_unavailable(page_name: str):
    st.warning(f"{page_name} در حال توسعه است.")


# ============================================================
# DASHBOARD
# ============================================================

def run_dashboard():

    st.set_page_config(
        page_title="FireGuard AI",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_styles()

    # --------------------------------------------------------
    # AUTO REFRESH
    # --------------------------------------------------------

    if st_autorefresh is not None:
        refresh_count = st_autorefresh(
            interval=REFRESH_SECONDS * 1000,
            key="fireguard_auto_refresh",
        )
    else:
        refresh_count = 0

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.title("🔥 FireGuard AI")

        st.caption(
            "سیستم پیش‌بینی آتش‌سوزی جنگل‌های هیرکانی"
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Live Monitoring",
                "Seasonal Forecast",
                "Alerts",
                "Analytics",
                "Fire Events",
                "Forecast",
                "NASA",
                "Hyrcanian Historical Evidence",
                "Weather",
                "Verification",
                "Model Status",
                "System Health",
                "Database",
            ],
        )

        st.divider()

        if st_autorefresh is not None:
            st.success(
                f"Auto-Refresh: {REFRESH_SECONDS}s"
            )
            st.caption(
                f"Refresh cycle: {refresh_count}"
            )
        else:
            st.info(
                "Auto-Refresh unavailable"
            )

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    pages = {
        "Overview": render_overview,
        "Live Monitoring": render_live,
        "Seasonal Forecast": render_seasonal_forecast,
        "Alerts": render_alerts,
        "Analytics": render_analytics,
        "Fire Events": render_events,
        "Forecast": render_forecast,
        "NASA": render_nasa,
        "Hyrcanian Historical Evidence": render_hyrcanian,
        "Weather": render_weather,
        "Verification": render_verification,
        "Model Status": render_models,
        "System Health": render_health,
        "Database": render_database,
    }

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    renderer = pages.get(page)

    if renderer is None:
        render_unavailable(page)
        return

    try:
        renderer()
    except Exception as exc:
        st.error("PAGE ERROR")
        st.exception(exc)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_dashboard()
