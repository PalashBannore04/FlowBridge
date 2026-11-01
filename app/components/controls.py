"""
Dashboard Controls and Scenario Parameter Sliders.
Encapsulates scenario parameters, supply sliders, SCADA configuration,
and interactive Light/Dark theme switching.
"""

from typing import Dict, Any, List, Optional
import os
import streamlit as st


def render_sidebar_controls(
    available_zones: List[str],
    current_demand_sum: Optional[float] = None,
    min_supply: Optional[float] = None,
    max_supply: Optional[float] = None,
    default_supply: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Renders sidebar scenario controls with theme toggle, dynamic bounds, and credentials manager.
    """
    demand_sum = current_demand_sum if current_demand_sum is not None else 3800.0
    min_s = min_supply if min_supply is not None else max(500.0, demand_sum * 0.4)
    max_s = max_supply if max_supply is not None else demand_sum * 1.6
    def_s = default_supply if default_supply is not None else demand_sum * 0.92

    curr_theme = st.session_state.get("theme", "light")
    is_lt = (curr_theme == "light")
    hdr_sub = "#505770" if is_lt else "#C4B5FD"
    hdr_title = "#5B61EB" if is_lt else "#C084FC"
    hdr_border = "#E4E7F4" if is_lt else "#271D45"

    st.sidebar.markdown(f"""
        <div style="padding: 4px 0 14px 0; border-bottom: 1px solid {hdr_border}; margin-bottom: 14px;">
            <div style="font-size: 0.70rem; letter-spacing: 0.12em; text-transform: uppercase; color: {hdr_sub}; font-weight: 700; font-family: var(--font-mono);">
                MUNICIPAL WATER DISTRICT SCADA
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: {hdr_title}; letter-spacing: -0.015em; font-family: var(--font-display); margin: 2px 0;">
                FlowBridge Operations
            </div>
            <div style="font-size: 0.72rem; color: {hdr_sub}; font-family: var(--font-mono);">
                v2.4.0-PROD | NODE-04A
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Theme Toggle
    st.sidebar.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:{hdr_sub}; margin-bottom:4px; letter-spacing:0.06em; text-transform:uppercase;'>Display Mode</div>", unsafe_allow_html=True)
    theme_choice = st.sidebar.radio(
        "Theme Mode",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state.get("theme", "light") == "light" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    theme_str = "light" if "Light" in theme_choice else "dark"
    st.session_state["theme"] = theme_str

    st.sidebar.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # 2. Supply Control with Dynamic Bounds
    st.sidebar.markdown("<div style='font-size:0.80rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:4px;'>Reservoir Dispatch Control</div>", unsafe_allow_html=True)
    st.sidebar.caption(f"Measured System Baseline: **{demand_sum:,.1f} kL/h**")

    total_supply = st.sidebar.slider(
        "Available Supply Volume (kL/h)",
        min_value=float(min_s),
        max_value=float(max_s),
        value=float(def_s),
        step=25.0,
        help="Adjust total reservoir volume dispatched into the municipal network to evaluate LP redistribution under varying supply deficits."
    )

    supply_stress_pct = (total_supply / max(demand_sum, 1.0)) * 100.0
    if supply_stress_pct < 90.0:
        st.sidebar.markdown(f"""
            <div style="background: var(--status-crit-bg, #FFE4E6); border: 1px solid var(--status-crit, #E11D48); border-radius: 6px; padding: 7px 10px; font-size: 0.78rem; color: var(--status-crit, #E11D48); margin-bottom: 12px; font-weight: 500;">
                <b>DEFICIT STATE:</b> Dispatching {supply_stress_pct:.1f}% of total demand. LP solver enforces priority constraints.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
            <div style="background: var(--status-good-bg, #E6F7F0); border: 1px solid var(--status-good, #059669); border-radius: 6px; padding: 7px 10px; font-size: 0.78rem; color: var(--status-good, #059669); margin-bottom: 12px; font-weight: 500;">
                <b>NOMINAL STATE:</b> Dispatching {supply_stress_pct:.1f}% of total demand.
            </div>
        """, unsafe_allow_html=True)

    # 3. Forecasting Horizon Selection
    st.sidebar.markdown("<div style='font-size:0.80rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin: 12px 0 4px 0;'>Forecast Horizon</div>", unsafe_allow_html=True)
    horizon = st.sidebar.selectbox(
        "Prediction Window",
        options=["1h", "6h", "24h"],
        format_func=lambda h: {
            "1h": "Tactical (1 Hour Horizon)",
            "6h": "Operational (6 Hour Window)",
            "24h": "Daily Strategic (24 Hours)"
        }.get(h, h),
        index=0,
        label_visibility="collapsed"
    )

    # 4. Weather Simulation Sliders
    st.sidebar.markdown("<div style='font-size:0.80rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin: 14px 0 4px 0;'>Meteorological Sensitivity</div>", unsafe_allow_html=True)
    temp_adjustment = st.sidebar.slider(
        "Temperature Offset (°C)",
        min_value=-5.0,
        max_value=8.0,
        value=0.0,
        step=0.5,
        help="Simulate seasonal temperature extremes to analyze consumption elasticity."
    )

    rain_simulation = st.sidebar.selectbox(
        "Precipitation Event",
        options=["Nominal Weather", "Dry Spell / Drought", "Heavy Rainfall (30mm)"],
        index=0
    )

    # 5. Zone Filter
    st.sidebar.markdown("<div style='font-size:0.80rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin: 14px 0 4px 0;'>Zone Focus Filter</div>", unsafe_allow_html=True)
    zone_filter = st.sidebar.selectbox(
        "Municipal District",
        options=["All Zones"] + available_zones,
        index=0,
        label_visibility="collapsed"
    )

    # 6. Credentials Management Expander
    st.sidebar.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    with st.sidebar.expander("API & Integration Settings", expanded=False):
        st.markdown("<div style='font-size:0.75rem; color:#9CA3AF;'>External data feeds and credentials:</div>", unsafe_allow_html=True)
        open_meteo_key = st.text_input(
            "Open-Meteo Key (Optional)",
            value=os.getenv("OPENMETEO_API_KEY", ""),
            type="password"
        )
        open_weather_key = st.text_input(
            "OpenWeatherMap Key (Optional)",
            value=os.getenv("OPENWEATHER_API_KEY", ""),
            type="password"
        )
        kaggle_user = st.text_input(
            "Kaggle Username (Optional)",
            value=os.getenv("KAGGLE_USERNAME", "")
        )

        if st.button("Save Configuration", use_container_width=True):
            env_content = f"""# FlowBridge Environment Configuration
OPENMETEO_API_KEY={open_meteo_key}
OPENWEATHER_API_KEY={open_weather_key}
KAGGLE_USERNAME={kaggle_user}
MUNICIPAL_DISTRICT_NAME="Neo-Veridia Metropolitan Water Authority"
CITY_LATITUDE=12.9716
CITY_LONGITUDE=77.5946
"""
            with open(".env", "w", encoding="utf-8") as f:
                f.write(env_content)
            st.success("Configuration persisted to .env")

    return {
        "total_supply": total_supply,
        "horizon": horizon,
        "temp_adjustment": temp_adjustment,
        "rain_simulation": rain_simulation,
        "zone_filter": zone_filter,
        "theme": theme_str
    }
