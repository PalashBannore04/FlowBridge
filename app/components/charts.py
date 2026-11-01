"""
Plotly Chart Visualizations for Municipal Water SCADA.
Provides responsive, high-fidelity engineering curves supporting both Light and Dark themes.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_theme_palette(theme: str = "light") -> Dict[str, str]:
    """Returns aesthetic color tokens for pastel light or purplish-black dark modes."""
    if theme == "dark":
        return {
            "plot_bg": "#140F24",
            "paper_bg": "#0B0813",
            "grid_color": "#271D45",
            "text_color": "#F5F3FF",
            "muted_color": "#8C7FA8",
            "accent_blue": "#9333EA",
            "accent_cyan": "#C084FC",
            "accent_green": "#34D399",
            "accent_amber": "#FBBF24",
            "accent_red": "#FB7185"
        }
    else:
        return {
            "plot_bg": "#FFFFFF",
            "paper_bg": "#F7F8FC",
            "grid_color": "#E4E7F4",
            "text_color": "#1E2238",
            "muted_color": "#7E86A2",
            "accent_blue": "#5B61EB",
            "accent_cyan": "#38BDF8",
            "accent_green": "#059669",
            "accent_amber": "#D97706",
            "accent_red": "#E11D48"
        }


def plot_historical_vs_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    zone_name: str = "Aggregate Grid",
    horizon: str = "1h",
    theme: str = "dark"
) -> go.Figure:
    """Renders time-series comparing historical observed demand against upcoming forecasted demand."""
    pal = get_theme_palette(theme)
    fig = go.Figure()

    # Observed actual demand
    fig.add_trace(go.Scatter(
        x=historical_df["timestamp"],
        y=historical_df["actual_consumption_kL"],
        mode="lines",
        name="Observed Flow (kL/h)",
        line=dict(color=pal["accent_cyan"], width=2.0),
        hovertemplate="%{x|%b %d, %H:%M}<br>Actual: <b>%{y:.1f} kL</b><extra></extra>"
    ))

    # Historical supply baseline
    if "allocated_supply_kL" in historical_df.columns:
        fig.add_trace(go.Scatter(
            x=historical_df["timestamp"],
            y=historical_df["allocated_supply_kL"],
            mode="lines",
            name="Allocated Supply",
            line=dict(color=pal["muted_color"], width=1.5, dash="dash"),
            opacity=0.75,
            hovertemplate="%{x|%b %d, %H:%M}<br>Allocated: <b>%{y:.1f} kL</b><extra></extra>"
        ))

    # Forecast curve
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_demand_kL"],
            mode="lines+markers",
            name=f"Predicted Demand ({horizon.upper()})",
            line=dict(color=pal["accent_green"], width=2.8),
            marker=dict(size=5, color=pal["accent_green"]),
            hovertemplate="%{x|%b %d, %H:%M}<br>Forecast: <b>%{y:.1f} kL</b><extra></extra>"
        ))

        # 95% Confidence envelope
        upper = forecast_df["predicted_demand_kL"] * 1.05
        lower = forecast_df["predicted_demand_kL"] * 0.95

        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(52, 211, 153, 0.16)" if theme == "dark" else "rgba(5, 150, 105, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Interval"
        ))

    fig.update_layout(
        title=f"<b>Water Consumption & Forecast Horizon</b> — {zone_name}",
        title_font=dict(size=14, color=pal["text_color"]),
        plot_bgcolor=pal["plot_bg"],
        paper_bgcolor=pal["paper_bg"],
        font=dict(color=pal["text_color"], family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=pal["grid_color"], showgrid=True, title="Timestamp"),
        yaxis=dict(gridcolor=pal["grid_color"], showgrid=True, title="Flow Rate (kL/hour)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=45, b=15),
        hovermode="x unified"
    )
    return fig


def plot_redistribution_breakdown(breakdown_df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Grouped bar chart: Predicted Demand vs Allocated Supply vs Shortage Deficit."""
    pal = get_theme_palette(theme)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=breakdown_df["zone_id"],
        y=breakdown_df["predicted_demand_kL"],
        name="Predicted Demand",
        marker_color=pal["accent_cyan"],
        hovertemplate="%{x}<br>Demand: <b>%{y:.1f} kL</b><extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=breakdown_df["zone_id"],
        y=breakdown_df["allocated_supply_kL"],
        name="LP Allocated Supply",
        marker_color=pal["accent_blue"],
        hovertemplate="%{x}<br>Allocated: <b>%{y:.1f} kL</b><extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=breakdown_df["zone_id"],
        y=breakdown_df["shortage_deficit_kL"],
        name="Shortage Deficit",
        marker_color=pal["accent_red"],
        hovertemplate="%{x}<br>Deficit: <b>%{y:.1f} kL</b><extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        title="<b>District Demand Satisfaction & Shortage Margins</b>",
        title_font=dict(size=14, color=pal["text_color"]),
        plot_bgcolor=pal["plot_bg"],
        paper_bgcolor=pal["paper_bg"],
        font=dict(color=pal["text_color"], family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=pal["grid_color"], title="Municipal District"),
        yaxis=dict(gridcolor=pal["grid_color"], title="Water Volume (kL/hour)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=45, b=15)
    )
    return fig


def plot_anomaly_detection_timeline(anomaly_df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Visualizes telemetry flow stream and flags statistical & physical burst anomalies."""
    pal = get_theme_palette(theme)
    fig = go.Figure()

    # Flow Stream
    fig.add_trace(go.Scatter(
        x=anomaly_df["timestamp"],
        y=anomaly_df["actual_consumption_kL"],
        mode="lines",
        name="Telemetry Stream (kL/h)",
        line=dict(color=pal["muted_color"], width=1.5),
        opacity=0.75
    ))

    # Flagged Anomalies
    anom_points = anomaly_df[anomaly_df["is_anomaly_detected"] == 1]
    if not anom_points.empty:
        fig.add_trace(go.Scatter(
            x=anom_points["timestamp"],
            y=anom_points["actual_consumption_kL"],
            mode="markers",
            name="Detected Main Rupture / Leak",
            marker=dict(
                color=pal["accent_red"],
                size=9,
                symbol="circle-open-dot",
                line=dict(width=2.5, color=pal["accent_red"])
            ),
            text=anom_points["anomaly_classification"],
            hovertemplate="<b>ALERT: %{text}</b><br>Time: %{x|%b %d, %H:%M}<br>Flow: <b>%{y:.1f} kL/h</b><extra></extra>"
        ))

    fig.update_layout(
        title="<b>Hydraulic Flow Stream & Automated Rupture Flags</b>",
        title_font=dict(size=14, color=pal["text_color"]),
        plot_bgcolor=pal["plot_bg"],
        paper_bgcolor=pal["paper_bg"],
        font=dict(color=pal["text_color"], family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=pal["grid_color"], title="Timeline"),
        yaxis=dict(gridcolor=pal["grid_color"], title="Flow Rate (kL/hour)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=45, b=15)
    )
    return fig


def plot_weather_elasticity(df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """
    Plots correlation between ambient temperature and water consumption.
    Uses numpy polyfit for resilient trendline computation without relying on external statsmodels.
    """
    pal = get_theme_palette(theme)
    sample_df = df.sample(min(len(df), 1200), random_state=42).sort_values(by="temperature_c")

    x_vals = sample_df["temperature_c"].values
    y_vals = sample_df["actual_consumption_kL"].values

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        name="Telemetry Observations",
        marker=dict(
            size=6,
            color=sample_df["humidity_pct"],
            colorscale="Purples" if theme == "dark" else "Blues",
            colorbar=dict(title=dict(text="Humidity (%)", font=dict(color=pal["text_color"])), tickfont=dict(color=pal["text_color"])),
            opacity=0.75
        ),
        hovertemplate="Temp: %{x:.1f}°C<br>Consumption: <b>%{y:.1f} kL/h</b><extra></extra>"
    ))

    # OLS Regression line using numpy
    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_trend = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_trend = slope * x_trend + intercept

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            name=f"OLS Trendline (+{slope:.2f} kL/°C)",
            line=dict(color=pal["accent_red"], width=2.5, dash="dash")
        ))

    fig.update_layout(
        title="<b>Meteorological Elasticity: Temperature vs Water Demand</b>",
        title_font=dict(size=14, color=pal["text_color"]),
        plot_bgcolor=pal["plot_bg"],
        paper_bgcolor=pal["paper_bg"],
        font=dict(color=pal["text_color"], family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=pal["grid_color"], title="Ambient Temperature (°C)"),
        yaxis=dict(gridcolor=pal["grid_color"], title="Hourly Consumption (kL/h)"),
        margin=dict(l=15, r=15, t=45, b=15)
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Displays top predictive driver rankings from XGBoost."""
    pal = get_theme_palette(theme)
    top_df = importance_df.head(10).sort_values(by="importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_df["normalized_pct"],
        y=top_df["feature"],
        orientation="h",
        marker=dict(color=pal["accent_cyan"]),
        hovertemplate="%{y}: <b>%{x:.2f}%</b> impact<extra></extra>"
    ))

    fig.update_layout(
        title="<b>XGBoost Feature Importance Driver Rankings</b>",
        title_font=dict(size=14, color=pal["text_color"]),
        plot_bgcolor=pal["plot_bg"],
        paper_bgcolor=pal["paper_bg"],
        font=dict(color=pal["text_color"], family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=pal["grid_color"], title="Relative Predictive Contribution (%)"),
        yaxis=dict(gridcolor=pal["grid_color"]),
        margin=dict(l=15, r=15, t=45, b=15)
    )
    return fig
