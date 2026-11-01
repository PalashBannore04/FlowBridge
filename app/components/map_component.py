"""
Geospatial Map Component using Folium and Streamlit-Folium.
Renders interactive choropleth polygons and status markers color-coded
by shortage severity: Green (Adequate), Yellow (Warning), Red (Critical).
"""

from typing import Dict, Any, List, Optional
import os
import json
import folium
from folium import plugins
import pandas as pd


# Default Centroids for the 6 SE Smart City Zones
ZONE_CENTROIDS: Dict[str, List[float]] = {
    "ZONE_1": [12.9750, 77.6000],  # CBD
    "ZONE_2": [13.0000, 77.5950],  # North Residential
    "ZONE_3": [12.9775, 77.6350],  # East Tech / Industrial
    "ZONE_4": [12.9475, 77.6000],  # South Medical & University
    "ZONE_5": [12.9650, 77.5650],  # West Heritage / Mixed
    "ZONE_6": [13.0100, 77.6375],  # Northeast High-Rise
}


def create_city_water_map(
    zone_metrics_df: pd.DataFrame,
    geojson_path: str = "data/zones_geojson.json",
    transfer_routes_df: Optional[pd.DataFrame] = None,
    anomaly_status_map: Optional[Dict[str, str]] = None,
    theme: str = "light"
) -> folium.Map:
    """
    Renders interactive Folium map with polygon boundaries and status markers.
    Adapts tiles and aesthetics for Pastel Light and Purplish-Black Dark themes.
    """
    # City Center Coordinates
    city_center = [12.9800, 77.6000]
    m = folium.Map(
        location=city_center,
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Theme-tailored map styling without external API key constraints
    if theme == "dark":
        map_css = """
        <style>
        .leaflet-container {
            background-color: #0B0813 !important;
        }
        .leaflet-tile {
            filter: brightness(0.65) invert(1) contrast(2) hue-rotate(225deg) saturate(1.2) !important;
        }
        .leaflet-bar a {
            background-color: #140F24 !important;
            color: #F5F3FF !important;
            border-bottom: 1px solid #271D45 !important;
        }
        .leaflet-bar a:hover {
            background-color: #1D1635 !important;
            color: #C084FC !important;
        }
        .leaflet-control-attribution {
            background: rgba(11, 8, 19, 0.85) !important;
            color: #8C7FA8 !important;
        }
        .leaflet-control-attribution a {
            color: #C084FC !important;
        }
        </style>
        """
    else:
        map_css = """
        <style>
        .leaflet-container {
            background-color: #F7F8FC !important;
        }
        .leaflet-tile {
            filter: saturate(0.65) brightness(1.04) contrast(0.92) !important;
        }
        .leaflet-bar a {
            background-color: #FFFFFF !important;
            color: #1E2238 !important;
            border-bottom: 1px solid #E4E7F4 !important;
        }
        .leaflet-bar a:hover {
            background-color: #F1F3FB !important;
            color: #5B61EB !important;
        }
        .leaflet-control-attribution {
            background: rgba(247, 248, 252, 0.85) !important;
            color: #7E86A2 !important;
        }
        .leaflet-control-attribution a {
            color: #5B61EB !important;
        }
        </style>
        """
    m.get_root().html.add_child(folium.Element(map_css))

    # Index metrics by zone_id for O(1) lookup
    metrics_by_zone = {}
    if not zone_metrics_df.empty:
        for _, row in zone_metrics_df.iterrows():
            metrics_by_zone[str(row["zone_id"])] = row.to_dict()

    # Load and Render GeoJSON Polygons if file exists
    if os.path.exists(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        def style_function(feature):
            zid = feature["properties"].get("zone_id", "")
            zone_info = metrics_by_zone.get(zid, {})
            risk = zone_info.get("risk_status", "Adequate")

            if risk == "Critical Shortage":
                fill_color = "#FB7185" if theme == "dark" else "#E11D48"
                fill_opacity = 0.50
            elif risk == "Warning":
                fill_color = "#FBBF24" if theme == "dark" else "#D97706"
                fill_opacity = 0.40
            else:
                fill_color = "#34D399" if theme == "dark" else "#059669"
                fill_opacity = 0.35

            border_color = "#C084FC" if theme == "dark" else "#5B61EB"
            return {
                "fillColor": fill_color,
                "color": border_color,
                "weight": 2.2,
                "fillOpacity": fill_opacity,
                "dashArray": "3, 6" if risk == "Critical Shortage" else None
            }

        folium.GeoJson(
            geojson_data,
            name="Municipal Water Zones",
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["zone_name", "category", "base_population"],
                aliases=["Zone:", "Category:", "Population:"],
                localize=True
            )
        ).add_to(m)

    # Render Centroid Diagnostic Markers
    for zid, coords in ZONE_CENTROIDS.items():
        z_data = metrics_by_zone.get(zid, {})
        z_name = z_data.get("zone_name", zid)
        risk = z_data.get("risk_status", "Adequate")
        dem = z_data.get("predicted_demand_kL", 0.0)
        alloc = z_data.get("allocated_supply_kL", 0.0)
        defic = z_data.get("shortage_deficit_kL", 0.0)
        defic_pct = z_data.get("deficit_pct", 0.0)
        prio = z_data.get("priority_level", 3)
        anom_alert = anomaly_status_map.get(zid, "NORMAL") if anomaly_status_map else "NORMAL"

        # Marker color & icon
        if risk == "Critical Shortage":
            icon_color = "red"
            circle_color = "#ef4444"
            badge_html = '<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:10px;font-weight:bold;font-size:11px;">CRITICAL SHORTAGE</span>'
        elif risk == "Warning":
            icon_color = "orange"
            circle_color = "#f59e0b"
            badge_html = '<span style="background:#f59e0b;color:black;padding:2px 8px;border-radius:10px;font-weight:bold;font-size:11px;">SUPPLY WARNING</span>'
        else:
            icon_color = "green"
            circle_color = "#10b981"
            badge_html = '<span style="background:#10b981;color:white;padding:2px 8px;border-radius:10px;font-weight:bold;font-size:11px;">ADEQUATE SUPPLY</span>'

        anomaly_badge = ""
        if anom_alert != "NORMAL":
            anomaly_badge = f'<div style="margin-top:6px;background:rgba(239,68,68,0.2);border:1px solid #ef4444;padding:4px 8px;border-radius:6px;color:#fca5a5;font-size:11px;">🚨 <b>ALERT:</b> {anom_alert}</div>'

        popup_html = f"""
        <div style="font-family:'Plus Jakarta Sans',sans-serif; min-width:210px; padding:6px; color:#1e293b;">
            <div style="font-size:14px; font-weight:700; margin-bottom:4px; color:#0f172a;">{z_name}</div>
            <div style="margin-bottom:8px;">{badge_html}</div>
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
                <tr><td style="color:#64748b; padding:2px 0;">Priority:</td><td style="font-weight:600; text-align:right;">Tier {prio}</td></tr>
                <tr><td style="color:#64748b; padding:2px 0;">Predicted Demand:</td><td style="font-weight:600; text-align:right;">{dem:,.1f} kL/h</td></tr>
                <tr><td style="color:#64748b; padding:2px 0;">Allocated Supply:</td><td style="font-weight:600; text-align:right; color:#5b61eb;">{alloc:,.1f} kL/h</td></tr>
                <tr><td style="color:#64748b; padding:2px 0;">Deficit Gap:</td><td style="font-weight:700; text-align:right; color:{circle_color};">{defic:,.1f} kL/h ({defic_pct}%)</td></tr>
            </table>
            {anomaly_badge}
        </div>
        """

        # Outer Pulsing Circle
        folium.CircleMarker(
            location=coords,
            radius=18,
            color=circle_color,
            weight=2,
            fill=True,
            fill_color=circle_color,
            fill_opacity=0.35
        ).add_to(m)

        # Centroid Icon Marker
        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{z_name} | {risk}",
            icon=folium.Icon(color=icon_color, icon="tint", prefix="fa")
        ).add_to(m)

    # Render Inter-Zone Redistribution Transfer Vectors
    if transfer_routes_df is not None and not transfer_routes_df.empty:
        for _, route in transfer_routes_df.iterrows():
            f_zone = route["from_zone"]
            t_zone = route["to_zone"]
            vol = route["transfer_volume_kL"]
            if f_zone in ZONE_CENTROIDS and t_zone in ZONE_CENTROIDS:
                start_pt = ZONE_CENTROIDS[f_zone]
                end_pt = ZONE_CENTROIDS[t_zone]

                route_color = "#C084FC" if theme == "dark" else "#5B61EB"
                folium.PolyLine(
                    locations=[start_pt, end_pt],
                    color=route_color,
                    weight=3.5,
                    opacity=0.85,
                    dash_array="6, 8",
                    tooltip=f"Redistribution Flow: {f_zone} ➔ {t_zone} ({vol:,.1f} kL/h)"
                ).add_to(m)

    # Map Controls
    folium.LayerControl().add_to(m)
    return m
