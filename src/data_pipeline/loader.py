"""
Data Loading and Generation Module for SE Smart City Water System.
Generates an authentic multi-zone smart water SCADA telemetry dataset
with dynamic rolling timestamps, hydraulic pressure, flow velocities,
water quality metrics (pH, turbidity, chlorine), and physical pipe burst events.
"""

from typing import Optional, List, Dict
import os
import json
import math
import numpy as np
import pandas as pd


# Default Operational Zones Specification
DEFAULT_ZONES: List[Dict] = [
    {
        "zone_id": "ZONE_1",
        "zone_name": "Central Business District",
        "category": "Commercial",
        "base_demand_kL": 520.0,
        "population": 85000,
        "priority_level": 3,
        "pipeline_capacity_kL": 1200.0,
        "nominal_pressure_bar": 4.2,
        "pump_station_id": "PS-101 (CBD Primary)",
    },
    {
        "zone_id": "ZONE_2",
        "zone_name": "North Residential Suburbs",
        "category": "Residential",
        "base_demand_kL": 780.0,
        "population": 140000,
        "priority_level": 2,
        "pipeline_capacity_kL": 1500.0,
        "nominal_pressure_bar": 3.9,
        "pump_station_id": "PS-102 (North Hills)",
    },
    {
        "zone_id": "ZONE_3",
        "zone_name": "East Tech & Industrial Park",
        "category": "Industrial",
        "base_demand_kL": 950.0,
        "population": 65000,
        "priority_level": 2,
        "pipeline_capacity_kL": 1800.0,
        "nominal_pressure_bar": 4.5,
        "pump_station_id": "PS-103 (Tech Hub Booster)",
    },
    {
        "zone_id": "ZONE_4",
        "zone_name": "South Medical & University Campus",
        "category": "Institutional",
        "base_demand_kL": 620.0,
        "population": 95000,
        "priority_level": 5,
        "pipeline_capacity_kL": 1400.0,
        "nominal_pressure_bar": 4.4,
        "pump_station_id": "PS-104 (Campus Dedicated)",
    },
    {
        "zone_id": "ZONE_5",
        "zone_name": "West Heritage & Eco-Reserve",
        "category": "Mixed/Residential",
        "base_demand_kL": 380.0,
        "population": 72000,
        "priority_level": 3,
        "pipeline_capacity_kL": 1000.0,
        "nominal_pressure_bar": 3.7,
        "pump_station_id": "PS-105 (West Reservoir Gravity Feed)",
    },
    {
        "zone_id": "ZONE_6",
        "zone_name": "Northeast High-Rise Enclave",
        "category": "Residential",
        "base_demand_kL": 690.0,
        "population": 115000,
        "priority_level": 4,
        "pipeline_capacity_kL": 1350.0,
        "nominal_pressure_bar": 4.6,
        "pump_station_id": "PS-106 (Tower High-Pressure Booster)",
    },
]


def generate_smart_water_dataset(
    days: int = 180,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic SCADA municipal water telemetry ending at the current time.
    Includes:
    - Diurnal, weekly, and seasonal demand cycles.
    - Weather interactions (temperature heatwaves, monsoon precipitation).
    - Hydraulic physics: pipeline pressure (bar) and flow velocity (m/s).
    - Water quality telemetry: pH, turbidity (NTU), free chlorine (mg/L).
    - Pump station RPM and energy consumption (kWh).
    - Ground-truth pipe bursts, leaks, and sensor faults.
    """
    np.random.seed(random_seed)

    # Dynamic rolling timestamp ending at current hour
    current_hour = pd.Timestamp.now().floor("h")
    timestamps = pd.date_range(end=current_hour, periods=days * 24, freq="h")
    total_hours = len(timestamps)

    all_zone_dfs = []

    # Reservoir simulation across time
    res_base = 78.0
    res_trend = 10.0 * np.sin(np.linspace(0, 4 * np.pi, total_hours))
    res_noise = np.random.normal(0, 1.2, total_hours)
    reservoir_level_pct = np.clip(res_base + res_trend + res_noise, 45.0, 96.0)

    for z in DEFAULT_ZONES:
        zone_id = z["zone_id"]
        category = z["category"]
        base = z["base_demand_kL"]
        capacity = z["pipeline_capacity_kL"]
        nominal_p = z["nominal_pressure_bar"]
        pump_id = z["pump_station_id"]

        # Time series vectors
        hours = timestamps.hour.values
        day_of_weeks = timestamps.dayofweek.values
        months = timestamps.month.values
        is_weekend = (day_of_weeks >= 5).astype(int)

        # 1. Weather Simulation
        temp_seasonal = 26.0 + 6.0 * np.sin((months - 4) * np.pi / 6)
        temp_diurnal = 5.5 * np.sin((hours - 8) * np.pi / 12)
        temperature = np.clip(temp_seasonal + temp_diurnal + np.random.normal(0, 1.2, total_hours), 12.0, 44.0)

        humidity = np.clip(80.0 - (temperature - 20.0) * 1.8 + np.random.normal(0, 4.0, total_hours), 20.0, 98.0)

        rain_prob = np.where(np.isin(months, [6, 7, 8, 9, 10]), 0.15, 0.03)
        rain_event = (np.random.random(total_hours) < rain_prob).astype(int)
        precipitation = rain_event * np.random.exponential(scale=6.0, size=total_hours)
        precipitation = np.round(precipitation, 2)

        # 2. Sector Consumption Multipliers
        diurnal_mult = np.ones(total_hours)

        if category == "Residential":
            morning_peak = 0.55 * np.exp(-((hours - 8) ** 2) / 3.0)
            evening_peak = 0.45 * np.exp(-((hours - 20) ** 2) / 4.0)
            night_dip = -0.50 * np.exp(-((hours - 3) ** 2) / 4.0)
            weekend_boost = 0.15 * is_weekend
            diurnal_mult = 1.0 + morning_peak + evening_peak + night_dip + weekend_boost

        elif category == "Commercial":
            workday_curve = np.where((hours >= 9) & (hours <= 18), 0.65, -0.45)
            diurnal_mult = np.where(
                is_weekend == 1,
                0.45 + 0.1 * np.sin((hours - 12) * np.pi / 12),
                1.0 + workday_curve
            )

        elif category == "Industrial":
            diurnal_mult = 1.0 + 0.12 * np.sin(hours * np.pi / 4) + 0.05 * (1 - is_weekend)

        elif category == "Institutional":
            diurnal_mult = 1.0 + 0.35 * np.exp(-((hours - 13) ** 2) / 10.0)

        else: # Mixed
            diurnal_mult = 1.0 + 0.30 * np.sin((hours - 7) * np.pi / 12)

        # 3. Weather Elasticity
        weather_temp_factor = 1.0 + np.maximum(0, temperature - 25.0) * 0.012
        weather_rain_factor = np.clip(1.0 - (precipitation / 10.0) * 0.08, 0.75, 1.0)

        noise = np.random.normal(1.0, 0.04, total_hours)
        clean_demand = base * diurnal_mult * weather_temp_factor * weather_rain_factor * noise
        actual_demand = clean_demand.copy()

        # 4. Hydraulic Physics & SCADA Telemetry
        # Pressure inversely proportional to flow velocity (Bernoulli effect)
        flow_velocity = (actual_demand / max(base, 1.0)) * 1.5 + np.random.normal(0, 0.05, total_hours)
        flow_velocity = np.clip(flow_velocity, 0.3, 3.2)

        pressure = nominal_p - 0.4 * (actual_demand / max(base, 1.0) - 1.0) + np.random.normal(0, 0.08, total_hours)
        pressure = np.clip(pressure, 2.5, 5.5)

        # Water Quality Telemetry
        ph_level = np.clip(7.35 + np.random.normal(0, 0.08, total_hours), 6.8, 7.9)
        turbidity_ntu = np.clip(0.85 + (flow_velocity * 0.15) + np.random.normal(0, 0.06, total_hours), 0.3, 2.5)
        free_chlorine = np.clip(0.85 - (temperature - 25.0) * 0.005 + np.random.normal(0, 0.04, total_hours), 0.4, 1.3)

        # Pump Station Telemetry
        pump_rpm = np.clip(1400 + (actual_demand / max(base, 1.0)) * 250 + np.random.normal(0, 20, total_hours), 1000, 2200)
        pump_power_kwh = (actual_demand * 0.42) + np.random.normal(0, 3.5, total_hours)

        # 5. Anomaly Injections
        anomaly_flags = np.zeros(total_hours, dtype=int)
        anomaly_types = ["NORMAL"] * total_hours

        # Zone 2: Pipe Burst event in the last 48 hours
        if zone_id == "ZONE_2":
            burst_start = total_hours - 28
            burst_duration = 14
            actual_demand[burst_start:burst_start + burst_duration] *= 2.65
            pressure[burst_start:burst_start + burst_duration] = np.random.uniform(1.7, 2.1, burst_duration)
            turbidity_ntu[burst_start:burst_start + burst_duration] = np.random.uniform(4.2, 5.8, burst_duration)
            anomaly_flags[burst_start:burst_start + burst_duration] = 1
            for idx in range(burst_start, burst_start + burst_duration):
                anomaly_types[idx] = "PIPE_BURST"

        # Zone 3: Persistent leak / illegal tapping
        if zone_id == "ZONE_3":
            leak_start = total_hours - 96
            leak_duration = 36
            actual_demand[leak_start:leak_start + leak_duration] += 320.0
            pressure[leak_start:leak_start + leak_duration] -= 0.6
            anomaly_flags[leak_start:leak_start + leak_duration] = 1
            for idx in range(leak_start, leak_start + leak_duration):
                anomaly_types[idx] = "PERSISTENT_LEAK"

        # Zone 5: Sensor failure
        if zone_id == "ZONE_5":
            drop_start = total_hours - 60
            drop_duration = 8
            actual_demand[drop_start:drop_start + drop_duration] *= 0.04
            pressure[drop_start:drop_start + drop_duration] = 0.0
            anomaly_flags[drop_start:drop_start + drop_duration] = 1
            for idx in range(drop_start, drop_start + drop_duration):
                anomaly_types[idx] = "SENSOR_DROPOUT"

        # Additional background micro-leaks (0.4%)
        random_spikes = np.random.random(total_hours) < 0.004
        for i in np.where(random_spikes)[0]:
            if anomaly_flags[i] == 0:
                actual_demand[i] *= np.random.uniform(1.7, 2.2)
                pressure[i] -= np.random.uniform(0.5, 0.9)
                anomaly_flags[i] = 1
                anomaly_types[i] = "UNSCHEDULED_SPIKE"

        # Dynamic Allocated Supply Baseline
        allocated_supply = clean_demand * np.random.uniform(0.92, 1.15, total_hours)
        allocated_supply = np.clip(allocated_supply, 0, capacity)

        zone_df = pd.DataFrame({
            "timestamp": timestamps,
            "zone_id": zone_id,
            "zone_name": z["zone_name"],
            "category": category,
            "priority_level": z["priority_level"],
            "pipeline_capacity_kL": capacity,
            "pump_station_id": pump_id,
            "temperature_c": np.round(temperature, 2),
            "humidity_pct": np.round(humidity, 1),
            "precipitation_mm": np.round(precipitation, 2),
            "actual_consumption_kL": np.round(actual_demand, 2),
            "allocated_supply_kL": np.round(allocated_supply, 2),
            "pipeline_pressure_bar": np.round(pressure, 2),
            "flow_velocity_ms": np.round(flow_velocity, 2),
            "water_quality_ph": np.round(ph_level, 2),
            "turbidity_ntu": np.round(turbidity_ntu, 2),
            "free_chlorine_mg_l": np.round(free_chlorine, 2),
            "pump_rpm": np.round(pump_rpm, 0),
            "pump_energy_kwh": np.round(pump_power_kwh, 1),
            "reservoir_level_pct": np.round(reservoir_level_pct, 1),
            "is_anomaly": anomaly_flags,
            "anomaly_type": anomaly_types
        })

        all_zone_dfs.append(zone_df)

    combined_df = pd.concat(all_zone_dfs, ignore_index=True)
    combined_df.sort_values(by=["timestamp", "zone_id"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def load_or_generate_raw_data(data_dir: str = "data/raw", force_regenerate: bool = False) -> pd.DataFrame:
    """
    Loads raw consumption dataset if it exists, otherwise synthesizes and persists it.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "water_consumption_raw.csv")

    if os.path.exists(csv_path) and not force_regenerate:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df

    print(f"[DataLoader] Generating authentic SCADA telemetry dataset at {csv_path}...")
    df = generate_smart_water_dataset(days=180)
    df.to_csv(csv_path, index=False)
    print(f"[DataLoader] Generated and saved {len(df):,} records.")
    return df
