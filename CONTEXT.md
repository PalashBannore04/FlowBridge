# Context File: SE Smart City — Smart Water Redistribution System

## Problem Overview
Water consumption varies by hour, day, season, weather, and past usage patterns. This project implements an ML-based smart water management system that forecasts future demand across city zones, detects abnormal consumption (leaks/spikes), calculates potential shortage zones, and optimizes water supply redistribution.

## Core Modules & Requirements
1. **Demand Prediction Engine**: Predict upcoming water consumption (hourly/daily) for each city zone using historical usage, temporal factors (hour, day, month), and weather parameters (temperature, humidity, precipitation).
2. **Abnormal Consumption Detection**: Flag zones experiencing usage significantly deviating from normal patterns (e.g., leaks, illegal usage, sensor malfunctions) and highlight them in the UI.
3. **Shortage Prediction**: Compare forecasted demand against available water supply to identify zones facing potential shortages, outputting estimated supply gaps.
4. **Optimization & Redistribution**: Use Linear Programming (PuLP/SciPy) to compute optimal water redistribution, prioritizing zones with higher predicted demand and greater shortage risks.

## Bonus Features ("Brownie Points")
- **Dynamic Supply Slider**: Interactive control enabling users to adjust total available supply and immediately view updated optimal redistribution plans.
- **Geospatial Zone Map**: Color-coded map indicating current demand, predicted demand, anomaly alerts, and shortage risk levels per zone (Green = Adequate, Yellow = Warning, Red = Critical Shortage).
- **Future vs Historical Trends**: Multi-period selection tool comparing future predicted demand against past consumption curves.
- **Weather Sensitivity**: Integration of live weather data to display how weather shifts affect water consumption.

## Tech Stack
- **Data & ML**: Python, Pandas, NumPy, Scikit-Learn, XGBoost, PuLP / SciPy (Optimization)
- **Frontend / Visualization**: Streamlit, Plotly, Folium / Streamlit-Folium
- **External Integration**: Open-Meteo Weather API (with robust offline fallback generator)
