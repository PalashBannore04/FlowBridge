# FlowBridge: SE Smart City — Smart Water Redistribution System

An end-to-end Machine Learning and Linear Programming decision support system for municipal water grids. Forecasts zone-level water demand, detects abnormal consumption (pipe bursts, illegal leaks, and sensor drops), calculates shortage risk, and computes optimal supply redistribution in real time.

---

## 🏛️ System Architecture & Workflow

```text
               [ Historical Consumption Data + Open-Meteo Weather API ]
                                          │
                                          ▼
                         [ Data Pipeline & Feature Engineering ]
                     (Temporal, Cyclical Sin/Cos, Lags, Rolling Stats)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
       [ Demand Forecaster ]                           [ Anomaly Detector ]
        (XGBoost Regressor)                      (Isolation Forest + Rolling Z-Score)
         R² > 0.94, MAPE ~5%                       (Burst Leaks & Sensor Dropouts)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                              [ Shortage & Risk Engine ]
                         (Predicted Demand vs Available Supply)
                                          │
                                          ▼
                        [ Supply Redistribution Optimizer ]
                        (Linear Programming via PuLP / SciPy)
                                          │
                                          ▼
                            [ Interactive Streamlit UI ]
             (Geospatial Folium Map, What-If Dynamic Sliders, Plotly Trends)
```

---

## 🚀 Key Modules & Capabilities

1. **Synthetic Kaggle-Style Data Generator (`src/data_pipeline/loader.py`)**:
   - Generates multi-zone time-series datasets modeled after smart water grid benchmarks (25,000+ hourly observations across 6 distinct municipal zones).
   - Simulates diurnal cycles (morning and evening peaks), sector-specific usage (commercial vs. residential vs. industrial), weather elasticity, and ground-truth anomaly events.

2. **Feature Engineering Store (`src/data_pipeline/feature_engineering.py`)**:
   - Computes temporal, calendar, cyclical trigonometry (`sin_hour`, `cos_hour`, `sin_month`, `cos_month`), multi-horizon lags (`lag_1h`, `lag_24h`, `lag_168h`), and rolling statistics.

3. **Demand Prediction Engine (`src/ml/demand_model.py`)**:
   - Supervised **XGBoost Regressor** trained across multi-horizon tactical and operational windows:
     - **1-Hour Horizon**: $R^2 \approx 0.942$, $\text{MAPE} \approx 5.6\%$
     - **6-Hour Horizon**: $R^2 \approx 0.980$, $\text{MAPE} \approx 3.4\%$
     - **24-Hour Horizon**: $R^2 \approx 0.986$, $\text{MAPE} \approx 2.5\%$
   - Saves artifacts to `models/demand_forecaster.pkl`.

4. **Abnormal Consumption & Leak Radar (`src/ml/anomaly_model.py`)**:
   - Dual-engine anomaly detection combining **Rolling Z-Score analysis** with **Isolation Forest**.
   - Pinpoints pipe bursts, persistent night leaks, illegal tapping, and sensor dropouts.
   - Saves artifacts to `models/anomaly_detector.pkl`.

5. **Linear Programming Redistribution Engine (`src/ml/optimizer.py`)**:
   - Formulates and solves a constrained optimization problem via **PuLP (CBC)** and **SciPy (HiGHS)**.
   - Minimizes priority-weighted unmet demand subject to total reservoir supply constraints and pipeline transmission capacities.
   - Classifies zones into **Adequate (Green)**, **Warning (Yellow)**, and **Critical Shortage (Red)**.

6. **Interactive Command Center (`app/main.py`)**:
   - **Geospatial Map (`Folium`)**: Color-coded zone boundaries, pulsing risk markers, and inter-zone pipeline flow vectors.
   - **Dynamic "What-If" Supply Slider**: Instantly recalculate optimal allocations as municipal reservoir levels change.
   - **Weather Sensitivity Simulator**: Adjust temperatures and simulate rainfall downpours to view climate-driven demand elasticity.

---

## 📁 Project Directory Structure

```text
FlowBridge/
├── CONTEXT.md                    # Problem specs and requirements
├── requirements.txt              # Core Python dependencies
├── README.md                     # System documentation
├── LICENSE                       # MIT License
│
├── data/
│   ├── raw/                      # Raw consumption time-series dataset
│   │   └── water_consumption_raw.csv
│   ├── processed/                # Engineered feature store
│   │   └── water_consumption_features.csv
│   └── zones_geojson.json        # GeoJSON boundary coordinates for city zones
│
├── models/
│   ├── demand_forecaster.pkl     # Trained XGBoost regression forecaster
│   └── anomaly_detector.pkl      # Trained Isolation Forest anomaly detector
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── loader.py             # Data loading and synthetic Kaggle-style generation
│   │   └── feature_engineering.py# Temporal, lag, and rolling statistics
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── demand_model.py       # XGBoost demand forecaster
│   │   ├── anomaly_model.py      # Isolation Forest & Z-Score anomaly engine
│   │   └── optimizer.py          # PuLP / SciPy linear programming optimizer
│   └── utils/
│       ├── __init__.py
│       └── weather_api.py        # Open-Meteo REST API client
│
└── app/
    ├── main.py                   # Streamlit dashboard entrypoint
    ├── components/
    │   ├── map_component.py      # Folium geospatial map renderer
    │   ├── charts.py             # Plotly interactive graphs
    │   └── controls.py           # Scenario sliders & "What-If" widgets
    └── styles/
        └── custom.css            # Dark mode glassmorphic CSS styling
```

---

## ⚡ Quickstart & Execution

### 1. Environment Setup
```powershell
# Using uv (recommended)
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Generate Dataset and Retrain Models (Optional)
The pre-generated dataset and trained models are already stored in `data/` and `models/`. To re-run the entire pipeline from scratch:
```powershell
python -c "from src.data_pipeline.loader import load_or_generate_raw_data; from src.data_pipeline.feature_engineering import build_feature_pipeline; from src.ml.demand_model import train_and_persist_demand_model; from src.ml.anomaly_model import train_and_persist_anomaly_detector; raw=load_or_generate_raw_data(); build_feature_pipeline(raw); train_and_persist_demand_model(); train_and_persist_anomaly_detector();"
```

### 3. Launch the Interactive Dashboard
```powershell
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
