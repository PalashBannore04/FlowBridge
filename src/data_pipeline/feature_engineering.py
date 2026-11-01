"""
Feature Engineering Pipeline for Smart Water Demand Forecasting.
Computes temporal indicators, cyclical trigonometric encodings,
multi-horizon lag features, and rolling consumption statistics.
"""

from typing import List, Tuple
import os
import numpy as np
import pandas as pd


# Canonical feature column names used by ML models
FEATURE_COLUMNS: List[str] = [
    # Temporal & Cyclical
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_business_hours",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    # Zone identity (one-hot or label encoded priority)
    "priority_level",
    # Weather metrics
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    # Lag features
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_168h",
    # Rolling statistics
    "rolling_mean_6h",
    "rolling_std_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_max_24h",
    "rolling_min_24h",
]


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and cyclical trigonometric features from timestamps."""
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 18) & (df["is_weekend"] == 0)).astype(int)

    # Cyclical representations to preserve continuity (23:00 -> 00:00)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def generate_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag and rolling window metrics grouped per zone to prevent data leakage across zones.
    """
    df = df.sort_values(by=["zone_id", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("zone_id")

    # Time-shifted lags
    df["lag_1h"] = grouped["actual_consumption_kL"].shift(1)
    df["lag_2h"] = grouped["actual_consumption_kL"].shift(2)
    df["lag_3h"] = grouped["actual_consumption_kL"].shift(3)
    df["lag_24h"] = grouped["actual_consumption_kL"].shift(24)    # Previous day same hour
    df["lag_168h"] = grouped["actual_consumption_kL"].shift(168)  # Previous week same hour

    # Rolling window statistics (computed strictly on past values with shift(1))
    df["rolling_mean_6h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.shift(1).rolling(window=6, min_periods=1).mean()
    )
    df["rolling_std_6h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.shift(1).rolling(window=6, min_periods=1).std().fillna(0)
    )
    df["rolling_mean_24h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.shift(1).rolling(window=24, min_periods=1).mean()
    )
    df["rolling_std_24h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.shift(1).rolling(window=24, min_periods=1).std().fillna(0)
    )
    df["rolling_max_24h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.shift(1).rolling(window=24, min_periods=1).max()
    )
    df["rolling_min_24h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.shift(1).rolling(window=24, min_periods=1).min()
    )

    # Multi-horizon forecasting targets
    # Horizon 1h (immediate next hour), 6h (next 6h average), 24h (next 24h average)
    df["target_lead_1h"] = grouped["actual_consumption_kL"].shift(-1)
    df["target_lead_6h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.iloc[::-1].rolling(window=6, min_periods=1).mean().iloc[::-1].shift(-1)
    )
    df["target_lead_24h"] = grouped["actual_consumption_kL"].transform(
        lambda s: s.iloc[::-1].rolling(window=24, min_periods=1).mean().iloc[::-1].shift(-1)
    )

    # Fill back initial lags for early periods using rolling backward fill
    fill_cols = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h", "rolling_max_24h", "rolling_min_24h"]
    for col in fill_cols:
        df[col] = df[col].bfill()

    return df


def build_feature_pipeline(
    raw_df: pd.DataFrame,
    save_processed: bool = True,
    processed_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Executes full feature extraction on raw dataframe and optionally persists to disk.
    """
    print("[FeatureEngineering] Extracting temporal and cyclical features...")
    df_temporal = extract_temporal_features(raw_df)

    print("[FeatureEngineering] Generating zone-wise lags and rolling statistics...")
    processed_df = generate_lag_and_rolling_features(df_temporal)

    # Drop trailing rows that do not have future targets
    processed_df = processed_df.dropna(subset=["target_lead_1h", "target_lead_6h", "target_lead_24h"]).copy()
    processed_df.reset_index(drop=True, inplace=True)

    if save_processed:
        os.makedirs(processed_dir, exist_ok=True)
        csv_path = os.path.join(processed_dir, "water_consumption_features.csv")
        parquet_path = os.path.join(processed_dir, "water_consumption_features.parquet")
        processed_df.to_csv(csv_path, index=False)
        try:
            processed_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        print(f"[FeatureEngineering] Processed features saved to {csv_path} ({len(processed_df):,} rows)")

    return processed_df


def load_processed_features(processed_path: str = "data/processed/water_consumption_features.csv") -> pd.DataFrame:
    """Loads feature store from disk."""
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed feature file not found at {processed_path}. Run feature pipeline first.")
    return pd.read_csv(processed_path, parse_dates=["timestamp"])
