"""
Anomaly and Abnormal Consumption Detection Engine.
Combines unsupervised Isolation Forest and Rolling Z-Score statistical analysis
to pinpoint pipe bursts, nocturnal leaks, and sensor malfunctions in real time.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class WaterAnomalyDetector:
    """
    Dual-Mechanism Anomaly Detector:
    1. Statistical Rolling Z-Score (measures deviation from 24h moving baseline)
    2. Multivariate Isolation Forest (evaluates volume, delta, weather context)
    """

    ANOMALY_FEATURES = [
        "actual_consumption_kL",
        "consumption_delta_1h",
        "z_score_24h",
        "night_ratio",
        "temperature_c",
        "humidity_pct"
    ]

    def __init__(self, contamination: float = 0.035, z_threshold: float = 2.8):
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=120,
            n_jobs=-1
        )
        self.is_fitted = False

    def _engineer_detector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes specialized time-local features for anomaly detection."""
        df = df.sort_values(by=["zone_id", "timestamp"]).copy()

        # Group-wise moving statistics
        grouped = df.groupby("zone_id")
        rolling_mean = grouped["actual_consumption_kL"].transform(
            lambda s: s.rolling(24, min_periods=3).mean()
        )
        rolling_std = grouped["actual_consumption_kL"].transform(
            lambda s: s.rolling(24, min_periods=3).std().fillna(1.0)
        )
        rolling_std = np.maximum(rolling_std, 1.0) # Prevent div by zero

        df["z_score_24h"] = (df["actual_consumption_kL"] - rolling_mean) / rolling_std

        # Immediate rate of change (1h delta)
        df["consumption_delta_1h"] = grouped["actual_consumption_kL"].diff().fillna(0)

        # Nocturnal usage factor (hours 01:00 - 05:00 should typically be minimum)
        if "hour" not in df.columns:
            if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour

        df["is_night"] = ((df["hour"] >= 1) & (df["hour"] <= 5)).astype(int)
        df["night_ratio"] = df["actual_consumption_kL"] / np.maximum(rolling_mean, 1.0) * df["is_night"]

        return df

    def fit(self, df: pd.DataFrame) -> "WaterAnomalyDetector":
        """Fits the Isolation Forest on clean/historical feature matrix."""
        print("[AnomalyDetector] Computing multivariate feature matrix for fitting...")
        feature_df = self._engineer_detector_features(df)
        X = feature_df[self.ANOMALY_FEATURES].fillna(0)

        self.isolation_forest.fit(X)
        self.is_fitted = True
        print("[AnomalyDetector] Isolation Forest successfully trained.")
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs dual detection on target dataframe.
        Appends columns:
        - z_score_24h
        - iforest_anomaly_score (-1 for anomaly, 1 for normal)
        - anomaly_probability (0.0 to 1.0)
        - is_anomaly_flag (0 or 1)
        - anomaly_category (NORMAL, CRITICAL_BURST, PERSISTENT_LEAK, SENSOR_FAULT)
        """
        df_feat = self._engineer_detector_features(df)
        X = df_feat[self.ANOMALY_FEATURES].fillna(0)

        if not self.is_fitted:
            self.fit(df)
            X = df_feat[self.ANOMALY_FEATURES].fillna(0)

        # Isolation forest raw decision function (lower = more anomalous)
        raw_scores = self.isolation_forest.decision_function(X)
        iforest_preds = self.isolation_forest.predict(X)  # -1 for anomaly

        # Normalize score into probability: [0.0, 1.0]
        # In sklearn, decision_function is around [-0.5, 0.5]
        anomaly_prob = np.clip(1.0 / (1.0 + np.exp(raw_scores * 8.0)), 0.0, 1.0)

        # Dual condition: either extreme Z-Score OR (Isolation Forest flag AND moderate Z-score)
        z_scores = df_feat["z_score_24h"].values
        is_z_anomaly = np.abs(z_scores) >= self.z_threshold
        is_iforest_anomaly = (iforest_preds == -1)

        final_flags = (is_z_anomaly | (is_iforest_anomaly & (np.abs(z_scores) >= 1.9))).astype(int)

        # Categorize anomalies based on physical signatures
        categories = []
        for i in range(len(df_feat)):
            if final_flags[i] == 0:
                categories.append("NORMAL")
            else:
                z = z_scores[i]
                c_val = df_feat["actual_consumption_kL"].iloc[i]
                delta = df_feat["consumption_delta_1h"].iloc[i]
                hour = df_feat["hour"].iloc[i] if "hour" in df_feat.columns else 12

                if z > 3.0 and delta > 150.0:
                    categories.append("CRITICAL_BURST")
                elif z > 1.8 and (1 <= hour <= 5):
                    categories.append("PERSISTENT_LEAK")
                elif z < -2.2 or c_val < 15.0:
                    categories.append("SENSOR_FAULT")
                else:
                    categories.append("UNSCHEDULED_SPIKE")

        result_df = df.copy()
        result_df["z_score_24h"] = np.round(z_scores, 2)
        result_df["anomaly_score"] = np.round(anomaly_prob, 3)
        result_df["is_anomaly_detected"] = final_flags
        result_df["anomaly_classification"] = categories

        return result_df

    def save(self, filepath: str = "models/anomaly_detector.pkl"):
        """Saves detector artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[AnomalyDetector] Artifact saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/anomaly_detector.pkl") -> "WaterAnomalyDetector":
        """Loads detector artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Anomaly detector artifact not found at {filepath}")
        return joblib.load(filepath)


def train_and_persist_anomaly_detector(
    data_path: str = "data/raw/water_consumption_raw.csv",
    save_path: str = "models/anomaly_detector.pkl"
) -> Tuple[WaterAnomalyDetector, Dict[str, Any]]:
    """Trains and serializes anomaly detector."""
    from src.data_pipeline.loader import load_or_generate_raw_data

    if not os.path.exists(data_path):
        df = load_or_generate_raw_data()
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    detector = WaterAnomalyDetector()
    detector.fit(df)
    detector.save(save_path)

    # Validate detection
    evaluated_df = detector.detect(df.tail(2000))
    detected_count = int(evaluated_df["is_anomaly_detected"].sum())
    stats = {
        "samples_evaluated": len(evaluated_df),
        "anomalies_detected": detected_count,
        "anomaly_rate_pct": round(detected_count / len(evaluated_df) * 100, 2),
        "categories_breakdown": evaluated_df["anomaly_classification"].value_counts().to_dict()
    }
    return detector, stats


if __name__ == "__main__":
    detector, stats = train_and_persist_anomaly_detector()
    print("\nAnomaly Detection Stats:")
    print(stats)
