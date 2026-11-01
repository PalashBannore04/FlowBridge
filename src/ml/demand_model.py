"""
Demand Prediction Module for SE Smart City Water Redistribution System.
Trains and deploys high-performance Gradient Boosted Tree models (XGBoost / HistGradientBoosting / RandomForest)
to forecast upcoming zone-level water consumption across 1h, 6h, and 24h horizons.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.data_pipeline.feature_engineering import FEATURE_COLUMNS


class WaterDemandForecaster:
    """
    Supervised Multi-Horizon Water Demand Forecaster.
    Encapsulates training, validation, multi-horizon inference, and persistence.
    """

    HORIZONS = {
        "1h": "target_lead_1h",
        "6h": "target_lead_6h",
        "24h": "target_lead_24h"
    }

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = FEATURE_COLUMNS
        self.zone_encoders: Dict[str, int] = {}

    def _get_estimator(self):
        """Constructs underlying regressor instance."""
        if self.model_type == "xgboost" and XGB_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=180,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
        else:
            return HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=7,
                learning_rate=0.06,
                random_state=42
            )

    def prepare_xy(self, df: pd.DataFrame, horizon: str = "1h") -> Tuple[pd.DataFrame, pd.Series]:
        """Prepares feature matrix X and target vector y for given horizon."""
        target_col = self.HORIZONS.get(horizon, "target_lead_1h")

        # Encode zone_id if present
        df_encoded = df.copy()
        if "zone_encoded" not in df_encoded.columns and "zone_id" in df_encoded.columns:
            if not self.zone_encoders:
                zones = sorted(df_encoded["zone_id"].unique())
                self.zone_encoders = {z: idx for idx, z in enumerate(zones)}
            df_encoded["zone_encoded"] = df_encoded["zone_id"].map(self.zone_encoders)

        available_features = [f for f in self.feature_names if f in df_encoded.columns]
        if "zone_encoded" in df_encoded.columns and "zone_encoded" not in available_features:
            available_features.append("zone_encoded")

        X = df_encoded[available_features].fillna(0)
        y = df_encoded[target_col].fillna(0)
        return X, y

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Trains models across all 3 horizons (1h, 6h, 24h) and records metrics."""
        print(f"[DemandForecaster] Initiating training across horizons using model type '{self.model_type}'...")

        for horizon_key in self.HORIZONS.keys():
            X_train, y_train = self.prepare_xy(train_df, horizon=horizon_key)
            regressor = self._get_estimator()
            regressor.fit(X_train, y_train)
            self.models[horizon_key] = regressor

            # Evaluation
            eval_target_df = val_df if val_df is not None else train_df
            X_eval, y_eval = self.prepare_xy(eval_target_df, horizon=horizon_key)
            preds = regressor.predict(X_eval)

            rmse = np.sqrt(mean_squared_error(y_eval, preds))
            mae = mean_absolute_error(y_eval, preds)
            r2 = r2_score(y_eval, preds)
            mape = np.mean(np.abs((y_eval - preds) / np.maximum(y_eval, 1e-4))) * 100.0

            self.metrics[horizon_key] = {
                "rmse": round(float(rmse), 2),
                "mae": round(float(mae), 2),
                "r2": round(float(r2), 4),
                "mape": round(float(mape), 2)
            }
            print(f"[DemandForecaster] Horizon [{horizon_key.upper()}] -> RMSE: {rmse:.2f} kL, MAE: {mae:.2f} kL, R²: {r2:.4f}, MAPE: {mape:.2f}%")

        return self.metrics

    def predict(self, df: pd.DataFrame, horizon: str = "1h") -> np.ndarray:
        """Generates future demand predictions for specified horizon."""
        if horizon not in self.models:
            horizon = "1h"
        model = self.models.get(horizon)
        if model is None:
            raise ValueError(f"Model for horizon '{horizon}' has not been trained.")

        X, _ = self.prepare_xy(df, horizon=horizon)
        preds = model.predict(X)
        return np.clip(preds, 0, None)

    def get_feature_importances(self, horizon: str = "1h") -> pd.DataFrame:
        """Extracts normalized feature importance rankings."""
        model = self.models.get(horizon)
        if model is None:
            return pd.DataFrame()

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = list(self.feature_names)

        if hasattr(model, "feature_importances_"):
            importances = list(model.feature_importances_)
            min_len = min(len(feature_names), len(importances))
            df_imp = pd.DataFrame({
                "feature": feature_names[:min_len],
                "importance": importances[:min_len]
            }).sort_values(by="importance", ascending=False)
            total_imp = df_imp["importance"].sum()
            df_imp["normalized_pct"] = (df_imp["importance"] / (total_imp if total_imp > 0 else 1.0)) * 100.0
            return df_imp

        return pd.DataFrame({"feature": feature_names, "importance": [1.0 / len(feature_names)] * len(feature_names)})

    def save(self, filepath: str = "models/demand_forecaster.pkl"):
        """Serializes forecaster artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[DemandForecaster] Model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/demand_forecaster.pkl") -> "WaterDemandForecaster":
        """Deserializes forecaster artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trained model artifact not found at {filepath}")
        return joblib.load(filepath)


def train_and_persist_demand_model(
    data_path: str = "data/processed/water_consumption_features.csv",
    save_path: str = "models/demand_forecaster.pkl"
) -> Tuple[WaterDemandForecaster, Dict[str, Any]]:
    """Convenience pipeline to load processed data, train forecaster, and save artifact."""
    from src.data_pipeline.loader import load_or_generate_raw_data
    from src.data_pipeline.feature_engineering import build_feature_pipeline

    if not os.path.exists(data_path):
        raw_df = load_or_generate_raw_data()
        df = build_feature_pipeline(raw_df)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    # Chronological train/validation split across all zones
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    unique_times = np.sort(df["timestamp"].unique())
    split_timestamp = unique_times[int(len(unique_times) * 0.8)]
    train_df = df[df["timestamp"] < split_timestamp].copy()
    val_df = df[df["timestamp"] >= split_timestamp].copy()

    forecaster = WaterDemandForecaster(model_type="xgboost")
    metrics = forecaster.train(train_df, val_df)
    forecaster.save(save_path)
    return forecaster, metrics


if __name__ == "__main__":
    model, metrics = train_and_persist_demand_model()
    print("\nTraining Metrics Summary:")
    print(metrics)
