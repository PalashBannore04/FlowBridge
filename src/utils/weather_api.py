"""
Open-Meteo Weather API Client with graceful offline fallback.
Provides real-time and historical meteorological observations
(temperature, relative humidity, precipitation, wind speed).
"""

from typing import Dict, Any, Optional
import os
import math
import datetime
import requests
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class WeatherAPIClient:
    """Connector for Open-Meteo & OpenWeatherMap APIs with offline fallback generator."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float = 12.9716, longitude: float = 77.5946, timeout: int = 5):
        self.latitude = latitude
        self.longitude = longitude
        self.timeout = timeout
        self.openmeteo_api_key = os.getenv("OPENMETEO_API_KEY", "").strip()
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()

    def fetch_current_and_forecast(self, days: int = 2) -> pd.DataFrame:
        """
        Fetch current and forecast weather from Open-Meteo API.
        Falls back to realistic synthetic weather if network request fails.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "forecast_days": days,
            "timezone": "auto"
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                times = pd.to_datetime(hourly.get("time", []))
                df = pd.DataFrame({
                    "timestamp": times,
                    "temperature_c": hourly.get("temperature_2m", []),
                    "humidity_pct": hourly.get("relative_humidity_2m", []),
                    "precipitation_mm": hourly.get("precipitation", []),
                    "wind_speed_kmh": hourly.get("wind_speed_10m", []),
                    "is_live_api": True
                })
                if not df.empty:
                    return df
        except Exception:
            pass

        # Fallback simulation
        return self.generate_synthetic_weather_window(
            start_time=pd.Timestamp.now().floor("h"),
            hours=days * 24
        )

    def generate_synthetic_weather_window(
        self,
        start_time: pd.Timestamp,
        hours: int = 48,
        base_temp: float = 27.0
    ) -> pd.DataFrame:
        """
        Generates realistic meteorological time-series when offline.
        Models diurnal temperature curves and random precipitation events.
        """
        timestamps = [start_time + pd.Timedelta(hours=i) for i in range(hours)]
        rows = []

        np.random.seed(42)
        for ts in timestamps:
            hour = ts.hour
            month = ts.month

            # Seasonal baseline
            seasonal_offset = 3.5 * math.sin((month - 3) * math.pi / 6)
            # Diurnal curve (coldest at 5 AM, hottest at 2 PM)
            diurnal_offset = 5.0 * math.sin((hour - 8) * math.pi / 12)
            noise = np.random.normal(0, 0.8)

            temp = base_temp + seasonal_offset + diurnal_offset + noise
            humidity = max(25.0, min(95.0, 75.0 - (diurnal_offset * 3.5) + np.random.normal(0, 3)))
            # Occasional rain event
            rain_prob = 0.12 if month in [6, 7, 8, 9, 10] else 0.04
            precip = float(np.random.exponential(2.5)) if np.random.random() < rain_prob else 0.0
            wind = max(2.0, float(np.random.normal(12.0, 4.0)))

            rows.append({
                "timestamp": ts,
                "temperature_c": round(temp, 2),
                "humidity_pct": round(humidity, 1),
                "precipitation_mm": round(precip, 2),
                "wind_speed_kmh": round(wind, 1),
                "is_live_api": False
            })

        return pd.DataFrame(rows)


def get_weather_client(lat: float = 12.9716, lon: float = 77.5946) -> WeatherAPIClient:
    """Factory helper for WeatherAPIClient."""
    return WeatherAPIClient(latitude=lat, longitude=lon)
