import requests
import pandas as pd
from datetime import datetime, timedelta
from config import LAT, LON, MARINE_API_URL, WEATHER_API_URL, HISTORICAL_API_URL, MARINE_HISTORICAL_URL, FORECAST_DAYS, TRAINING_YEARS


def fetch_forecast() -> pd.DataFrame:
    """
    Fetch 7-day hourly marine and weather forecast for Montalivet.
    Combines wave data (Open-Meteo Marine) and wind/weather data (Open-Meteo Weather).
    """
    # Marine forecast — waves
    marine_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period",
        "forecast_days": FORECAST_DAYS,
        "timezone": "Europe/Paris"
    }
    marine_resp = requests.get(MARINE_API_URL, params=marine_params, timeout=10)
    marine_resp.raise_for_status()
    marine_data = marine_resp.json()["hourly"]

    # Weather forecast — wind
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "windspeed_10m,winddirection_10m,precipitation,temperature_2m",
        "forecast_days": FORECAST_DAYS,
        "timezone": "Europe/Paris"
    }
    weather_resp = requests.get(WEATHER_API_URL, params=weather_params, timeout=10)
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()["hourly"]

    # Merge into single DataFrame
    df = pd.DataFrame({
        "datetime": pd.to_datetime(marine_data["time"]),
        "wave_height": marine_data["wave_height"],
        "wave_period": marine_data["wave_period"],
        "wave_direction": marine_data["wave_direction"],
        "swell_height": marine_data["swell_wave_height"],
        "swell_period": marine_data["swell_wave_period"],
        "wind_speed": weather_data["windspeed_10m"],
        "wind_direction": weather_data["winddirection_10m"],
        "precipitation": weather_data["precipitation"],
        "temperature": weather_data["temperature_2m"]
    })

    df = df.set_index("datetime")
    return df


def fetch_historical() -> pd.DataFrame:
    """
    Fetch historical hourly marine and weather data for model training.
    Covers the last TRAINING_YEARS years.
    """
    end_date = datetime.today() - timedelta(days=7)
    start_date = end_date - timedelta(days=365 * TRAINING_YEARS)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Historical marine data
    marine_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period",
        "start_date": start_str,
        "end_date": end_str,
        "timezone": "Europe/Paris"
    }
    marine_resp = requests.get(MARINE_HISTORICAL_URL, params=marine_params, timeout=30)
    marine_resp.raise_for_status()
    marine_data = marine_resp.json()["hourly"]

    # Historical weather data
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "windspeed_10m,winddirection_10m,precipitation,temperature_2m",
        "start_date": start_str,
        "end_date": end_str,
        "timezone": "Europe/Paris"
    }
    weather_resp = requests.get(HISTORICAL_API_URL, params=weather_params, timeout=30)
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(marine_data["time"]),
        "wave_height": marine_data["wave_height"],
        "wave_period": marine_data["wave_period"],
        "wave_direction": marine_data["wave_direction"],
        "swell_height": marine_data["swell_wave_height"],
        "swell_period": marine_data["swell_wave_period"],
        "wind_speed": weather_data["windspeed_10m"],
        "wind_direction": weather_data["winddirection_10m"],
        "precipitation": weather_data["precipitation"],
        "temperature": weather_data["temperature_2m"]
    })

    df = df.set_index("datetime")
    df = df.dropna()
    df = df.sort_index()
    return df

def fetch_recent(days: int = 30) -> pd.DataFrame:
    """Fetch the last N days of historical data for lag feature computation."""
    from datetime import datetime, timedelta
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    marine_params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period",
        "start_date": start_str, "end_date": end_str,
        "timezone": "Europe/Paris"
    }
    marine_resp = requests.get(MARINE_HISTORICAL_URL, params=marine_params, timeout=30)
    marine_resp.raise_for_status()
    marine_data = marine_resp.json()["hourly"]

    weather_params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "windspeed_10m,winddirection_10m,precipitation,temperature_2m",
        "start_date": start_str, "end_date": end_str,
        "timezone": "Europe/Paris"
    }
    weather_resp = requests.get(HISTORICAL_API_URL, params=weather_params, timeout=30)
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(marine_data["time"]),
        "wave_height": marine_data["wave_height"],
        "wave_period": marine_data["wave_period"],
        "wave_direction": marine_data["wave_direction"],
        "swell_height": marine_data["swell_wave_height"],
        "swell_period": marine_data["swell_wave_period"],
        "wind_speed": weather_data["windspeed_10m"],
        "wind_direction": weather_data["winddirection_10m"],
        "precipitation": weather_data["precipitation"],
        "temperature": weather_data["temperature_2m"]
    })
    df = df.set_index("datetime")
    df = df.dropna()
    return df.sort_index()