import os
from dotenv import load_dotenv

load_dotenv()

# Montalivet coordinates
LAT = 45.3647
LON = -1.1483

# Open-Meteo URLs
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_API_URL = "https://archive-api.open-meteo.com/v1/archive"
MARINE_HISTORICAL_URL = "https://marine-api.open-meteo.com/v1/marine"

# Forecast horizon
FORECAST_DAYS = 7

# ML
TRAINING_YEARS = 5
TARGET_VARIABLE = "wave_height"
FORECAST_HORIZON = 6  # hours ahead

# Surf scoring thresholds
WAVE_HEIGHT_IDEAL = (1.0, 2.5)   # metres
WAVE_PERIOD_IDEAL = (8, 16)       # seconds
WIND_SPEED_MAX = 20               # km/h