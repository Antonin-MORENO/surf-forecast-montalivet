import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the surf score prediction model.

    Creates lag features, rolling statistics, and cyclical time encodings
    from raw marine and weather data to capture temporal patterns.
    """
    df = df.copy()

    # --- Lag features (past conditions as predictors) ---
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"wave_height_lag{lag}"] = df["wave_height"].shift(lag)
        df[f"wind_speed_lag{lag}"] = df["wind_speed"].shift(lag)
        df[f"wave_period_lag{lag}"] = df["wave_period"].shift(lag)

    # --- Rolling statistics (trend and variability) ---
    for window in [6, 12, 24]:
        df[f"wave_height_roll_mean{window}"] = df["wave_height"].rolling(window).mean()
        df[f"wave_height_roll_std{window}"] = df["wave_height"].rolling(window).std()
        df[f"wind_speed_roll_mean{window}"] = df["wind_speed"].rolling(window).mean()

    # --- Cyclical time encodings ---
    # Hour of day: encode as sin/cos to capture daily patterns (dawn/dusk winds)
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Month: encode as sin/cos to capture seasonal patterns
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Target: surf score at H+24 ---
    df["target_surf_score"] = df["surf_score"].shift(-6)

    # Drop rows with NaN (from lags and target shift)
    df = df.dropna()

    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for model training."""
    base = [
        "wave_height", "wave_period", "wave_direction",
        "swell_height", "swell_period",
        "wind_speed", "wind_direction",
        "precipitation", "temperature"
    ]
    lags = [
        f"{col}_lag{lag}"
        for col in ["wave_height", "wind_speed", "wave_period"]
        for lag in [1, 2, 3, 6, 12, 24]
    ]
    rolls = [
        f"wave_height_roll_mean{w}" for w in [6, 12, 24]
    ] + [
        f"wave_height_roll_std{w}" for w in [6, 12, 24]
    ] + [
        f"wind_speed_roll_mean{w}" for w in [6, 12, 24]
    ]
    time = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

    return list(dict.fromkeys(base + lags + rolls + time))