import pandas as pd
import numpy as np
from config import WAVE_HEIGHT_IDEAL, WAVE_PERIOD_IDEAL, WIND_SPEED_MAX


def compute_surf_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an hourly surf score (0-10) based on wave and wind conditions.

    Scoring breakdown:
        - Wave height  : 4 points — ideal range 1.0-2.5m
        - Wave period  : 3 points — ideal range 8-16s (longer = cleaner waves)
        - Wind speed   : 2 points — below 20 km/h is good, above is choppy
        - Swell height : 1 point  — bonus for strong groundswell
    """
    df = df.copy()

    # --- Wave height score (0-4) ---
    def height_score(h):
        if pd.isna(h):
            return 0
        low, high = WAVE_HEIGHT_IDEAL
        if h < 0.3:
            return 0
        elif h < low:
            return round(2 * (h / low), 1)
        elif h <= high:
            return 4.0
        else:
            # Penalise waves that are too big
            return max(0, round(4 - (h - high) * 1.5, 1))

    # --- Wave period score (0-3) ---
    def period_score(p):
        if pd.isna(p):
            return 0
        low, high = WAVE_PERIOD_IDEAL
        if p < 5:
            return 0
        elif p < low:
            return round(1.5 * (p / low), 1)
        elif p <= high:
            return 3.0
        else:
            return 3.0

    # --- Wind speed score (0-2) ---
    def wind_score(w):
        if pd.isna(w):
            return 0
        if w <= 10:
            return 2.0
        elif w <= WIND_SPEED_MAX:
            return round(2 * (1 - (w - 10) / 10), 1)
        else:
            return 0.0

    # --- Swell bonus score (0-1) ---
    def swell_score(s):
        if pd.isna(s):
            return 0
        if s >= 1.0:
            return 1.0
        return round(s, 1)

    df["score_height"] = df["wave_height"].apply(height_score)
    df["score_period"] = df["wave_period"].apply(period_score)
    df["score_wind"] = df["wind_speed"].apply(wind_score)
    df["score_swell"] = df["swell_height"].apply(swell_score)

    df["surf_score"] = (
        df["score_height"] +
        df["score_period"] +
        df["score_wind"] +
        df["score_swell"]
    ).clip(0, 10).round(1)

    # Surf quality label
    def label(score):
        if score >= 7:
            return "🟢 Good"
        elif score >= 4:
            return "🟡 Fair"
        else:
            return "🔴 Poor"

    df["surf_label"] = df["surf_score"].apply(label)

    return df