import pandas as pd
import numpy as np
import pickle
from src.fetcher import fetch_forecast
from src.scorer import compute_surf_score
from src.features import build_features


def load_model() -> dict:
    """Load the trained XGBoost model and feature list from disk."""
    with open("models/surf_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_surf_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the trained model to forecast data to predict surf score at H+6.

    Args:
        df: DataFrame with surf scores and engineered features.

    Returns:
        DataFrame with an additional 'predicted_score' column.
    """
    model_data = load_model()
    model = model_data["model"]
    features = model_data["features"]

    # Keep only available feature columns
    available = [f for f in features if f in df.columns]
    X = df[available]

    df = df.copy()
    df["predicted_score"] = model.predict(X).clip(0, 10).round(1)

    # Predicted label
    def label(score):
        if score >= 7:
            return "🟢 Good"
        elif score >= 4:
            return "🟡 Fair"
        else:
            return "🔴 Poor"

    df["predicted_label"] = df["predicted_score"].apply(label)

    return df


def get_full_forecast() -> pd.DataFrame:
    """
    Full pipeline: fetch recent history + forecast → score → features → predict.
    Uses 30 days of recent data to compute lag features for the forecast window.
    """
    from src.fetcher import fetch_recent

    hist = fetch_recent(days=30)
    forecast = fetch_forecast()

    combined = pd.concat([hist, forecast])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    combined = compute_surf_score(combined)
    combined = build_features(combined)
    combined = predict_surf_score(combined)

    now = pd.Timestamp.now(tz='Europe/Paris').tz_convert('UTC').tz_localize(None)
    combined.index = combined.index.tz_localize(None) if combined.index.tzinfo else combined.index
    return combined[combined.index >= now].dropna(subset=["predicted_score"])