import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from src.fetcher import fetch_historical
from src.scorer import compute_surf_score
from src.features import build_features, get_feature_columns


def train_model() -> dict:
    """
    Train an XGBoost model to predict surf score 24 hours ahead.

    Uses TimeSeriesSplit cross-validation to respect temporal ordering
    and avoid data leakage between train and test sets.

    Returns:
        Dict with trained model, feature columns, and performance metrics.
    """
    print("Fetching historical data...")
    df = fetch_historical()

    print("Computing surf scores...")
    df = compute_surf_score(df)

    print("Building features...")
    df = build_features(df)

    feature_cols = get_feature_columns()
    # Keep only columns that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["target_surf_score"]

    # TimeSeriesSplit — respects temporal order, no future leakage
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []
    r2_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, preds))
        r2_scores.append(r2_score(y_test, preds))

    # Train final model on all data
    final_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42
    )
    final_model.fit(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/surf_model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "features": feature_cols}, f)

    print("Model saved.")

    return {
        "model": final_model,
        "features": feature_cols,
        "mae": round(np.mean(mae_scores), 3),
        "r2": round(np.mean(r2_scores), 3)
    }