import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.predictor import get_full_forecast, load_model
from src.fetcher import fetch_historical
from src.scorer import compute_surf_score
from src.features import build_features, get_feature_columns
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🤖 ML Model — Surf Score Prediction")
st.caption("XGBoost model predicting surf score 6 hours ahead from marine and weather features")

# --- Model performance ---
st.subheader("Model Performance")

@st.cache_data(ttl=86400)
def get_cv_metrics():
    df = fetch_historical()
    df = compute_surf_score(df)
    df = build_features(df)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]
    y = df["target_surf_score"]

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, r2_scores = [], []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=500, max_depth=5,
            learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mae_scores.append(mae)
        r2_scores.append(r2)
        fold_results.append({
            "Fold": fold + 1,
            "MAE": round(mae, 3),
            "R²": round(r2, 3),
            "Test size": len(y_test)
        })

    return {
        "mae": round(np.mean(mae_scores), 3),
        "r2": round(np.mean(r2_scores), 3),
        "folds": fold_results
    }

with st.spinner("Running cross-validation..."):
    metrics = get_cv_metrics()

col1, col2, col3 = st.columns(3)
col1.metric("Mean MAE", metrics["mae"], help="Mean Absolute Error on score (0-10)")
col2.metric("Mean R²", metrics["r2"], help="Coefficient of determination")
col3.metric("CV Strategy", "TimeSeriesSplit (5 folds)", help="Respects temporal ordering")

st.dataframe(pd.DataFrame(metrics["folds"]), use_container_width=True, hide_index=True)
st.caption("TimeSeriesSplit ensures no future data leaks into training — each fold trains on the past and tests on the future.")

st.divider()

# --- Prediction vs actual on forecast ---
st.subheader("Predicted vs Rule-based Score")

@st.cache_data(ttl=3600)
def load_forecast():
    return get_full_forecast()

with st.spinner("Loading forecast..."):
    df = load_forecast()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index, y=df["surf_score"],
    mode="lines", name="Rule-based Score",
    line=dict(color="#00b4d8", width=2)
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["predicted_score"],
    mode="lines", name="ML Prediction (H+6)",
    line=dict(color="#f77f00", width=2, dash="dash")
))
fig.update_layout(
    yaxis=dict(range=[0, 10], title="Score"),
    xaxis_title="Date",
    height=350,
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Feature importance ---
st.subheader("Feature Importance")

@st.cache_data(ttl=86400)
def get_feature_importance():
    model_data = load_model()
    model = model_data["model"]
    features = model_data["features"]
    importance = model.feature_importances_
    return pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(15)

fi = get_feature_importance()
fig2 = go.Figure(go.Bar(
    x=fi["Importance"],
    y=fi["Feature"],
    orientation="h",
    marker_color="#00b4d8"
))
fig2.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="Importance",
    height=450
)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Feature importance from XGBoost — shows which variables drive surf score predictions most.")