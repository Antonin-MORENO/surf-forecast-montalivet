import streamlit as st
import plotly.graph_objects as go
from src.predictor import get_full_forecast

st.title("🏄 Surf Forecast — Montalivet")
st.caption("7-day hourly surf conditions with ML-powered score prediction (H+6)")

refresh = st.button("🔄 Refresh forecast", type="primary")

@st.cache_data(ttl=3600)
def load_forecast():
    return get_full_forecast()

if refresh:
    st.cache_data.clear()

with st.spinner("Loading forecast..."):
    df = load_forecast()

if df.empty:
    st.error("No forecast data available.")
    st.stop()

# --- Current conditions ---
st.subheader("Current Conditions")
now = df.iloc[0]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🌊 Wave Height", f"{now['wave_height']:.1f}m")
col2.metric("⏱️ Wave Period", f"{now['wave_period']:.0f}s")
col3.metric("💨 Wind Speed", f"{now['wind_speed']:.0f} km/h")
col4.metric("🌡️ Temperature", f"{now['temperature']:.0f}°C")
col5.metric("🏄 Surf Score", f"{now['surf_score']:.1f}/10", now['surf_label'])

st.divider()

# --- Surf score chart ---
st.subheader("Surf Score Forecast")

fig = go.Figure()

# Colour zones: green = good, orange = fair, red = poor
fig.add_hrect(y0=7, y1=10, fillcolor="green", opacity=0.07, line_width=0)
fig.add_hrect(y0=4, y1=7, fillcolor="orange", opacity=0.07, line_width=0)
fig.add_hrect(y0=0, y1=4, fillcolor="red", opacity=0.07, line_width=0)

fig.add_trace(go.Scatter(
    x=df.index, y=df["surf_score"],
    mode="lines", name="Rule-based Score (Open-Meteo forecast)",
    line=dict(color="#00b4d8", width=2)
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["predicted_score"],
    mode="lines", name="ML Score (XGBoost H+6)",
    line=dict(color="#f77f00", width=2, dash="dash")
))
fig.update_layout(
    yaxis=dict(range=[0, 10], title="Score"),
    xaxis_title="Date",
    height=350,
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Rule-based score applies fixed thresholds to Open-Meteo forecast data. ML score uses XGBoost trained on 10 years of historical patterns.")

# --- Wave height chart ---
st.subheader("Wave & Swell Height")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df.index, y=df["wave_height"],
    mode="lines", name="Wave Height (m)",
    line=dict(color="#0077b6", width=2), fill="tozeroy", fillcolor="rgba(0,119,182,0.1)"
))
fig2.add_trace(go.Scatter(
    x=df.index, y=df["swell_height"],
    mode="lines", name="Swell Height (m)",
    line=dict(color="#48cae4", width=1.5, dash="dot")
))
fig2.update_layout(xaxis_title="Date", yaxis_title="Metres", height=300)
st.plotly_chart(fig2, use_container_width=True)

# --- Wind chart ---
st.subheader("Wind Speed")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df.index, y=df["wind_speed"],
    mode="lines", name="Wind Speed (km/h)",
    line=dict(color="#90e0ef", width=1.5), fill="tozeroy", fillcolor="rgba(144,224,239,0.15)"
))
fig3.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Max ideal wind")
fig3.update_layout(xaxis_title="Date", yaxis_title="km/h", height=280)
st.plotly_chart(fig3, use_container_width=True)

# --- Best windows table ---
st.subheader("🟢 Best Surf Windows")
good = df[df["surf_score"] >= 7][["wave_height", "wave_period", "wind_speed", "surf_score", "surf_label"]]
if not good.empty:
    good.index = good.index.strftime("%a %d %b %H:%M")
    st.dataframe(good.rename(columns={
        "wave_height": "Height (m)",
        "wave_period": "Period (s)",
        "wind_speed": "Wind (km/h)",
        "surf_score": "Score",
        "surf_label": "Quality"
    }), use_container_width=True)
else:
    st.info("No good surf windows in the forecast period.")