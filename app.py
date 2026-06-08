import streamlit as st

st.set_page_config(
    page_title="Surf Forecast — Montalivet",
    page_icon="🏄",
    layout="wide"
)

pages = {
    "Forecast": [
        st.Page("views/forecast.py", title="Surf Forecast", icon="🏄"),
        st.Page("views/ml_prediction.py", title="ML Model", icon="🤖"),
    ]
}

pg = st.navigation(pages)
pg.run()