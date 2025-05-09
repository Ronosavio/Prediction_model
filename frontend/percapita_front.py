import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.models.percapita_predict import predict_percapita
from backend.schema import percapita_feature
from backend.config import settings

with open("performance_details\\percapita_accuracy.txt", "r") as file:
    accuracy = file.read()
accuracy_value = float(accuracy.strip()) 

with open("performance_details\\percapita_rmse.txt", "r") as file:
    rmse = file.read()
rmse_value = float(rmse.strip())

st.set_page_config(page_title="Per Capita Income Predictor", layout="centered")

st.title("📈 Per Capita Income Predictor")
st.subheader(f"R2_Score {accuracy_value * 100:.2f}%**")
st.subheader(f"Model RMSE SCORE {rmse_value:.2f}")

st.markdown("### 🗓️ Choose a Year")
st.markdown("Use the slider below to select the year for which you want to predict the per capita income:")

year = st.slider(
    "Select Year:",
    min_value=1950,
    max_value=2016,  # fixed upper limit
    value=2016,
    step=1,
    help="Move the slider to pick a year"
)

st.markdown("---")


# Predict button
if st.button("Predict"):
    try:
        year_input = percapita_feature(year=year)
        prediction = predict_percapita(year_input.year)
        st.success(prediction)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        

st.image(settings.percapita_plot_path, caption=f"Regression Plot (R2 Score: {accuracy_value* 100:.2f})")
        

