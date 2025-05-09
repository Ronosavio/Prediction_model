import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.config import settings
import joblib

model = joblib.load(settings.percapita_slr_trained_model)

def predict_percapita(data):
    prediction = model.predict([[data]])
    return f"Predicted per capita income for the year {data}:  ₹{prediction[0]:.2f}"

