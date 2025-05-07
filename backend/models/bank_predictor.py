
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings
from schema import UserData_bank
import joblib

model = joblib.load(settings.bank_load_trained_data_lg)
scaler = joblib.load(settings.bank_load_trained_data_sc)

def predict_user(data: UserData_bank):
    # Convert Pydantic model to list of values
    input_list = [[
        data.default,
        data.housing,
        data.loan,
        data.duration,
        data.poutcome,
        data.emp_var_rate,
        data.cons_price_idx,
        data.cons_conf_idx,
        data.euribor3m,
        data.blue_collar,
        data.entrepreneur,
        data.housemaid,
        data.management,
        data.retired,
        data.self_employed,
        data.services,
        data.student,
        data.technician,
        data.unemployed,
        data.unknown
    ]]

    # Now you can transform and predict
    scaled_input = scaler.transform(input_list)
    prediction = model.predict(scaled_input)
    return {"prediction": prediction.tolist()}





