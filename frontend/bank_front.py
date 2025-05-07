import streamlit as st
import sys
import os
import re
import ast
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.schema import UserData_bank  # wherever it's defined
from backend.models.bank_predictor import predict_user

with open("performance_details\\bank_confusion_matrix.txt", "r") as file:
    conf_matrix_str = file.read()
with open("performance_details\\bank_accuracy.txt", "r") as file:
    accuracy = file.read() 
with open("performance_details\\bank_report.txt", "r") as file:
    report = file.read() 
    
accuracy_value = float(accuracy.strip())   



st.title("Subscription Prediction")
st.subheader(f"Model Accuracy {accuracy_value * 100:.2f}%**")

# Sidebar input form
options_map = {"Yes": 1, "Unknown": 0, "No": -1}
options_map_2 = {'success': 1, 'nonexistent': 0, 'failure': -1}

default = options_map[st.sidebar.selectbox("Default", list(options_map.keys()))]
housing = options_map[st.sidebar.selectbox("Housing", list(options_map.keys()))]
loan = options_map[st.sidebar.selectbox("Loan", list(options_map.keys()))]
poutcome = options_map_2[st.sidebar.selectbox("Poutcome", list(options_map_2.keys()))]

duration = st.sidebar.slider("Duration", min_value=0, max_value=500, step=1)
emp_var_rate = st.sidebar.slider("Emp Var Rate", min_value=-3.0, max_value=3.0, step=0.01)
cons_price_idx = st.sidebar.slider("Cons Price Index", min_value=92.0, max_value=94.0, step=0.01)
cons_conf_idx = st.sidebar.slider("Cons Conf Index", min_value=-50.0, max_value=50.0, step=0.01)
euribor3m = st.sidebar.slider("Euribor 3m", min_value=0.0, max_value=6.0, step=0.01)

# Sidebar for job selection (selectbox)
job = st.sidebar.selectbox("Select Job Type", [
    "Blue Collar", "Entrepreneur", "Housemaid", "Management", "Retired", 
    "Self-employed", "Services", "Student", "Technician", "Unemployed", "Unknown"
])

# Map job selection to binary features
job_features = {
    "blue_collar": 0,
    "entrepreneur": 0,
    "housemaid": 0,
    "management": 0,
    "retired": 0,
    "self_employed": 0,
    "services": 0,
    "student": 0,
    "technician": 0,
    "unemployed": 0,
    "unknown": 0
}

# Map selected job to corresponding key
job_key_map = {
    "Blue Collar": "blue_collar",
    "Entrepreneur": "entrepreneur",
    "Housemaid": "housemaid",
    "Management": "management",
    "Retired": "retired",
    "Self-employed": "self_employed",
    "Services": "services",
    "Student": "student",
    "Technician": "technician",
    "Unemployed": "unemployed",
    "Unknown": "unknown"
}

job_features[job_key_map[job]] = 1

input_data = UserData_bank(
    default=default,
    housing=housing,
    loan=loan,
    duration=duration,
    poutcome=poutcome,
    emp_var_rate=emp_var_rate,
    cons_price_idx=cons_price_idx,
    cons_conf_idx=cons_conf_idx,
    euribor3m=euribor3m,
    blue_collar=job_features["blue_collar"],
    entrepreneur=job_features["entrepreneur"],
    housemaid=job_features["housemaid"],
    management=job_features["management"],
    retired=job_features["retired"],
    self_employed=job_features["self_employed"],
    services=job_features["services"],
    student=job_features["student"],
    technician=job_features["technician"],
    unemployed=job_features["unemployed"],
    unknown=job_features["unknown"]
)

# Prediction button
if st.button("Predict"):
    print("Input Data:", input_data.dict)
    result = predict_user(input_data)
    prediction = result["prediction"][0]
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")

cleaned_str = re.sub(r'\s*\n\s*', ' ', conf_matrix_str)  # Remove newlines and extra spaces
cleaned_str = re.sub(r'(\d)\s+(\d)', r'\1, \2', cleaned_str)  # Fix spaces between digits

# Step 2: Ensure correct list format by adding brackets
fixed =  cleaned_str.strip() 
corrected_string = fixed.replace('] [', '], [')
fixed_matrix = eval(corrected_string)


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
plot_confusion_matrix(fixed_matrix)


report_dict = ast.literal_eval(report)

# Handle the 'accuracy' key and other similar float values
for key, value in report_dict.items():
    if isinstance(value, float):  # 'accuracy' is a float, so we wrap it in a dict
        report_dict[key] = {'precision': value, 'recall': value, 'f1-score': value, 'support': value}
        
# Now, create the DataFrame
report_df = pd.DataFrame.from_dict(report_dict, orient='index').round(2)

# Display the classification report in Streamlit
st.subheader("Classification Report")
st.dataframe(report_df)