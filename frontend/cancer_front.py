import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
import ast
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.models.cancer_predict import predict_cancer
from backend.schema import Cancer_Features
import re


with open("performance_details\\cancer_confusion_matrix.txt", "r") as file:
    conf_matrix_str = file.read()   
with open("performance_details\\cancer_accuracy.txt", "r") as file:
    accuracy = file.read() 
with open("performance_details\\cancer_report.txt", "r") as file:
    report = file.read() 
    
accuracy_value = float(accuracy.strip())   
    
st.title("Breast Cancer Feature Input")
st.subheader(f"Model Accuracy {accuracy_value * 100:.2f}%**")

st.markdown("Adjust the sliders below to input the values for cancer prediction:")

# Dictionary of features with their typical min and max values from real datasets
feature_ranges = {
    'radius_mean': (6.0, 30.0),
    'texture_mean': (10.0, 40.0),
    'perimeter_mean': (40.0, 200.0),
    'area_mean': (100.0, 2500.0),
    'smoothness_mean': (0.05, 0.2),
    'compactness_mean': (0.02, 0.35),
    'concavity_mean': (0.0, 0.5),
    'concave_points_mean': (0.0, 0.2),
    'symmetry_mean': (0.1, 0.3),
    'fractal_dimension_mean': (0.05, 0.1),
    'radius_se': (0.1, 3.0),
    'texture_se': (0.5, 5.0),
    'perimeter_se': (0.5, 10.0),
    'area_se': (5.0, 200.0),
    'smoothness_se': (0.005, 0.05),
    'compactness_se': (0.005, 0.2),
    'concavity_se': (0.0, 0.4),
    'concave_points_se': (0.0, 0.2),
    'symmetry_se': (0.01, 0.08),
    'fractal_dimension_se': (0.001, 0.03),
    'radius_worst': (7.0, 40.0),
    'texture_worst': (10.0, 50.0),
    'perimeter_worst': (50.0, 250.0),
    'area_worst': (200.0, 5000.0),
    'smoothness_worst': (0.1, 0.3),
    'compactness_worst': (0.05, 1.0),
    'concavity_worst': (0.0, 1.5),
    'concave_points_worst': (0.0, 0.5),
    'symmetry_worst': (0.1, 0.5),
    'fractal_dimension_worst': (0.05, 0.3)
}

user_input = {}

# Display sliders in two columns
col1, col2 = st.columns(2)

for idx, (feature, (min_val, max_val)) in enumerate(feature_ranges.items()):
    with (col1 if idx % 2 == 0 else col2):
        user_input[feature] = st.slider(
            label=feature.replace("_", " ").title(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=(max_val - min_val) / 100
        )

if st.button("Predict"):
    try:
        unpacked_input = Cancer_Features(**user_input)  
        result = predict_cancer(unpacked_input)
        prediction = result["prediction"][0]
        st.success(f"Prediction: {'Malignent' if prediction == 1 else 'Benign'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# st.markdown("### User Input Summary")
# st.write(user_input)

cleaned_str = re.sub(r'\s*\n\s*', ' ', conf_matrix_str)  # Remove newlines and extra spaces
cleaned_str = re.sub(r'(\d)\s+(\d)', r'\1, \2', cleaned_str)  # Fix spaces between digits

# Step 2: Ensure correct list format by adding brackets
fixed =  cleaned_str.strip() 
corrected_string = fixed.replace('] [', '], [')
fixed_matrix = eval(corrected_string)


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
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
