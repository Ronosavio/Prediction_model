import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import Cancer_Features
import joblib
from config import settings

model = joblib.load(settings.cancer_load_trained_data_lg)
scaler = joblib.load(settings.cancer_load_trained_data_sc)
def predict_cancer(data: Cancer_Features):
    input_list = [[
        data.radius_mean,
        data.texture_mean,
        data.perimeter_mean,
        data.area_mean,
        data.smoothness_mean,
        data.compactness_mean,
        data.concavity_mean,
        data.concave_points_mean,
        data.symmetry_mean,
        data.fractal_dimension_mean,
        data.radius_se,
        data.texture_se,
        data.perimeter_se,
        data.area_se,
        data.smoothness_se,
        data.compactness_se,
        data.concavity_se,
        data.concave_points_se,
        data.symmetry_se,
        data.fractal_dimension_se,
        data.radius_worst,
        data.texture_worst,
        data.perimeter_worst,
        data.area_worst,
        data.smoothness_worst,
        data.compactness_worst,
        data.concavity_worst,
        data.concave_points_worst,
        data.symmetry_worst,
        data.fractal_dimension_worst
    ]]
    # You can now use input_list for prediction


    # Now you can transform and predict
    scaled_input = scaler.transform(input_list)
    prediction = model.predict(scaled_input)
    return {"prediction": prediction.tolist()}
