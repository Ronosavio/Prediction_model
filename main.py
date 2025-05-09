import subprocess
import os
from fastapi import FastAPI
from backend.models.bank_trainer import apply_lg_model_bank
from backend.models.cancer_train import apply_lg_model_cancer
from backend.models.percapita_train import apply_slr_model

app = FastAPI()

@app.get("/")
def welcome():
    return {"Welcome to my prediction API"}

@app.get("/bank")
def bank_model():
    accuracy, report, confuse = apply_lg_model_bank()
    base_path = "performance_details\\"
    with open(os.path.join(base_path, "bank_accuracy.txt"), "w") as f:
        f.write(str(accuracy))
    with open(os.path.join(base_path, "bank_report.txt"), "w") as f:
        f.write(str(report))
    with open(os.path.join(base_path, "bank_confusion_matrix.txt"), "w") as f:
        f.write(str(confuse)) 
    frontend_path = os.path.join(os.path.dirname(__file__),"frontend", "bank_front.py")
    subprocess.Popen(["streamlit", "run", frontend_path], shell=True)
    return {"status": f"Streamlit app launched for bank subscription prediction: Accuracy-> {accuracy*100}"}

@app.get("/cancer")
def  cancer_model():
    accuracy , report , confuse = apply_lg_model_cancer()
    base_path = "performance_details\\"
    with open(os.path.join(base_path, "cancer_accuracy.txt"), "w") as f:
        f.write(str(accuracy))
    with open(os.path.join(base_path, "cancer_report.txt"), "w") as f:
        f.write(str(report))
    with open(os.path.join(base_path, "cancer_confusion_matrix.txt"), "w") as f:
        f.write(str(confuse)) 
    frontend_path = os.path.join(os.path.dirname(__file__),"frontend", "cancer_front.py")
    subprocess.Popen(["streamlit", "run", frontend_path], shell=True)
    return {"status": f"Streamlit app launched for cancer prediction: Accuracy-> {accuracy*100}"}

@app.get("/percapita")
def percapita():
    accuracy, rmse = apply_slr_model()
    base_path = "performance_details\\"
    with open(os.path.join(base_path, "percapita_accuracy.txt"), "w") as f:
        f.write(str(accuracy))
    with open(os.path.join(base_path, "percapita_rmse.txt"), "w") as f:
        f.write(str(rmse))
    frontend_path = os.path.join(os.path.dirname(__file__),"frontend", "percapita_front.py")
    subprocess.Popen(["streamlit", "run", frontend_path], shell=True)
    return {"status": f"Streamlit app launched for percaptia income prediction: Accuracy-> {accuracy*100}"}
    
    
    

