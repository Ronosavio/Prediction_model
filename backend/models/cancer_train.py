import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from backend.config import settings
import joblib

def lg_data():
    data = pd.read_csv(settings.cancer_data_path)
    columns_to_drop = ['id', 'Unnamed: 32']
    data.drop([col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)
    data.diagnosis = [1 if value == 'M' else 0 for value in data.diagnosis]
    Y = data['diagnosis'] # target variable
    X = data.drop(['diagnosis'], axis =1)
    return X, Y

    
def apply_lg_model_cancer():
    X, Y = lg_data()
     # create a scalar object  
    scaler = StandardScaler()
    #. fit scaler to the data and transform the data 
    X_scaled = scaler.fit_transform(X)
    X_train, x_test,  Y_train, y_test = train_test_split(X_scaled, Y, test_size= 0.3, random_state=42 )
    #create the logistic regression model 
    model = LogisticRegression()
    #Train the model on training data 
    model.fit(X_train, Y_train)
    #predicting the target variable on test data
    y_predict = model.predict(x_test)
    #Evaluvation  of the model 
    accuracy = accuracy_score(y_test, y_predict)
    joblib.dump(model, settings.cancer_lg_trained_model)
    joblib.dump(scaler, settings.cancer_sc_trained_model)
    confuse = confusion_matrix(y_test, y_predict)
    report = classification_report(y_test, y_predict, output_dict=True)
    return accuracy, report, confuse

