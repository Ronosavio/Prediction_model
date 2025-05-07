import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from backend.config import settings
import joblib




def lg_data():
    data = pd.read_csv(settings.bank_data_path)

    # Map categorical binary/ternary fields
    data['loan'] = data['loan'].map({'yes': 1, 'unknown': 0, 'no': -1})
    data['housing'] = data['housing'].map({'yes': 1, 'unknown': 0, 'no': -1})
    data['poutcome'] = data['poutcome'].map({'success': 1, 'nonexistent': 0, 'failure': -1})
    data['default'] = data['default'].map({'no': -1, 'yes': 1, 'unknown': 0})

    # Drop unwanted columns
    drop_cols = ['age', 'marital', 'education', 'contact', 'month', 'day_of_week',
                 'campaign', 'previous', 'pdays', 'nr_employed']
    data.drop(columns=drop_cols, inplace=True)

    # One-hot encode 'job' and drop first to avoid multicollinearity
    job_data = pd.get_dummies(data['job'], drop_first=True)

    # Combine data
    features = pd.concat([data.drop(columns=['job', 'y']), job_data], axis=1)
    X = features
    Y = data['y']

    return X, Y


    
def apply_lg_model_bank():
    X, Y = lg_data()
     # create a scalar object  
    scaler = StandardScaler()
    #. fit scaler to the data and transform the data 
    X_scaled = scaler.fit_transform(X)
    X_train, x_test,  Y_train, y_test = train_test_split(X_scaled, Y, test_size= 0.2, random_state=42 )
    #create the logistic regression model 
    model = LogisticRegression(C=15, class_weight='balanced')
    #Train the model on training data 
    model.fit(X_train, Y_train)
    #predicting the target variable on test data
    y_predict = model.predict(x_test)
    #Evaluvation  of the model 
    accuracy = accuracy_score(y_test, y_predict)
    joblib.dump(model, settings.bank_lg_trained_model)
    joblib.dump(scaler, settings.bank_sc_trained_model)
    confuse = confusion_matrix(y_test, y_predict)
    report = classification_report(y_test, y_predict, output_dict=True)
    return accuracy, report, confuse

