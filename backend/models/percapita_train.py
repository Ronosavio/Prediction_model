import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.config import settings
import joblib

def slr_data():
    df = pd.read_csv(settings.percapita_data_path)
    X = df[['year']]
    y = df['pci']
    return X, y

# Train the model
def apply_slr_model():
    X, y = slr_data()
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, settings.percapita_slr_trained_model)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create years for prediction range (1970 to 2016)
    years = np.array(range(1970, 2017)).reshape(-1, 1)
    predictions = model.predict(years)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X.values.flatten(), y=y, label="Actual", color='blue')  # .values.flatten() to extract NumPy array
    sns.lineplot(x=years.flatten(), y=predictions, color="red", label="Regression Line")
    plt.xlabel("Year")
    plt.ylabel("Per Capita Income")
    plt.title("Year vs Per Capita Income")
    plt.legend()
    
    # Save the plot
    plt.savefig(settings.percapita_plot_path)
    plt.close()
    rmse = np.sqrt(mean_squared_error(y, y_pred)) 
    return r2_score(y, y_pred), rmse


