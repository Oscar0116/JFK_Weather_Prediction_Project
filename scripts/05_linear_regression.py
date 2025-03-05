import os
import pickle  # To save models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define file paths
PROJECT_DIR = "C:\\Users\\oscar\\OneDrive\\Escritorio\\Python_Projects\\JFK_Weather_Prediction_Project"
TRAIN_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "train_data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load training dataset
train_data = pd.read_csv(TRAIN_DATA_PATH)

# Define predictors and response variable
predictors = ["relative_humidity", "dry_bulb_temp_f", "wind_speed", "station_pressure"]
response = "precip"

# Check missing values before dropping NaNs
print("Missing values in dataset before dropping NaNs:")
print(train_data[predictors + [response]].isna().sum())

# Drop rows where any predictor or response variable is NaN
train_data = train_data.dropna(subset=predictors + [response])

# Check missing values after cleaning
print("\nMissing values after dropping NaNs:")
print(train_data[predictors + [response]].isna().sum())

# Initialize Matplotlib style
sns.set_style("whitegrid")

# Iterate over predictors to create separate linear regression models
for predictor in predictors:
    # Prepare data
    X = train_data[[predictor]]  # Independent variable
    y = train_data[response]     # Dependent variable

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, f"linear_regression_{predictor}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Predictions
    y_pred = model.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)

    # Plot results
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X[predictor], y=y, alpha=0.5, label="Actual Data")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
    plt.xlabel(predictor)
    plt.ylabel(response)
    plt.title(f"Linear Regression: {response} ~ {predictor}\nMSE: {mse:.5f}")
    plt.legend()
    
    # Save figure
    plot_path = os.path.join(IMAGES_DIR, f"linear_regression_{predictor}.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"✅ Model for {response} ~ {predictor} trained successfully! MSE: {mse:.5f}")
    print(f"📂 Model saved at: {model_path}")
    print(f"📊 Plot saved at: {plot_path}\n")
