import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

# Drop rows with NaNs in predictors or target variable
train_data = train_data.dropna(subset=predictors + [response])
X = train_data[predictors]
y = train_data[response]

def plot_and_save(model, X, y, title, filename):
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("Actual Precipitation")
    plt.ylabel("Predicted Precipitation")
    plt.title(title)
    plt.savefig(os.path.join(IMAGES_DIR, filename))
    plt.close()

### 1️⃣ Multivariate Linear Regression ###
multi_model = LinearRegression()
multi_model.fit(X, y)

multi_model_path = os.path.join(MODELS_DIR, "linear_regression_multivariate.pkl")
with open(multi_model_path, "wb") as f:
    pickle.dump(multi_model, f)
plot_and_save(multi_model, X, y, "Multivariate Linear Regression", "linear_regression_multivariate.png")

### 2️⃣ Ridge Regression (L2 Regularization) ###
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

ridge_model_path = os.path.join(MODELS_DIR, "ridge_regression.pkl")
with open(ridge_model_path, "wb") as f:
    pickle.dump(ridge_model, f)
plot_and_save(ridge_model, X, y, "Ridge Regression", "ridge_regression.png")

### 3️⃣ Lasso Regression (L1 Regularization) ###
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)

lasso_model_path = os.path.join(MODELS_DIR, "lasso_regression.pkl")
with open(lasso_model_path, "wb") as f:
    pickle.dump(lasso_model, f)
plot_and_save(lasso_model, X, y, "Lasso Regression", "lasso_regression.png")

### 4️⃣ Polynomial Regression (Degree 2) ###
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

poly_model_path = os.path.join(MODELS_DIR, "polynomial_regression.pkl")
with open(poly_model_path, "wb") as f:
    pickle.dump(poly_model, f)
plot_and_save(poly_model, X_poly, y, "Polynomial Regression (Degree 2)", "polynomial_regression.png")

print("✅ Improved models trained, saved, and plots generated!")
