import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Define file paths
PROJECT_DIR = "C:\\Users\\oscar\\OneDrive\\Escritorio\\Python_Projects\\JFK_Weather_Prediction_Project"
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "test_data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load test dataset
test_data = pd.read_csv(TEST_DATA_PATH)

# Define predictors and response variable
predictors = ["relative_humidity", "dry_bulb_temp_f", "wind_speed", "station_pressure"]
response = "precip"

# Drop rows with NaNs in predictors or target variable
test_data = test_data.dropna(subset=predictors + [response])
X_test = test_data[predictors]
y_test = test_data[response]

# Model paths
model_files = {
    "linear_regression_multivariate": "linear_regression_multivariate.pkl",
    "ridge_regression": "ridge_regression.pkl",
    "lasso_regression": "lasso_regression.pkl",
    "polynomial_regression": "polynomial_regression.pkl"
}

# Evaluate models
results = []
for model_name, model_file in model_files.items():
    model_path = os.path.join(MODELS_DIR, model_file)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    if "polynomial" in model_name:
        poly = PolynomialFeatures(degree=2)
        X_test_transformed = poly.fit_transform(X_test)
    else:
        X_test_transformed = X_test
    
    y_pred = model.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append([model_name, mse, rmse, r2])

# Create results DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R²"])
print(results_df)

# Save results to CSV
results_csv_path = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
results_df.to_csv(results_csv_path, index=False)

# Find best model (lowest RMSE)
best_model = results_df.loc[results_df["RMSE"].idxmin()]
best_model_info = f"Best Model: {best_model['Model']} with RMSE = {best_model['RMSE']:.4f}"
print(best_model_info)

# Save best model info to a text file
best_model_path = os.path.join(RESULTS_DIR, "best_model.txt")
with open(best_model_path, "w", encoding="utf-8") as f:
    f.write(best_model_info + "\n")

print(f"Results saved to: {RESULTS_DIR}")
