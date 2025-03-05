# JFK Weather Prediction Project

This project focuses on predicting weather at JFK Airport using the NOAA Weather Dataset. The goal is to preprocess the data, perform exploratory data analysis (EDA), and create models to make accurate predictions.

## Project Structure
JFK_Weather_Prediction_Project
+---data
|   +---processed
|   |       test_data.csv
|   |       train_data.csv
|   |
|   \---raw
|       |   ._noaa-weather-sample-data
|       |
|       \---noaa-weather-sample-data
|               ._LICENSE.txt
|               ._README.txt
|               jfk_weather_sample.csv
|               LICENSE.txt
|               README.txt
|
+---images
|       boxplots.png
|       histograms.png
|       lasso_regression.png
|       linear_regression_dry_bulb_temp_f.png
|       linear_regression_multivariate.png
|       linear_regression_relative_humidity.png
|       linear_regression_station_pressure.png
|       linear_regression_wind_speed.png
|       polynomial_regression.png
|       ridge_regression.png
|
+---models
|       lasso_regression.pkl
|       linear_regression_dry_bulb_temp_f.pkl
|       linear_regression_multivariate.pkl
|       linear_regression_relative_humidity.pkl
|       linear_regression_station_pressure.pkl
|       linear_regression_wind_speed.pkl
|       polynomial_regression.pkl
|       ridge_regression.pkl
|
+---notebooks
+---results
|       best_model.txt
|       model_evaluation_results.csv
|
+---scripts
        01_download_noaa.py
        02_explore_data.py
        03_data_preprocessing.py
        04_exploratory_data_analysis.py
        05_linear_regression.py
        06_improved_models.py
        07_model_evaluation.py

## Model Results

The following models were evaluated:

| Model                    | MSE                | RMSE               | R²                |
|--------------------------|--------------------|--------------------|-------------------|
| Linear Regression         | 0.000494           | 0.0222             | 0.0547            |
| Ridge Regression          | 0.000494           | 0.0222             | 0.0547            |
| Lasso Regression          | 0.000501           | 0.0224             | 0.0419            |
| Polynomial Regression     | 0.000488           | 0.0221             | 0.0675            |

**Best Model:** Polynomial Regression with RMSE = 0.0221

## Installation

To run this project locally, install the necessary dependencies:

1. Create and activate a virtual environment (if not done already):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   
2. Install the required packages:
    pip install -r requirements.txt

## License

This project is licensed under the MIT License.
