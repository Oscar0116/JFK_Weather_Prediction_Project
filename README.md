# JFK Weather Prediction Project

This project focuses on predicting weather at JFK Airport using the NOAA Weather Dataset. The goal is to preprocess the data, perform exploratory data analysis (EDA), and create models to make accurate predictions.

## Project Structure
```
📂 rainfall_prediction_classifier_project
├── 📂 data                  # Raw and processed data
   ├── 📂 processed
   ├── 📂 raw          
├── 📂 images                # Saved plot images
├── 📂 models                # Saved models
├── 📂 scripts               # Python scripts for different stages
│   ├── 01_download_noaa.py                   # Load and preprocess the dataset
│   ├── 02_explore_data.py                    # Handle missing values and clean data
│   ├── 03_data_preprocessing.py              # Feature extraction and transformation
│   ├── 04_exploratory_data_analysis.py       # Define the preprocessing pipeline
│   ├── 05_linear_regression.py               # Train linear regression model
│   ├── 06_improved_models.py                 # Trains and evaluates multiple regression models
│   ├── 07_model_evaluation.py                # Evaluate multiple models and identify the best one
└── README.md                  # Project summary and instructions
```
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
