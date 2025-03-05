import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define file paths
BASE_DIR = r"C:\Users\oscar\OneDrive\Escritorio\Python_Projects\JFK_Weather_Prediction_Project"
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "noaa-weather-sample-data", "jfk_weather_sample.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Load dataset with error handling
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"File not found: {RAW_DATA_PATH}")

df = pd.read_csv(RAW_DATA_PATH)

# Select relevant columns
SELECTED_COLUMNS = {
    "HOURLYRelativeHumidity": "relative_humidity",
    "HOURLYDRYBULBTEMPF": "dry_bulb_temp_f",
    "HOURLYPrecip": "precip",
    "HOURLYWindSpeed": "wind_speed",
    "HOURLYStationPressure": "station_pressure"
}
df_subset = df[list(SELECTED_COLUMNS.keys())].rename(columns=SELECTED_COLUMNS)

# Clean 'precip' column
df_subset["precip"] = df_subset["precip"].replace("T", "0.0").str.replace(r"s$", "", regex=True)
df_subset["precip"] = pd.to_numeric(df_subset["precip"], errors="coerce")

# Split data into training (80%) and testing (20%)
train_data, test_data = train_test_split(df_subset, test_size=0.2, random_state=1234)

# Save datasets
train_path = os.path.join(PROCESSED_DATA_PATH, "train_data.csv")
test_path = os.path.join(PROCESSED_DATA_PATH, "test_data.csv")
train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

# Print confirmation
print(f"Training set shape: {train_data.shape}")
print(f"Testing set shape: {test_data.shape}")
print(f"Processed datasets saved successfully!\n- Training data: {train_path}\n- Testing data: {test_path}")

# Verify if files were saved
print("Train data exists:", os.path.exists(train_path))
print("Test data exists:", os.path.exists(test_path))
