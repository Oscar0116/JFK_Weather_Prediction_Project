import pandas as pd
import os

# Define the file path
file_path = os.path.join("data", "raw", "noaa-weather-sample-data", "jfk_weather_sample.csv")

# Load the dataset
df = pd.read_csv(file_path)

# Display basic info
print(df.info())
print(df.head())  # Show first few rows
