import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
BASE_DIR = r"C:\Users\oscar\OneDrive\Escritorio\Python_Projects\JFK_Weather_Prediction_Project"
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load the training dataset
train_data = pd.read_csv(TRAIN_DATA_PATH)

# Set plot style
sns.set_style("whitegrid")

# Define numeric columns to visualize
columns = ["relative_humidity", "dry_bulb_temp_f", "precip", "wind_speed", "station_pressure"]

# Create histograms
plt.figure(figsize=(12, 6))
for i, col in enumerate(columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train_data[col], bins=30, kde=True)
    plt.title(f"Histogram of {col}")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "histograms.png"))  # Save image
plt.show()

# Create boxplots
plt.figure(figsize=(12, 6))
for i, col in enumerate(columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=train_data[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "boxplots.png"))  # Save image
plt.show()
