import os
import requests
import tarfile

def download_and_extract(url, download_path, extract_path):
    """Download and extract a .tar.gz file from a URL."""
    os.makedirs(download_path, exist_ok=True)  # Ensure directories exist
    os.makedirs(extract_path, exist_ok=True)

    file_name = os.path.join(download_path, "noaa-weather-sample-data.tar.gz")

    # Download the file
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete!")
    else:
        print("Failed to download the file.")
        return

    # Extract the file
    print("Extracting dataset...")
    try:
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=extract_path, filter="data")  # Fix for Python 3.14+
        print("Extraction complete!")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return

    # Verify extraction
    extracted_files = os.listdir(extract_path)
    if extracted_files:
        print("Extracted files:", extracted_files)
    else:
        print("Warning: No files found after extraction!")

    # Remove the tar.gz file after extraction
    os.remove(file_name)
    print("Cleaned up downloaded file.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")

    download_and_extract(
        "https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz",
        DATA_RAW_PATH,
        DATA_RAW_PATH
    )
