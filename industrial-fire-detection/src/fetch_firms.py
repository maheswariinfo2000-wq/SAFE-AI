import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("FIRMS_API_KEY")

# India bounding box (min_lon, min_lat, max_lon, max_lat)
BBOX = "68,6,97,37"

# Satellite source - VIIRS gives better resolution (375m) than MODIS
SOURCE = "VIIRS_SNPP_NRT"

# Number of days of data to fetch (max 10 for near-real-time)
DAYS = 3

def fetch_firms_data():
    if not API_KEY:
        print("ERROR: FIRMS_API_KEY not found. Check your .env file.")
        return None

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{SOURCE}/{BBOX}/{DAYS}"

    print("Fetching data from FIRMS API...")
    response = requests.get(url)
    response.raise_for_status()

    # Save raw CSV
    output_path = "data/raw/firms_hotspots.csv"
    with open(output_path, "w") as f:
        f.write(response.text)

    # Load into pandas to verify
    df = pd.read_csv(output_path)
    print(f"Fetched {len(df)} hotspot records")
    print(f"Saved to: {output_path}")
    print("\nSample data:")
    print(df.head())
    print("\nColumns:", list(df.columns))

    return df

if __name__ == "__main__":
    fetch_firms_data()