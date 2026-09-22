import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("FIRMS_API_KEY")

#tamil nadu only
BBOX = "76.2,8.0,80.4,13.6"

# Satellite sources - VIIRS gives high resolution (375m)
# Querying all 3 operational VIIRS satellites (SNPP, NOAA-20, NOAA-21) provides complete coverage
SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT"]

# Number of days of data to fetch (max 5 for NASA FIRMS area API)
DAYS = 5

def fetch_firms_data():
    if not API_KEY:
        print("ERROR: FIRMS_API_KEY not found. Check your .env file.")
        return None

    all_dfs = []
    for source in SOURCES:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{source}/{BBOX}/{DAYS}"
        print(f"Fetching data from FIRMS API for {source}...")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            from io import StringIO
            df_src = pd.read_csv(StringIO(response.text))
            if len(df_src) > 0 and "latitude" in df_src.columns:
                print(f"  -> Got {len(df_src)} records from {source}")
                all_dfs.append(df_src)
            else:
                print(f"  -> No data returned for {source}")
        except Exception as e:
            print(f"  -> Warning: failed to fetch {source}: {e}")

    if not all_dfs:
        print("ERROR: No data fetched from any FIRMS source.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Deduplicate matching detections
    dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in combined_df.columns]
    combined_df.drop_duplicates(subset=dedup_cols, inplace=True)

    output_path = "data/raw/firms_hotspots.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nTotal combined & deduplicated hotspot records: {len(combined_df)}")
    print(f"Saved to: {output_path}")

    return combined_df

if __name__ == "__main__":
    fetch_firms_data()