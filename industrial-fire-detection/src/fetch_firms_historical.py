import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from io import StringIO

load_dotenv()
API_KEY = os.getenv("FIRMS_API_KEY")

BBOX = "68,6,97,37"
SOURCE = "VIIRS_SNPP_NRT"
CHUNK_DAYS = 5
TOTAL_DAYS = 60

def fetch_chunk(start_date):
    date_str = start_date.strftime("%Y-%m-%d")
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{SOURCE}/{BBOX}/{CHUNK_DAYS}/{date_str}"
    print(f"  Fetching chunk starting {date_str}...")

    response = requests.get(url)

    if response.status_code != 200:
        print(f"  Status: {response.status_code}")
        print(f"  Response body: {response.text[:300]}")
        response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    print(f"  Got {len(df)} records")
    return df

def fetch_historical():
    today = datetime.utcnow().date()
    all_chunks = []

    num_chunks = TOTAL_DAYS // CHUNK_DAYS   # 60 // 5 = 12 chunks
    for i in range(num_chunks):
        start_date = today - timedelta(days=(i + 1) * CHUNK_DAYS)
        try:
            chunk = fetch_chunk(start_date)
            all_chunks.append(chunk)
        except Exception as e:
            print(f"  Failed chunk {start_date}: {e}")
        time.sleep(2)

    if not all_chunks:
        print("\nNo data fetched at all. Check the error messages above.")
        return None

    combined = pd.concat(all_chunks, ignore_index=True)
    combined.drop_duplicates(inplace=True)

    output_path = "data/raw/firms_historical.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} historical records to {output_path}")
    return combined

if __name__ == "__main__":
    fetch_historical()