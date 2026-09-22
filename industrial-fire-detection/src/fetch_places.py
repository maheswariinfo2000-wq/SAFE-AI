import requests
import pandas as pd
import os
import time

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "TamilNaduPlaceClassifier/1.0"}

def fetch_tamilnadu_places():
    output_path = "data/raw/tamilnadu_places.csv"
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"Places dataset already exists with {len(df)} places.")
        return df

    print("Fetching Tamil Nadu towns, villages, and cities from OpenStreetMap...")
    query = """
    [out:json][timeout:60];
    (
      node["place"="city"](8.0, 76.2, 13.6, 80.4);
      node["place"="town"](8.0, 76.2, 13.6, 80.4);
      node["place"="village"](8.0, 76.2, 13.6, 80.4);
      node["place"="suburb"](8.0, 76.2, 13.6, 80.4);
    );
    out body;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            elements = r.json().get("elements", [])
            print(f"Got {len(elements)} place nodes from OSM.")
            records = []
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name")
                if name:
                    records.append({
                        "osm_id": el["id"],
                        "name": name,
                        "place_type": tags.get("place", "locality"),
                        "latitude": el["lat"],
                        "longitude": el["lon"],
                        "district": tags.get("is_in:state_district") or tags.get("addr:district") or tags.get("district") or ""
                    })
            df = pd.DataFrame(records)
            df.drop_duplicates(subset=["name", "latitude", "longitude"], inplace=True)
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} places to {output_path}")
            return df
        else:
            print(f"Overpass returned status {r.status_code}")
    except Exception as e:
        print(f"Error fetching places: {e}")
    return None

if __name__ == "__main__":
    fetch_tamilnadu_places()
