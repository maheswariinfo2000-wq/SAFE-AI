import requests
import pandas as pd
import time

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {
    "User-Agent": "SIH-Industrial-Fire-Detection/1.0"
}

# Major Indian industrial regions (south, west, north, east)
# Covers key clusters: Gujarat, Maharashtra, Jharkhand/Odisha (steel belt),
# Andhra/Vizag, Tamil Nadu, Rajasthan, UP/NCR
REGIONS = {
    "gujarat": (20.0, 68.0, 24.5, 74.5),
    "maharashtra": (15.5, 72.5, 22.0, 80.5),
    "jharkhand_odisha": (17.5, 82.0, 25.5, 87.5),
    "andhra_telangana": (12.5, 76.5, 19.5, 84.5),
    "tamil_nadu": (8.0, 76.5, 13.5, 80.5),
    "rajasthan": (23.0, 69.0, 30.0, 78.5),
    "up_ncr": (25.5, 76.5, 30.5, 84.5),
}

def build_query(bbox):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:60];
    (
      way["landuse"="industrial"]({s},{w},{n},{e});
      way["power"="plant"]({s},{w},{n},{e});
      way["man_made"="works"]({s},{w},{n},{e});
    );
    out center;
    """

def fetch_region(region_name, bbox):
    query = build_query(bbox)
    for mirror in OVERPASS_MIRRORS:
        try:
            print(f"  Trying {mirror} for {region_name}...")
            response = requests.post(
                mirror,
                data={"data": query},
                headers=HEADERS,
                timeout=70
            )
            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])
                print(f"  Success: {len(elements)} features found")
                return elements
            else:
                print(f"  Failed with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
        time.sleep(2)
    print(f"  All mirrors failed for {region_name}, skipping.")
    return []

def fetch_osm_industrial():
    all_records = []

    for region_name, bbox in REGIONS.items():
        print(f"\nFetching region: {region_name}")
        elements = fetch_region(region_name, bbox)

        for el in elements:
            if "center" in el:
                tags = el.get("tags", {})
                all_records.append({
                    "osm_id": el["id"],
                    "region": region_name,
                    "type": tags.get("landuse") or tags.get("power") or tags.get("man_made"),
                    "name": tags.get("name", "unnamed"),
                    "latitude": el["center"]["lat"],
                    "longitude": el["center"]["lon"]
                })

        time.sleep(3)  # be polite to the free server between regions

    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset="osm_id", inplace=True)

    output_path = "data/raw/osm_industrial.csv"
    df.to_csv(output_path, index=False)
    print(f"\nTotal saved: {len(df)} unique records to {output_path}")
    print(df.head())

    return df

if __name__ == "__main__":
    fetch_osm_industrial()