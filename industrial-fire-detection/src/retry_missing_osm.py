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

# Only the 2 regions that failed
MISSING_REGIONS = {
    "jharkhand_odisha": (17.5, 82.0, 25.5, 87.5),
    "rajasthan": (23.0, 69.0, 30.0, 78.5),
}

def build_query(bbox):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:90];
    (
      way["landuse"="industrial"]({s},{w},{n},{e});
      way["power"="plant"]({s},{w},{n},{e});
      way["man_made"="works"]({s},{w},{n},{e});
    );
    out center;
    """

def fetch_region(region_name, bbox, max_retries=3):
    query = build_query(bbox)

    for attempt in range(1, max_retries + 1):
        for mirror in OVERPASS_MIRRORS:
            try:
                print(f"  Attempt {attempt}: Trying {mirror} for {region_name}...")
                response = requests.post(
                    mirror,
                    data={"data": query},
                    headers=HEADERS,
                    timeout=100
                )
                if response.status_code == 200:
                    data = response.json()
                    elements = data.get("elements", [])
                    print(f"  Success: {len(elements)} features found")
                    return elements
                elif response.status_code == 429:
                    print(f"  Rate limited (429). Waiting 30s before retry...")
                    time.sleep(30)
                else:
                    print(f"  Failed with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"  Error: {e}")
            time.sleep(5)

    print(f"  Giving up on {region_name} after {max_retries} attempts.")
    return []

def retry_missing():
    all_new_records = []

    for region_name, bbox in MISSING_REGIONS.items():
        print(f"\nRetrying region: {region_name}")
        elements = fetch_region(region_name, bbox)

        for el in elements:
            if "center" in el:
                tags = el.get("tags", {})
                all_new_records.append({
                    "osm_id": el["id"],
                    "region": region_name,
                    "type": tags.get("landuse") or tags.get("power") or tags.get("man_made"),
                    "name": tags.get("name", "unnamed"),
                    "latitude": el["center"]["lat"],
                    "longitude": el["center"]["lon"]
                })

        time.sleep(5)

    new_df = pd.DataFrame(all_new_records)

    if new_df.empty:
        print("\nNo new records fetched. Servers may still be busy - try again later.")
        return

    # Merge with existing file
    existing_path = "data/raw/osm_industrial.csv"
    existing_df = pd.read_csv(existing_path)

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset="osm_id", inplace=True)

    combined.to_csv(existing_path, index=False)
    print(f"\nAdded {len(new_df)} new records.")
    print(f"Total records now: {len(combined)}")
    print(combined['region'].value_counts())

if __name__ == "__main__":
    retry_missing()