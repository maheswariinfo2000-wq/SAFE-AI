import requests
import pandas as pd
import time

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {"User-Agent": "SIH-Industrial-Fire-Detection/1.0"}

# Only the 3 regions that failed
REGIONS = {
    "tn_west":    (10.5, 76.8, 12.5, 78.6),
    "tn_central": (9.5, 78.0, 12.5, 79.8),
    "tn_south":   (8.0, 77.0, 9.5, 79.0),
}

def build_query(bbox, tag_filter):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:120];
    (
      way{tag_filter}({s},{w},{n},{e});
    );
    out geom;
    """

def fetch_region(region_name, bbox, tag_filter, max_retries=6):
    query = build_query(bbox, tag_filter)
    for attempt in range(1, max_retries + 1):
        for mirror in OVERPASS_MIRRORS:
            try:
                print(f"  Attempt {attempt}: {mirror} for {region_name}...")
                response = requests.post(mirror, data={"data": query}, headers=HEADERS, timeout=130)
                if response.status_code == 200:
                    elements = response.json().get("elements", [])
                    print(f"  Success: {len(elements)} features")
                    return elements
                elif response.status_code == 429:
                    print("  Rate limited, waiting 45s...")
                    time.sleep(45)
                else:
                    print(f"  Failed status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"  Error: {e}")
            time.sleep(15)  # longer pause between attempts
    print(f"  Giving up on {region_name} after {max_retries} attempts")
    return []

def fetch_remaining_farmland():
    all_new_records = []
    for region_name, bbox in REGIONS.items():
        print(f"\nFetching farmland - region: {region_name}")
        elements = fetch_region(region_name, bbox, '["landuse"="farmland"]')

        for el in elements:
            if "geometry" in el and len(el["geometry"]) >= 3:
                coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
                all_new_records.append({
                    "osm_id": el["id"],
                    "region": region_name,
                    "landcover_type": "farmland",
                    "geometry_wkt": "POLYGON((" + ",".join(f"{x} {y}" for x, y in coords) + "))"
                })
        time.sleep(10)

    new_df = pd.DataFrame(all_new_records)

    if new_df.empty:
        print("\nNo new records fetched this time. Servers may still be busy — try again in a few minutes.")
        return

    existing_df = pd.read_csv("data/raw/osm_farmland.csv")
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset="osm_id", inplace=True)
    combined.to_csv("data/raw/osm_farmland.csv", index=False)

    print(f"\nAdded {len(new_df)} new records.")
    print(f"Total farmland records now: {len(combined)}")
    print(combined["region"].value_counts())

if __name__ == "__main__":
    fetch_remaining_farmland()