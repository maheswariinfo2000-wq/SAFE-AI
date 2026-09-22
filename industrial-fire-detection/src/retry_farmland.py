import requests
import pandas as pd
import time

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {"User-Agent": "SIH-Industrial-Fire-Detection/1.0"}

REGIONS = {
    "tn_north":   (12.5, 79.0, 13.6, 80.4),
    "tn_west":    (10.5, 76.8, 12.5, 78.6),
    "tn_central": (9.5, 78.0, 12.5, 79.8),
    "tn_south":   (8.0, 77.0, 9.5, 79.0),
}

def build_query(bbox, tag_filter):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:90];
    (
      way{tag_filter}({s},{w},{n},{e});
    );
    out geom;
    """

def fetch_region(region_name, bbox, tag_filter, max_retries=4):
    query = build_query(bbox, tag_filter)
    for attempt in range(1, max_retries + 1):
        for mirror in OVERPASS_MIRRORS:
            try:
                print(f"  Attempt {attempt}: {mirror} for {region_name}...")
                response = requests.post(mirror, data={"data": query}, headers=HEADERS, timeout=100)
                if response.status_code == 200:
                    elements = response.json().get("elements", [])
                    print(f"  Success: {len(elements)} features")
                    return elements
                elif response.status_code == 429:
                    print("  Rate limited, waiting 30s...")
                    time.sleep(30)
                else:
                    print(f"  Failed status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"  Error: {e}")
            time.sleep(8)
    print(f"  Giving up on {region_name}")
    return []

def fetch_farmland():
    all_records = []
    for region_name, bbox in REGIONS.items():
        print(f"\nFetching farmland - region: {region_name}")
        elements = fetch_region(region_name, bbox, '["landuse"="farmland"]')

        for el in elements:
            if "geometry" in el and len(el["geometry"]) >= 3:
                coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
                all_records.append({
                    "osm_id": el["id"],
                    "region": region_name,
                    "landcover_type": "farmland",
                    "geometry_wkt": "POLYGON((" + ",".join(f"{x} {y}" for x, y in coords) + "))"
                })
        time.sleep(5)

    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset="osm_id", inplace=True)
    df.to_csv("data/raw/osm_farmland.csv", index=False)
    print(f"\nSaved {len(df)} farmland polygons to data/raw/osm_farmland.csv")

if __name__ == "__main__":
    fetch_farmland()