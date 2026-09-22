import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

df = pd.read_csv("data/processed/all_tamilnadu_hotspots.csv")
places = pd.read_csv("data/raw/tamilnadu_places.csv")

p_rad = np.radians(places[["latitude", "longitude"]].values)
tree = BallTree(p_rad, metric="haversine")

h_rad = np.radians(df[["latitude", "longitude"]].values)
dists, idxs = tree.query(h_rad, k=1)

df["nearest_place_km"] = dists.flatten() * 6371.0
df["nearest_place_name"] = places.iloc[idxs.flatten()]["name"].values
df["nearest_place_type"] = places.iloc[idxs.flatten()]["place_type"].values

def classify_enhanced(row):
    if row["nearest_industrial_km"] <= 2.0:
        return "industrial_fire", f"Industrial facility ({row.get('nearest_industrial_name', 'industrial zone')})"
    elif row.get("in_forest", False):
        return "wildfire", "Forest / Woodland reserve"
    elif row.get("in_farmland", False) or (row["nearest_place_type"] in ["village", "hamlet"] and row["nearest_place_km"] <= 5.0):
        return "agricultural_burn", f"Agricultural cropland near {row['nearest_place_name']} ({row['nearest_place_type']})"
    elif row["nearest_place_type"] in ["city", "town", "suburb"] and row["nearest_place_km"] <= 2.5:
        return "urban_commercial_fire", f"Urban area ({row['nearest_place_name']})"
    elif row["frp"] >= 15.0 and row["nearest_industrial_km"] <= 6.0:
        return "industrial_fire", f"Industrial cluster near {row.get('nearest_industrial_name', 'manufacturing zone')}"
    else:
        return "scrub_biomass_burn", f"Rural scrubland / open biomass near {row['nearest_place_name']}"

res = df.apply(classify_enhanced, axis=1)
df["enhanced_class"] = [r[0] for r in res]
df["place_description"] = [r[1] for r in res]

print("Enhanced Class distribution across all 1597 hotspots:")
print(df["enhanced_class"].value_counts())
print("\nSample rows:")
for _, r in df.head(8).iterrows():
    print(f"[{r['acq_date']}] FRP {r['frp']} -> Class: {r['enhanced_class']} | Place: {r['place_description']}")
