import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import os

def load_polygons(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)

    def safe_load_wkt(wkt_str):
        try:
            geom = wkt.loads(wkt_str)
            if not geom.is_valid:
                geom = geom.buffer(0)
            return geom
        except Exception:
            return None

    df["geometry"] = df["geometry_wkt"].apply(safe_load_wkt)
    df = df[df["geometry"].notna()]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf

def classify_hotspots():
    print("Loading fused hotspots...")
    hotspots = pd.read_csv("data/processed/fused_hotspots.csv")

    # 1. Match nearest Tamil Nadu settlement/village/town
    places_path = "data/raw/tamilnadu_places.csv"
    if os.path.exists(places_path):
        print("Matching hotspots to 25,000+ Tamil Nadu settlements...")
        places = pd.read_csv(places_path)
        p_rad = np.radians(places[["latitude", "longitude"]].values)
        tree = BallTree(p_rad, metric="haversine")
        h_rad = np.radians(hotspots[["latitude", "longitude"]].values)
        dists, idxs = tree.query(h_rad, k=1)
        
        nearest_idx = idxs.flatten()
        hotspots["nearest_place_km"] = dists.flatten() * 6371.0
        hotspots["nearest_place_name"] = places.iloc[nearest_idx]["name"].values
        hotspots["nearest_place_type"] = places.iloc[nearest_idx]["place_type"].values
    else:
        hotspots["nearest_place_km"] = 999.0
        hotspots["nearest_place_name"] = "Tamil Nadu rural area"
        hotspots["nearest_place_type"] = "locality"

    # 2. Check forest and farmland polygons
    forest = load_polygons("data/raw/osm_forest.csv")
    farmland = load_polygons("data/raw/osm_farmland.csv")

    hotspots["geometry"] = hotspots.apply(lambda r: Point(r["longitude"], r["latitude"]), axis=1)
    hotspots_gdf = gpd.GeoDataFrame(hotspots, geometry="geometry", crs="EPSG:4326")
    hotspots_buf = hotspots_gdf.copy()
    hotspots_buf["geometry"] = hotspots_buf.geometry.buffer(0.015) # ~1.5km buffer

    if forest is not None and len(forest) > 0:
        in_forest = gpd.sjoin(hotspots_buf, forest[["geometry"]], how="left", predicate="intersects")
        forest_flag = in_forest.groupby(in_forest.index)["index_right"].apply(lambda x: x.notna().any())
        hotspots["in_forest"] = hotspots.index.map(forest_flag).fillna(False)
    else:
        hotspots["in_forest"] = False

    if farmland is not None and len(farmland) > 0:
        in_farmland = gpd.sjoin(hotspots_buf, farmland[["geometry"]], how="left", predicate="intersects")
        farmland_flag = in_farmland.groupby(in_farmland.index)["index_right"].apply(lambda x: x.notna().any())
        hotspots["in_farmland"] = hotspots.index.map(farmland_flag).fillna(False)
    else:
        hotspots["in_farmland"] = False

    # 3. Comprehensive classification & place identification
    def assign_class_and_place(row):
        # Industrial
        if row["nearest_industrial_km"] <= 2.0:
            ind_name = row.get("nearest_industrial_name", "")
            ind_disp = ind_name if ind_name and ind_name != "unnamed" else "Industrial facility"
            return "industrial_fire", f"Industrial facility ({ind_disp})"
        # Forest / Wildfire
        elif row["in_forest"]:
            return "wildfire", f"Forest / Woodland reserve near {row['nearest_place_name']}"
        # Agricultural
        elif row["in_farmland"] or (row["nearest_place_type"] in ["village", "hamlet"] and row["nearest_place_km"] <= 5.0):
            dist_str = f"{row['nearest_place_km']:.1f} km"
            return "agricultural_burn", f"Agricultural cropland near {row['nearest_place_name']} ({row['nearest_place_type']}, {dist_str})"
        # Urban / Commercial settlement
        elif row["nearest_place_type"] in ["city", "town", "suburb"] and row["nearest_place_km"] <= 2.5:
            return "urban_commercial_fire", f"Urban / Commercial area ({row['nearest_place_name']})"
        # High FRP near industry cluster
        elif row["frp"] >= 15.0 and row["nearest_industrial_km"] <= 6.0:
            return "industrial_fire", f"Industrial cluster vicinity ({row.get('nearest_industrial_name', 'manufacturing zone')})"
        # Scrub / open rural biomass
        else:
            return "scrub_biomass_burn", f"Rural scrubland / open biomass near {row['nearest_place_name']} ({row['nearest_place_km']:.1f} km)"

    res = hotspots.apply(assign_class_and_place, axis=1)
    hotspots["final_class"] = [r[0] for r in res]
    hotspots["place_description"] = [r[1] for r in res]

    output_path = "data/processed/final_classified_hotspots.csv"
    hotspots.drop(columns="geometry", errors="ignore").to_csv(output_path, index=False)

    print(f"\nSaved final classification to: {output_path}")
    print("\nFinal class distribution:")
    print(hotspots["final_class"].value_counts())

    return hotspots

if __name__ == "__main__":
    classify_hotspots()