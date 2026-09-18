import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

def load_polygons(csv_path):
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

    before = len(df)
    df = df[df["geometry"].notna()]
    after = len(df)
    print(f"  Loaded {after}/{before} valid polygons from {csv_path} ({before - after} skipped)")

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf

def classify_hotspots():
    print("Loading fused hotspots (with industrial proximity already computed)...")
    hotspots = pd.read_csv("data/processed/fused_hotspots.csv")

    print("Loading forest and farmland polygons...")
    forest = load_polygons("data/raw/osm_forest.csv")
    farmland = load_polygons("data/raw/osm_farmland.csv")

    hotspots["geometry"] = hotspots.apply(lambda r: Point(r["longitude"], r["latitude"]), axis=1)
    hotspots_gdf = gpd.GeoDataFrame(hotspots, geometry="geometry", crs="EPSG:4326")

    # ~1km buffer around each point before checking overlap
    hotspots_gdf["geometry"] = hotspots_gdf.geometry.buffer(0.01)

    print("Checking which hotspots fall inside forest polygons...")
    in_forest = gpd.sjoin(hotspots_gdf, forest[["geometry"]], how="left", predicate="intersects")
    forest_flag = in_forest.groupby(in_forest.index)["index_right"].apply(lambda x: x.notna().any())
    hotspots_gdf["in_forest"] = hotspots_gdf.index.map(forest_flag).fillna(False)

    print("Checking which hotspots fall inside farmland polygons...")
    in_farmland = gpd.sjoin(hotspots_gdf, farmland[["geometry"]], how="left", predicate="intersects")
    farmland_flag = in_farmland.groupby(in_farmland.index)["index_right"].apply(lambda x: x.notna().any())
    hotspots_gdf["in_farmland"] = hotspots_gdf.index.map(farmland_flag).fillna(False)

    def assign_final_class(row):
        if row["preliminary_class"] == "likely_industrial":
            return "industrial_fire"
        elif row["in_forest"]:
            return "wildfire"
        elif row["in_farmland"]:
            return "agricultural_burn"
        else:
            return "unclassified"

    hotspots_gdf["final_class"] = hotspots_gdf.apply(assign_final_class, axis=1)

    output_path = "data/processed/final_classified_hotspots.csv"
    hotspots_gdf.drop(columns="geometry").to_csv(output_path, index=False)

    print(f"\nSaved final classification to: {output_path}")
    print("\nFinal class distribution:")
    print(hotspots_gdf["final_class"].value_counts())

    return hotspots_gdf

if __name__ == "__main__":
    classify_hotspots()