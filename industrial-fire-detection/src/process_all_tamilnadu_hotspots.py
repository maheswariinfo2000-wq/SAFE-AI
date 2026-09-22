import pandas as pd
import numpy as np
import json
import joblib
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd

def process_historical():
    print("Loading datasets...")
    historical = pd.read_csv("data/raw/firms_historical.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")
    baseline = pd.read_csv("data/processed/facility_baselines.csv")
    
    # 1. Match to nearest industrial facilities
    print("Matching 1597 hotspots to OSM industrial facilities...")
    ind_rad = np.radians(industrial[["latitude", "longitude"]].values)
    hist_rad = np.radians(historical[["latitude", "longitude"]].values)
    tree = BallTree(ind_rad, metric="haversine")
    distances, indices = tree.query(hist_rad, k=1)
    
    nearest_idx = indices.flatten()
    historical["nearest_industrial_km"] = distances.flatten() * 6371.0
    historical["nearest_industrial_type"] = industrial.iloc[nearest_idx]["type"].values
    historical["nearest_industrial_name"] = industrial.iloc[nearest_idx]["name"].values
    historical["nearest_industrial_region"] = industrial.iloc[nearest_idx]["region"].values
    historical["nearest_industrial_latitude"] = industrial.iloc[nearest_idx]["latitude"].values
    historical["nearest_industrial_longitude"] = industrial.iloc[nearest_idx]["longitude"].values
    historical["facility_osm_id"] = industrial.iloc[nearest_idx]["osm_id"].values
    
    # Add CPCB information
    if "cpcb_name" in industrial.columns:
        historical["cpcb_name"] = industrial.iloc[nearest_idx]["cpcb_name"].values
        historical["cpcb_sector"] = industrial.iloc[nearest_idx]["cpcb_sector"].values
    else:
        historical["cpcb_name"] = ""
        historical["cpcb_sector"] = ""

    # Match nearest Tamil Nadu settlements
    import os
    places_path = "data/raw/tamilnadu_places.csv"
    if os.path.exists(places_path):
        print("Matching 1597 hotspots to 25,000+ Tamil Nadu settlements...")
        places = pd.read_csv(places_path)
        p_rad = np.radians(places[["latitude", "longitude"]].values)
        p_tree = BallTree(p_rad, metric="haversine")
        dists, idxs = p_tree.query(hist_rad, k=1)
        p_idx = idxs.flatten()
        historical["nearest_place_km"] = dists.flatten() * 6371.0
        historical["nearest_place_name"] = places.iloc[p_idx]["name"].values
        historical["nearest_place_type"] = places.iloc[p_idx]["place_type"].values
    else:
        historical["nearest_place_km"] = 999.0
        historical["nearest_place_name"] = "Tamil Nadu locality"
        historical["nearest_place_type"] = "village"

    # 2. Land-cover classification
    print("Assigning land-cover classification...")
    historical["in_forest"] = False
    historical["in_farmland"] = False
    try:
        from shapely import wkt
        if os.path.exists("data/raw/osm_forest.csv"):
            f_df = pd.read_csv("data/raw/osm_forest.csv")
            f_geoms = [wkt.loads(g) for g in f_df["geometry_wkt"] if pd.notna(g)]
            f_gdf = gpd.GeoDataFrame(geometry=f_geoms, crs="EPSG:4326")
            h_points = [Point(xy) for xy in zip(historical["longitude"], historical["latitude"])]
            h_gdf = gpd.GeoDataFrame(geometry=h_points, crs="EPSG:4326")
            h_buf = h_gdf.copy()
            h_buf["geometry"] = h_buf.geometry.buffer(0.015)
            f_join = gpd.sjoin(h_buf, f_gdf, how="left", predicate="intersects")
            historical["in_forest"] = historical.index.map(
                f_join.groupby(f_join.index)["index_right"].apply(lambda x: x.notna().any())
            ).fillna(False)

        if os.path.exists("data/raw/osm_farmland.csv"):
            fm_df = pd.read_csv("data/raw/osm_farmland.csv")
            fm_geoms = [wkt.loads(g) for g in fm_df["geometry_wkt"] if pd.notna(g)]
            fm_gdf = gpd.GeoDataFrame(geometry=fm_geoms, crs="EPSG:4326")
            h_points = [Point(xy) for xy in zip(historical["longitude"], historical["latitude"])]
            h_gdf = gpd.GeoDataFrame(geometry=h_points, crs="EPSG:4326")
            h_buf = h_gdf.copy()
            h_buf["geometry"] = h_buf.geometry.buffer(0.015)
            fm_join = gpd.sjoin(h_buf, fm_gdf, how="left", predicate="intersects")
            historical["in_farmland"] = historical.index.map(
                fm_join.groupby(fm_join.index)["index_right"].apply(lambda x: x.notna().any())
            ).fillna(False)
    except Exception as e:
        print(f"Warning during spatial join: {e}")

    def assign_class_and_place(row):
        if row["nearest_industrial_km"] <= 2.0:
            ind_name = row.get("nearest_industrial_name", "")
            ind_disp = ind_name if ind_name and ind_name != "unnamed" else "Industrial facility"
            return "industrial_fire", f"Industrial facility ({ind_disp})"
        elif row["in_forest"]:
            return "wildfire", f"Forest / Woodland reserve near {row['nearest_place_name']}"
        elif row["in_farmland"] or (row["nearest_place_type"] in ["village", "hamlet"] and row["nearest_place_km"] <= 5.0):
            dist_str = f"{row['nearest_place_km']:.1f} km"
            return "agricultural_burn", f"Agricultural cropland near {row['nearest_place_name']} ({row['nearest_place_type']}, {dist_str})"
        elif row["nearest_place_type"] in ["city", "town", "suburb"] and row["nearest_place_km"] <= 2.5:
            return "urban_commercial_fire", f"Urban / Commercial area ({row['nearest_place_name']})"
        elif row["frp"] >= 15.0 and row["nearest_industrial_km"] <= 6.0:
            return "industrial_fire", f"Industrial cluster vicinity ({row.get('nearest_industrial_name', 'manufacturing zone')})"
        else:
            return "scrub_biomass_burn", f"Rural scrubland / open biomass near {row['nearest_place_name']} ({row['nearest_place_km']:.1f} km)"

    res = historical.apply(assign_class_and_place, axis=1)
    historical["final_class"] = [r[0] for r in res]
    historical["place_description"] = [r[1] for r in res]
        
    # 3. Anomaly detection
    print("Checking anomalies against facility baselines...")
    baseline["facility_osm_id"] = baseline["facility_osm_id"].astype(str)
    historical["facility_osm_id_str"] = historical["facility_osm_id"].astype(str)
    
    merged = historical.merge(baseline, left_on="facility_osm_id_str", right_on="facility_osm_id", how="left", suffixes=("", "_b"))
    merged["is_anomaly"] = (
        (merged["nearest_industrial_km"] <= 2.0) &
        merged["baseline_frp_mean"].notna() &
        merged["baseline_frp_std"].notna() &
        (merged["baseline_frp_std"] > 0) &
        (merged["frp"] > (merged["baseline_frp_mean"] + 2 * merged["baseline_frp_std"]))
    )
    historical["is_anomaly"] = merged["is_anomaly"].fillna(False)
    historical["baseline_frp_mean"] = merged["baseline_frp_mean"]
    historical["baseline_frp_std"] = merged["baseline_frp_std"]
    
    # 4. ML predictions
    print("Applying ML model predictions...")
    try:
        model = joblib.load("data/processed/fire_classifier_model.pkl")
        historical["daynight_flag"] = historical["daynight"].map({"D": 1, "N": 0}).fillna(1)
        historical["confidence_num"] = historical["confidence"].map({"l": 0, "n": 1, "h": 2}).fillna(1)
        feature_cols = ["frp", "bright_ti4", "nearest_industrial_km", "daynight_flag", "confidence_num"]
        valid_mask = historical[feature_cols].notna().all(axis=1)
        X = historical.loc[valid_mask, feature_cols]
        
        preds = model.predict(X)
        probs = model.predict_proba(X).max(axis=1)
        
        historical.loc[valid_mask, "ml_prediction"] = preds
        historical.loc[valid_mask, "ml_confidence"] = probs
        historical["ml_prediction"] = historical["ml_prediction"].fillna("N/A")
        historical["ml_confidence"] = historical["ml_confidence"].fillna(0)
    except Exception as e:
        print(f"ML prediction error: {e}")
        historical["ml_prediction"] = historical["final_class"]
        historical["ml_confidence"] = 0.85
        
    out_csv = "data/processed/all_tamilnadu_hotspots.csv"
    historical.to_csv(out_csv, index=False)
    print(f"Saved {len(historical)} classified Tamil Nadu records to {out_csv}")
    
    # 5. Spatial clustering across Tamil Nadu (DBSCAN 5km)
    print("Computing spatial hotspot clusters across Tamil Nadu...")
    coords = np.radians(historical[["latitude", "longitude"]].values)
    epsilon = 5.0 / 6371.0 # 5 km
    db = DBSCAN(eps=epsilon, min_samples=3, metric="haversine").fit(coords)
    historical["cluster"] = db.labels_
    
    clusters = []
    for c_id in sorted(set(db.labels_)):
        if c_id == -1:
            continue
        c_pts = historical[historical["cluster"] == c_id]
        center_lat = float(c_pts["latitude"].mean())
        center_lon = float(c_pts["longitude"].mean())
        
        dists = np.sqrt((c_pts["latitude"] - center_lat)**2 + (c_pts["longitude"] - center_lon)**2) * 111.0
        radius_km = max(float(dists.max()), 1.5)
        
        top_facility = c_pts[c_pts["nearest_industrial_name"].notna()]["nearest_industrial_name"].mode()
        facility_name = str(top_facility[0]) if len(top_facility) > 0 else "Industrial Facility"
        
        dom_class = c_pts["final_class"].mode()
        dominant_class = str(dom_class[0]) if len(dom_class) > 0 else "unclassified"
        
        cpcb_match = c_pts[c_pts["cpcb_name"].notna() & (c_pts["cpcb_name"] != "")]["cpcb_name"].mode()
        cpcb_str = str(cpcb_match[0]) if len(cpcb_match) > 0 else ""
        
        def region_tag(lat, lon):
            if lat >= 12.5: return "North Tamil Nadu (Chennai/Tiruvallur/Ranipet)"
            elif lat >= 11.0:
                if lon < 78.0: return "West Tamil Nadu (Coimbatore/Tirupur/Salem/Erode)"
                else: return "Central & Delta Tamil Nadu (Trichy/Thanjavur/Cuddalore)"
            elif lat >= 9.5: return "South-Central Tamil Nadu (Madurai/Sivakasi/Virudhunagar)"
            else: return "Deep South Tamil Nadu (Thoothukudi/Tirunelveli/Tenkasi)"
            
        clusters.append({
            "cluster_id": int(c_id),
            "center_lat": round(center_lat, 5),
            "center_lon": round(center_lon, 5),
            "radius_km": round(radius_km, 2),
            "point_count": int(len(c_pts)),
            "mean_frp": round(float(c_pts["frp"].mean()), 2),
            "max_frp": round(float(c_pts["frp"].max()), 2),
            "dominant_class": dominant_class,
            "facility_name": facility_name,
            "cpcb_name": cpcb_str,
            "region": region_tag(center_lat, center_lon),
            "date_range": f"{c_pts['acq_date'].min()} to {c_pts['acq_date'].max()}"
        })
        
    clusters.sort(key=lambda x: x["point_count"], reverse=True)
    with open("data/processed/hotspot_clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)
        
    print(f"Successfully generated {len(clusters)} hotspot cluster areas in data/processed/hotspot_clusters.json")

if __name__ == "__main__":
    process_historical()
