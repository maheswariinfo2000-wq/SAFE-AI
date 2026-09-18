import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

DISTANCE_THRESHOLD_KM = 2.0
ANOMALY_STD_MULTIPLIER = 2.0   # flag if FRP > mean + 2*std

def link_to_nearest_facility(points_df, industrial_df):
    industrial_rad = np.radians(industrial_df[["latitude", "longitude"]].values)
    points_rad = np.radians(points_df[["latitude", "longitude"]].values)

    tree = BallTree(industrial_rad, metric="haversine")
    distances, indices = tree.query(points_rad, k=1)

    EARTH_RADIUS_KM = 6371.0
    distances_km = distances.flatten() * EARTH_RADIUS_KM
    nearest_indices = indices.flatten()

    return distances_km, nearest_indices

def build_baselines():
    print("Loading historical hotspots and industrial zones...")
    historical = pd.read_csv("data/raw/firms_historical.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")

    print(f"Historical records: {len(historical)} | Industrial zones: {len(industrial)}")

    print("Linking historical hotspots to nearest facility...")
    distances_km, nearest_indices = link_to_nearest_facility(historical, industrial)

    historical["facility_osm_id"] = industrial.iloc[nearest_indices]["osm_id"].values
    historical["facility_distance_km"] = distances_km
    historical["facility_name"] = industrial.iloc[nearest_indices]["name"].values

    # Only keep historical points that are actually near a facility (within threshold)
    near_facility = historical[historical["facility_distance_km"] <= DISTANCE_THRESHOLD_KM].copy()
    print(f"Historical points near a facility: {len(near_facility)}")

    # Build baseline stats per facility
    baseline = near_facility.groupby("facility_osm_id")["frp"].agg(["mean", "std", "count"]).reset_index()
    baseline.columns = ["facility_osm_id", "baseline_frp_mean", "baseline_frp_std", "baseline_sample_count"]

    # Facilities with only 1 sample have std = NaN, fill with a safe default
    baseline["baseline_frp_std"] = baseline["baseline_frp_std"].fillna(baseline["baseline_frp_mean"] * 0.5)

    baseline.to_csv("data/processed/facility_baselines.csv", index=False)
    print(f"\nSaved {len(baseline)} facility baselines to data/processed/facility_baselines.csv")
    print(baseline.head(10))

    return baseline

def detect_anomalies():
    print("\nLoading current (live) classified hotspots...")
    current = pd.read_csv("data/processed/final_classified_hotspots.csv")
    baseline = pd.read_csv("data/processed/facility_baselines.csv")
    industrial = pd.read_csv("data/raw/osm_industrial.csv")

    # Link current hotspots to nearest facility (reuse same logic)
    distances_km, nearest_indices = link_to_nearest_facility(current, industrial)
    current["facility_osm_id"] = industrial.iloc[nearest_indices]["osm_id"].values
    current["facility_distance_km"] = distances_km

    # Merge baseline stats
    merged = current.merge(baseline, on="facility_osm_id", how="left")

    def check_anomaly(row):
        if row["final_class"] != "industrial_fire":
            return False
        if pd.isna(row["baseline_frp_mean"]):
            return False  # no baseline history for this facility
        threshold = row["baseline_frp_mean"] + ANOMALY_STD_MULTIPLIER * row["baseline_frp_std"]
        return row["frp"] > threshold

    merged["is_anomaly"] = merged.apply(check_anomaly, axis=1)

    output_path = "data/processed/hotspots_with_anomalies.csv"
    merged.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"\nTotal anomalies detected: {merged['is_anomaly'].sum()}")
    print(merged[merged["is_anomaly"]][[
        "latitude", "longitude", "frp", "baseline_frp_mean", "facility_distance_km"
    ]])

    return merged

if __name__ == "__main__":
    build_baselines()
    detect_anomalies()