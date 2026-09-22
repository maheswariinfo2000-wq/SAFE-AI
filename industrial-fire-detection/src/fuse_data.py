import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# Distance threshold to consider a hotspot "near" an industrial zone (in km)
DISTANCE_THRESHOLD_KM = 2.0


def haversine_distance_matrix(hotspots_df, industrial_df):
    """
    Uses BallTree with haversine metric to find, for each hotspot,
    the nearest industrial zone and the distance to it (in km).
    """

    # Convert degrees to radians (required for haversine metric)
    industrial_rad = np.radians(
        industrial_df[["latitude", "longitude"]].values
    )

    hotspot_rad = np.radians(
        hotspots_df[["latitude", "longitude"]].values
    )

    # Build a spatial index over industrial zones
    tree = BallTree(industrial_rad, metric="haversine")

    # Query nearest industrial zone for every hotspot
    distances, indices = tree.query(hotspot_rad, k=1)

    # Convert distance from radians to km (Earth radius ~6371 km)
    EARTH_RADIUS_KM = 6371.0
    distances_km = distances.flatten() * EARTH_RADIUS_KM
    nearest_indices = indices.flatten()

    return distances_km, nearest_indices


def fuse_data():
    print("Loading datasets...")

    hotspots = pd.read_csv(
        "data/raw/firms_hotspots.csv"
    )

    industrial = pd.read_csv(
        "data/raw/osm_industrial.csv"
    )

    print(
        f"Hotspots: {len(hotspots)} | "
        f"Industrial zones: {len(industrial)}"
    )

    print("Computing nearest industrial zone for each hotspot...")

    distances_km, nearest_indices = haversine_distance_matrix(
        hotspots,
        industrial
    )

    # =========================================================
    # ATTACH FUSION RESULTS TO HOTSPOTS DATAFRAME
    # =========================================================

    hotspots["nearest_industrial_km"] = distances_km

    hotspots["nearest_industrial_type"] = (
        industrial.iloc[nearest_indices]["type"].values
    )

    hotspots["nearest_industrial_name"] = (
        industrial.iloc[nearest_indices]["name"].values
    )

    hotspots["nearest_industrial_region"] = (
        industrial.iloc[nearest_indices]["region"].values
    )

    # =========================================================
    # EXTRA: NEAREST INDUSTRY COORDINATES
    # =========================================================

    hotspots["nearest_industrial_latitude"] = (
        industrial.iloc[nearest_indices]["latitude"].values
    )

    hotspots["nearest_industrial_longitude"] = (
        industrial.iloc[nearest_indices]["longitude"].values
    )

    # =========================================================
    # PRELIMINARY CLASSIFICATION BASED ON PROXIMITY
    # =========================================================

    hotspots["preliminary_class"] = np.where(
        hotspots["nearest_industrial_km"] <= DISTANCE_THRESHOLD_KM,
        "likely_industrial",
        "likely_natural"
    )

    # =========================================================
    # SAVE FUSED DATASET
    # =========================================================

    output_path = "data/processed/fused_hotspots.csv"

    hotspots.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved fused dataset to: {output_path}")

    # =========================================================
    # CLASS DISTRIBUTION
    # =========================================================

    print("\nClass distribution:")

    print(
        hotspots["preliminary_class"].value_counts()
    )

    # =========================================================
    # SAMPLE ROWS
    # =========================================================

    print("\nSample rows:")

    print(
        hotspots[
            [
                "latitude",
                "longitude",
                "frp",
                "nearest_industrial_km",
                "nearest_industrial_name",
                "nearest_industrial_type",
                "nearest_industrial_latitude",
                "nearest_industrial_longitude",
                "preliminary_class"
            ]
        ].head(10)
    )

    return hotspots


if __name__ == "__main__":
    fuse_data()