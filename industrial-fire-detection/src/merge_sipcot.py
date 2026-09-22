import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np

SIPCOT_ESTATES = [
    {"name": "SIPCOT Ranipet",        "lat": 12.9247, "lon": 79.3308, "sector": "Leather/Chemical"},
    {"name": "SIPCOT Hosur",          "lat": 12.7409, "lon": 77.8253, "sector": "Auto/Engineering"},
    {"name": "SIPCOT Cuddalore",      "lat": 11.7480, "lon": 79.7714, "sector": "Petrochemical"},
    {"name": "SIPCOT Gummidipoondi",  "lat": 13.3990, "lon": 80.1020, "sector": "General"},
    {"name": "SIPCOT Thoothukudi",    "lat": 8.7642,  "lon": 78.1348, "sector": "Chemical/Port"},
    {"name": "SIPCOT Perundurai",     "lat": 11.2762, "lon": 77.5850, "sector": "Textile/General"},
    {"name": "SIPCOT Sriperumbudur",  "lat": 12.9675, "lon": 79.9430, "sector": "Auto/Electronics"},
    {"name": "SIPCOT Siruseri",       "lat": 12.8231, "lon": 80.2210, "sector": "IT"},
    {"name": "SIPCOT Manamadurai",    "lat": 9.6975,  "lon": 78.4761, "sector": "General"},
    {"name": "SIPCOT Pudukkottai",    "lat": 10.3813, "lon": 78.8214, "sector": "General"},
    {"name": "SIPCOT Cheyyar",        "lat": 12.6667, "lon": 79.5333, "sector": "General"},
    {"name": "SIPCOT Irungattukottai","lat": 12.9820, "lon": 79.9930, "sector": "Auto"},
]

def mark_sipcot_verified():
    industrial = pd.read_csv("data/raw/osm_industrial.csv")
    sipcot_df = pd.DataFrame(SIPCOT_ESTATES)

    sipcot_rad = np.radians(sipcot_df[["lat", "lon"]].values)
    industrial_rad = np.radians(industrial[["latitude", "longitude"]].values)

    tree = BallTree(sipcot_rad, metric="haversine")
    distances, indices = tree.query(industrial_rad, k=1)
    distances_km = distances.flatten() * 6371.0

    industrial["sipcot_verified"] = distances_km <= 5.0  # within 5km of a known SIPCOT estate
    industrial["sipcot_sector"] = ""
    matched = distances_km <= 5.0
    industrial.loc[matched, "sipcot_sector"] = sipcot_df.iloc[indices.flatten()[matched]]["sector"].values

    industrial.to_csv("data/raw/osm_industrial.csv", index=False)
    print(f"Marked {matched.sum()} facilities as SIPCOT-verified")

if __name__ == "__main__":
    mark_sipcot_verified()