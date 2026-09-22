import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np

# Source: CPCB "17 Category of Highly Polluting Industries" - Tamil Nadu (27.07.2018)
# Coordinates are town/village-center approximations (facility-exact lat/lon not published)
KNOWN_POLLUTING_INDUSTRIES = [
    {"name": "Madras Aluminium Company Ltd (MALCO)", "lat": 11.7880, "lon": 77.8004, "sector": "Aluminium", "district": "Salem"},
    {"name": "Tamil Nadu Cement Corp - Ariyalur", "lat": 11.1401, "lon": 79.0782, "sector": "Cement", "district": "Ariyalur"},
    {"name": "Jeppiar Cements Pvt Ltd", "lat": 11.2342, "lon": 78.8807, "sector": "Cement", "district": "Perambalur"},
    {"name": "Sterlite Industries (India) Ltd - Copper Smelter", "lat": 8.7642, "lon": 78.1348, "sector": "Copper", "district": "Thoothukudi"},
    {"name": "Amaravathy Co-Op Sugar Mills (Distillery)", "lat": 10.5876, "lon": 77.2495, "sector": "Distillery", "district": "Tiruppur"},
    {"name": "Salem Co-Op Sugar Mills (Distillery)", "lat": 11.2189, "lon": 78.1674, "sector": "Distillery", "district": "Namakkal"},
    {"name": "Chemplast Sanmar Limited", "lat": 11.7739, "lon": 79.5540, "sector": "Distillery/Chemical", "district": "Cuddalore"},
    {"name": "Mohan Breweries and Distilleries Ltd", "lat": 12.6819, "lon": 79.9864, "sector": "Distillery", "district": "Kancheepuram"},
    {"name": "S.V. Sugars Ltd (Distillery)", "lat": 12.8342, "lon": 79.7036, "sector": "Distillery", "district": "Kancheepuram"},
    {"name": "Trichy Distilleries & Chemicals Ltd", "lat": 10.7905, "lon": 78.7047, "sector": "Distillery", "district": "Trichy"},
    {"name": "Bhavani Distilleries and Chemicals Ltd", "lat": 12.9051, "lon": 79.3308, "sector": "Distillery", "district": "Vellore"},
    {"name": "Southern Agrifurane Industries Ltd", "lat": 11.9401, "lon": 79.4861, "sector": "Distillery", "district": "Villupuram"},
    {"name": "Sanmar Speciality Chemicals Ltd", "lat": 12.7409, "lon": 77.8253, "sector": "Pharma", "district": "Dharmapuri"},
    {"name": "IND Barath Thermal Power Ltd", "lat": 8.8894, "lon": 77.9350, "sector": "Powerplant", "district": "Thoothukudi"},
    {"name": "Thirumakotai Gas Turbine Power Project", "lat": 10.6664, "lon": 79.4514, "sector": "Powerplant", "district": "Thiruvarur"},
    {"name": "Southern Energy Development Corp Ltd", "lat": 10.6664, "lon": 79.4514, "sector": "Powerplant", "district": "Thiruvarur"},
    {"name": "Shriram Power Gen Ltd (Biomass)", "lat": 10.3673, "lon": 77.9803, "sector": "Powerplant", "district": "Dindigul"},
    {"name": "Shriram Non-Conventional Energy Ltd", "lat": 10.4256, "lon": 79.3175, "sector": "Powerplant", "district": "Thanjavur"},
    {"name": "Sahali Exports Pvt Ltd (Biomass)", "lat": 11.1017, "lon": 79.6552, "sector": "Powerplant", "district": "Nagapattinam"},
    {"name": "Pioneer Power Ltd", "lat": 9.3639, "lon": 78.8395, "sector": "Powerplant", "district": "Ramanathapuram"},
    {"name": "T.C.P. Limited - Gummidipoondi", "lat": 13.3990, "lon": 80.1020, "sector": "Powerplant", "district": "Tiruvallur"},
    {"name": "Synergy Shakthi Renewable Energy Ltd", "lat": 12.5266, "lon": 78.2150, "sector": "Powerplant", "district": "Krishnagiri"},
    {"name": "North Chennai Thermal Power Station", "lat": 13.3357, "lon": 80.1955, "sector": "Powerplant", "district": "Tiruvallur"},
    {"name": "Global Powertech Equipments Ltd (Biomass)", "lat": 12.5085, "lon": 79.6103, "sector": "Powerplant", "district": "Tiruvannamalai"},
    {"name": "Coromandel Electric Company Ltd", "lat": 9.3639, "lon": 78.8395, "sector": "Powerplant", "district": "Ramanathapuram"},
    {"name": "Coastal Energen Power Ltd", "lat": 8.7642, "lon": 78.1348, "sector": "Powerplant", "district": "Thoothukudi"},
    {"name": "Aurobindo Agro Energy Pvt Ltd", "lat": 9.8434, "lon": 78.4809, "sector": "Powerplant", "district": "Sivagangai"},
    {"name": "Arkay Energy (Rameswaram) Ltd", "lat": 9.2876, "lon": 79.3129, "sector": "Powerplant", "district": "Ramanathapuram"},
    {"name": "Sun Paper Mills Limited", "lat": 8.6822, "lon": 77.4747, "sector": "Pulp & Paper", "district": "Tirunelveli"},
    {"name": "EID Parry - Karur", "lat": 10.9601, "lon": 78.0766, "sector": "Sugar", "district": "Karur"},
    {"name": "Dharmapuri District Co-op Sugar Mills", "lat": 12.1211, "lon": 78.1583, "sector": "Sugar", "district": "Dharmapuri"},
    {"name": "N.P.K.R. Ramasamy Co-op Sugar Mills", "lat": 11.1017, "lon": 79.6552, "sector": "Sugar", "district": "Nagapattinam"},
    {"name": "Cheyyar Co-Op Sugar Mills Ltd", "lat": 12.6667, "lon": 79.5333, "sector": "Sugar", "district": "Tiruvannamalai"},
    {"name": "Tirupattur Co-operative Sugar Mills", "lat": 12.9165, "lon": 79.1325, "sector": "Sugar", "district": "Vellore"},
    {"name": "Vellore Co-Op Sugar Mills", "lat": 12.9165, "lon": 79.1325, "sector": "Sugar", "district": "Vellore"},
    {"name": "Chengalrayan Co-operative Sugar Mills", "lat": 11.6167, "lon": 79.3167, "sector": "Sugar", "district": "Villupuram"},
    {"name": "EID Parry India Ltd - Sivagangai", "lat": 9.8434, "lon": 78.4809, "sector": "Sugar", "district": "Sivagangai"},
    {"name": "EID Parry (I) Ltd - Trichy", "lat": 10.7905, "lon": 78.7047, "sector": "Sugar", "district": "Trichy"},
    {"name": "Empee Sugars and Chemicals Ltd", "lat": 8.7089, "lon": 77.4442, "sector": "Sugar", "district": "Tirunelveli"},
    {"name": "Eastern Chrome Tanning Corporation", "lat": 12.6813, "lon": 78.6197, "sector": "Tannery", "district": "Vellore"},
    {"name": "T. Abdul Wahid & Co - 'A' Tannery", "lat": 12.6813, "lon": 78.6197, "sector": "Tannery", "district": "Vellore"},
    {"name": "T. Abdul Wahid Tanneries - 'C' Tannery", "lat": 12.6813, "lon": 78.6197, "sector": "Tannery", "district": "Vellore"},
    {"name": "Bachi Shoes India Pvt Ltd (Ambur)", "lat": 12.7917, "lon": 78.7161, "sector": "Tannery", "district": "Vellore"},
]


def merge_polluting_industries():
    industrial = pd.read_csv("data/raw/osm_industrial.csv")
    known_df = pd.DataFrame(KNOWN_POLLUTING_INDUSTRIES)

    known_rad = np.radians(known_df[["lat", "lon"]].values)
    industrial_rad = np.radians(industrial[["latitude", "longitude"]].values)

    tree = BallTree(known_rad, metric="haversine")
    distances, indices = tree.query(industrial_rad, k=1)
    distances_km = distances.flatten() * 6371.0

    matched = distances_km <= 5.0  # within 5km of a known CPCB-listed facility
    industrial["cpcb_verified"] = matched
    industrial["cpcb_name"] = ""
    industrial["cpcb_sector"] = ""
    industrial.loc[matched, "cpcb_name"] = known_df.iloc[indices.flatten()[matched]]["name"].values
    industrial.loc[matched, "cpcb_sector"] = known_df.iloc[indices.flatten()[matched]]["sector"].values

    # Also add any known facilities NOT already near an OSM point, as new rows
    industrial_rad2 = np.radians(industrial[["latitude", "longitude"]].values)
    tree2 = BallTree(industrial_rad2, metric="haversine")
    d2, i2 = tree2.query(known_rad, k=1)
    d2_km = d2.flatten() * 6371.0
    unmatched_known = known_df[d2_km > 5.0]

    if len(unmatched_known) > 0:
        new_rows = pd.DataFrame({
            "osm_id": [f"cpcb_{i}" for i in unmatched_known.index],
            "region": "cpcb_verified",
            "type": "industrial",
            "name": unmatched_known["name"].values,
            "latitude": unmatched_known["lat"].values,
            "longitude": unmatched_known["lon"].values,
            "cpcb_verified": True,
            "cpcb_name": unmatched_known["name"].values,
            "cpcb_sector": unmatched_known["sector"].values,
        })
        industrial = pd.concat([industrial, new_rows], ignore_index=True)

    industrial.to_csv("data/raw/osm_industrial.csv", index=False)
    print(f"Marked {matched.sum()} existing facilities as CPCB-verified")
    print(f"Added {len(unmatched_known)} new CPCB facilities not previously in database")
    print(f"Total industrial facilities now: {len(industrial)}")


if __name__ == "__main__":
    merge_polluting_industries()