import pandas as pd
import folium

def create_map():
    df = pd.read_csv("data/processed/fused_hotspots.csv")

    # Center map on India
    m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="OpenStreetMap")

    for _, row in df.iterrows():
        color = "red" if row["preliminary_class"] == "likely_industrial" else "green"
        popup_text = (
            f"FRP: {row['frp']}<br>"
            f"Distance to industrial: {row['nearest_industrial_km']:.2f} km<br>"
            f"Type: {row['nearest_industrial_type']}<br>"
            f"Class: {row['preliminary_class']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    output_path = "data/processed/hotspot_map.html"
    m.save(output_path)
    print(f"Map saved to: {output_path}")
    print("Open this file in your browser to view it.")

if __name__ == "__main__":
    create_map()