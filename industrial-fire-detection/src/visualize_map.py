import pandas as pd
import folium


def create_map():

    df = pd.read_csv("data/processed/fused_hotspots.csv")

    # =========================================================
    # CENTER MAP ON INDIA
    # =========================================================
    m = folium.Map(
        location=[22.0, 79.0],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    # =========================================================
    # SATELLITE LAYER
    # =========================================================
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # =========================================================
    # HOTSPOTS
    # =========================================================
    for _, row in df.iterrows():

        color = (
            "red"
            if row["preliminary_class"] == "likely_industrial"
            else "green"
        )

        # -----------------------------------------------------
        # INDUSTRY INFORMATION
        # -----------------------------------------------------
        industry_name = str(
            row.get("nearest_industrial_name", "Unknown")
        )

        industry_lat = row.get(
            "nearest_industrial_latitude",
            None
        )

        industry_lon = row.get(
            "nearest_industrial_longitude",
            None
        )

        # Convert coordinates safely
        try:
            industry_lat = float(industry_lat)
            industry_lon = float(industry_lon)
        except (ValueError, TypeError):
            industry_lat = None
            industry_lon = None

        # -----------------------------------------------------
        # CREATE INDUSTRY MARKER FIRST
        # -----------------------------------------------------
        industry_marker = None

        if (
            industry_lat is not None
            and industry_lon is not None
        ):

            industry_popup = folium.Popup(
                f"""
                <b>🏭 Industry</b><br><br>
                <b>Name:</b> {industry_name}<br>
                <b>Type:</b> {row['nearest_industrial_type']}<br>
                <b>Distance:</b> {row['nearest_industrial_km']:.2f} km
                """,
                max_width=350
            )

            industry_marker = folium.Marker(
                location=[
                    industry_lat,
                    industry_lon
                ],
                popup=industry_popup,
                tooltip=industry_name,
                icon=folium.Icon(
                    icon="industry",
                    prefix="fa",
                    color="blue"
                )
            )

            industry_marker.add_to(m)

        # -----------------------------------------------------
        # HOTSPOT POPUP
        # -----------------------------------------------------
        popup_html = f"""
        <div style="width:250px">

            <h4>🔥 Thermal Hotspot</h4>

            <b>FRP:</b> {row['frp']}<br>

            <b>Nearest Industry:</b><br>
            {industry_name}<br><br>

            <b>Distance:</b>
            {row['nearest_industrial_km']:.2f} km<br>

            <b>Type:</b>
            {row['nearest_industrial_type']}<br>

            <b>Class:</b>
            {row['preliminary_class']}<br>
        """

        # -----------------------------------------------------
        # ADD INDUSTRY VIEW BUTTON
        # -----------------------------------------------------
        if (
            industry_lat is not None
            and industry_lon is not None
        ):

            map_name = m.get_name()
            industry_marker_name = industry_marker.get_name()

            popup_html += f"""
            <br>

            <button
                onclick="
                    {map_name}.setView(
                        [{industry_lat}, {industry_lon}],
                        18
                    );

                    {industry_marker_name}.openPopup();
                "
                style="
                    background:#1976d2;
                    color:white;
                    border:none;
                    padding:8px 12px;
                    border-radius:6px;
                    cursor:pointer;
                    font-weight:bold;
                "
            >
                🏭 View Industry
            </button>
            """

        popup_html += "</div>"

        # -----------------------------------------------------
        # HOTSPOT MARKER
        # -----------------------------------------------------
        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                popup_html,
                max_width=350
            ),
            tooltip="🔥 Hotspot"
        ).add_to(m)

    # =========================================================
    # LAYER CONTROL
    # =========================================================
    folium.LayerControl().add_to(m)

    # =========================================================
    # SAVE
    # =========================================================
    output_path = "data/processed/hotspot_map.html"

    m.save(output_path)

    print(f"Map saved to: {output_path}")
    print("Open this file in your browser to view it.")


if __name__ == "__main__":
    create_map()