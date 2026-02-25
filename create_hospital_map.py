from datetime import datetime
import os

import folium
from folium.plugins import MarkerCluster
import pandas as pd


def get_spatial_preview_path(output_dir: str = "output") -> str:
    """Return unique output path like output/Spatial_Preview_YYMMDD_*.html."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    base = os.path.join(output_dir, f"Spatial_Preview_{date_str}_{time_str}")
    path = f"{base}.html"
    counter = 1
    while os.path.exists(path):
        path = f"{base}_{counter}.html"
        counter += 1
    return path


def main() -> None:
    file_path = r"c:\agent_b\data\public\1.병원정보서비스(2025.12.).xlsx"
    df = pd.read_excel(file_path)

    target_types = ["종합병원", "상급종합"]
    df_filtered = df[df["종별코드명"].isin(target_types)].copy()
    df_filtered = df_filtered.dropna(subset=["좌표(X)", "좌표(Y)"])

    m = folium.Map(location=[36.5, 127.5], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_filtered.iterrows():
        name = row["요양기관명"]
        h_type = row["종별코드명"]
        addr = row["주소"]
        lat = row["좌표(Y)"]
        lng = row["좌표(X)"]

        color = "red" if h_type == "상급종합" else "blue"
        popup_text = f"<b>{name}</b><br>구분: {h_type}<br>주소: {addr}"

        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=name,
            icon=folium.Icon(color=color, icon="plus", prefix="fa"),
        ).add_to(marker_cluster)

    output_path = get_spatial_preview_path("output")
    m.save(output_path)

    # Keep legacy base map for current sandbox map tab compatibility.
    legacy_path = "hospital_map.html"
    m.save(legacy_path)

    print(f"Map saved to {output_path}")
    print(f"Legacy map updated: {legacy_path}")
    print(f"Total hospitals filtered: {len(df_filtered)}")


if __name__ == "__main__":
    main()
