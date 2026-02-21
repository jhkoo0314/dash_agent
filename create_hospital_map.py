import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 1. Load Data
file_path = r"c:\agent_b\data\public\1.병원정보서비스(2025.12.).xlsx"
df = pd.read_excel(file_path)

# 2. Filter for General Hospitals and Tertiary General Hospitals
# '종합병원' and '상급종합'
target_types = ['종합병원', '상급종합']
df_filtered = df[df['종별코드명'].isin(target_types)].copy()

# 3. Clean mapping data (Remove missing coordinates)
df_filtered = df_filtered.dropna(subset=['좌표(X)', '좌표(Y)'])

# 4. Initialize Map (Center of Korea)
m = folium.Map(location=[36.5, 127.5], zoom_start=7)

# 5. Add Markers
marker_cluster = MarkerCluster().add_to(m)

for idx, row in df_filtered.iterrows():
    name = row['요양기관명']
    h_type = row['종별코드명']
    addr = row['주소']
    lat = row['좌표(Y)']
    lng = row['좌표(X)']
    
    # Color coding
    color = 'red' if h_type == '상급종합' else 'blue'
    
    popup_text = f"<b>{name}</b><br>구분: {h_type}<br>주소: {addr}"
    
    folium.Marker(
        location=[lat, lng],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=name,
        icon=folium.Icon(color=color, icon='plus', prefix='fa')
    ).add_to(marker_cluster)

# 6. Save Map
save_path = "hospital_map.html"
m.save(save_path)
print(f"Map saved to {save_path}")
print(f"Total hospitals filtered: {len(df_filtered)}")
