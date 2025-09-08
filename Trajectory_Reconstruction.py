import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import movingpandas as mpd
from shapely.geometry import Point
import geopandas as gpd
import os

file_path = r"C:\Users\slab\Desktop\Stage1\data\Device_AB00006.csv"
try:
    df
except NameError:
    df = pd.read_csv(file_path, low_memory=False)
df = pd.read_csv(file_path, low_memory=False)
df.dtypes
len(df)

# 我只想畫出416464000這艘船的航線圖，先把其他的洗掉
mmsi_target = 416464000
df_filtered = df[df['MMSI'] == mmsi_target].copy()
print(len(df_filtered))
print(df_filtered.head())

#大概看一下有效資料占比
print("有效data數量約為:", len(df_filtered)/len(df))

# found that some error value has been recorded
df_filtered = df_filtered[
    (df_filtered['Lat'] >= -90) & (df_filtered['Lat'] <= 90) &
    (df_filtered['Long'] >= -180) & (df_filtered['Long'] <= 180)
].copy()

# 再檢查經緯度是否都合理
print(df_filtered[['Lat', 'Long']].describe())
print("有效data數量約為:", len(df_filtered)/len(df))

# check out columns
print(df_filtered.columns)

# Modify the longitude and latitude 欄位
df_filtered['Lat'] = df_filtered['Lat'].astype(float)
df_filtered['Long'] = df_filtered['Long'].astype(float)

# Timestamp to datetime
df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'], errors='coerce')

# dropna
df_filtered = df_filtered.dropna(subset=['Lat', 'Long', 'Timestamp'])
print(df_filtered.dtypes)

# sort by the datetime
df_filtered = df_filtered.sort_values('Timestamp').reset_index(drop=True)

# build up GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_filtered,
    geometry=gpd.points_from_xy(df_filtered['Long'], df_filtered['Lat']),
    crs="EPSG:4326"
)

print("df_filtered columns:", df_filtered.columns.tolist())
print("gdf columns:", gdf.columns.tolist())

# Generate MovingPandas trajectory
traj = mpd.Trajectory(gdf, traj_id=str(df_filtered['MMSI'].iloc[0]), t='Timestamp')
print(traj.df.index)
print(traj)
print(type(traj))
print("軌跡點數：", len(traj.df))
print(traj.df.columns)

# 取軌跡的 GeoDataFrame
gdf = traj.df

# 處理跨越國際換日線的函數 - 經度連續化
def make_longitude_continuous(df):
    """讓跨越國際換日線的經度連續化"""
    df = df.copy()
    adjusted_coords = []
    
    if len(df) == 0:
        return []
    
    # 第一個點保持原樣
    prev_long = df.iloc[0]['Long']
    adjusted_coords.append([df.iloc[0]['Lat'], prev_long])
    
    # 處理後續點
    for i in range(1, len(df)):
        current_long = df.iloc[i]['Long']
        current_lat = df.iloc[i]['Lat']
        
        # 計算經度差異
        long_diff = current_long - prev_long
        
        # 如果差異超過180度，調整經度
        if long_diff > 180:
            # 往西跨越換日線：減去360度
            adjusted_long = current_long - 360
        elif long_diff < -180:
            # 往東跨越換日線：加上360度
            adjusted_long = current_long + 360
        else:
            adjusted_long = current_long
        
        adjusted_coords.append([current_lat, adjusted_long])
        prev_long = adjusted_long  # 更新參考點
    
    return adjusted_coords

# 以軌跡的第一個點為地圖中心
start_lat = gdf['Lat'].iloc[0]
start_lon = gdf['Long'].iloc[0]

# 建立 folium 地圖
m = folium.Map(location=[start_lat, start_lon], zoom_start=10)

# 分割軌跡以處理跨國際換日線問題
segments = split_trajectory_at_dateline(df_filtered)
print(f"軌跡被分割為 {len(segments)} 段")

# 為每段軌跡畫線
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
for i, segment in enumerate(segments):
    color = colors[i % len(colors)]
    folium.PolyLine(
        segment, 
        color=color, 
        weight=3, 
        opacity=0.7,
        popup=f"航段 {i+1} (共{len(segment)}個點)"
    ).add_to(m)

# 加上起點和終點標記
if len(df_filtered) > 0:
    # 起點標記
    start_point = df_filtered.iloc[0]
    folium.Marker(
        location=[start_point['Lat'], start_point['Long']],
        popup=f"起點\n時間: {start_point['Timestamp']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    # 終點標記
    end_point = df_filtered.iloc[-1]
    folium.Marker(
        location=[end_point['Lat'], end_point['Long']],
        popup=f"終點\n時間: {end_point['Timestamp']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

# 儲存地圖為 HTML 檔案
m.save("ship_trajectory_fixed.html")
print("地圖已儲存為 ship_trajectory_fixed.html")

# 自動用瀏覽器開啟（可選）
import webbrowser
webbrowser.open('file://' + os.path.realpath("ship_trajectory_fixed.html"))