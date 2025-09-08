# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import movingpandas as mpd
from shapely.geometry import Point
import os

# %%
file_path = r"C:\Users\slab\Desktop\Stage1\data\Device_AB00006.csv"


# %%
try:
    df
except NameError:
    df = pd.read_csv(file_path, low_memory=False)
df = pd.read_csv(file_path, low_memory=False)


# %%
df.dtypes

# %%
len(df)

# %%
# 我只想畫出416464000這艘船的航線圖，先把其他的洗掉
mmsi_target = 416464000
df_filtered = df[df['MMSI'] == mmsi_target].copy()
print(len(df_filtered))
print(df_filtered.head())

# %%
#大概看一下有效資料占比
print("有效data數量約為:", len(df_filtered)/len(df))

# %%
# found that some error value has been recorded
# lat should between [90,-90], long between [180, -180]
df_filtered = df_filtered[
    (df_filtered['Lat'] >= -90) & (df_filtered['Lat'] <= 90) &
    (df_filtered['Long'] >= -180) & (df_filtered['Long'] <= 180)
].copy()

# 再檢查經緯度是否都合理
print(df_filtered[['Lat', 'Long']].describe())

# %%
print("有效data數量約為:", len(df_filtered)/len(df))

# %%
# check out columns
print(df_filtered.columns)

# %%
# Modify the logitude and latitude 欄位
df_filtered['Lat'] = df_filtered['Lat'].astype(float)
df_filtered['Long'] = df_filtered['Long'].astype(float)


# %%
# Timestamp to datetime
df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'], errors='coerce')

# %%
# dropna
df_filtered = df_filtered.dropna(subset=['Lat', 'Long', 'Timestamp'])

# %%
# 為了不讓圖片上線斷掉，將經度[-180, 180] 轉成 [0, 360]
df_filtered['Long_360'] = df_filtered['Long'] % 360


# %%
# 計算地圖中心
map_center_lat = df_filtered['Lat'].mean()
map_center_lon = df_filtered['Long_360'].mean()

# %%
print(df_filtered.dtypes)
#print(df_filtered.head())

# %%
import movingpandas as mpd
import geopandas as gpd

# %%
# sort by the datetime
#df_filtered = df_filtered.sort_values(df_filtered['Timestamp']).reset_index(drop = True)
df_filtered = df_filtered.sort_values('Timestamp').reset_index(drop=True)
# build up GeoDataFrame
##
""" gdf = gpd.GeoDataFrame(
    df_filtered,
    geometry=gpd.points_from_xy(df_filtered['Long'], df_filtered['Lat']),
    crs="EPSG:4326"
) """
##
m = folium.Map(
    location=[map_center_lat, map_center_lon],
    zoom_start=4,
    tiles='OpenStreetMap'
)

# %%
# Generate MovingPandas trajectory
# traj = mpd.Trajectory(gdf, traj_id=str(df_filtered['MMSI'].iloc[0]), t='Timestamp')


# %%
# print(traj.df.index)

# %%
# print(traj)
#print("軌跡點數：", len(traj))

# %%
# print(type(traj))
# print("軌跡點數：", len(traj.df))

# %%
# print(traj.df.columns)

# %%
# %%
import folium

coords = df_filtered[['Lat', 'Long_360']].iloc[::1].values.tolist()  # ::10 可抽樣10倍
folium.PolyLine(
    coords,
    color='blue',
    weight=3,
    opacity=0.7,
    popup=f"航線點數: {len(coords)}"
).add_to(m)

# 起訖點標記

if len(coords) > 0:
    # 起點
    folium.Marker(
        location=coords[0],
        popup=f"起點\n時間: {df_filtered.iloc[0]['ReceiveTime']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    # 終點
    folium.Marker(
        location=coords[-1],
        popup=f"終點\n時間: {df_filtered.iloc[-1]['ReceiveTime']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

# %%
html_file = "ship_trajectory_360.html"
m.save(html_file)

# %%
# 自動用瀏覽器開啟
import webbrowser
webbrowser.open('file://' + os.path.realpath(html_file))




