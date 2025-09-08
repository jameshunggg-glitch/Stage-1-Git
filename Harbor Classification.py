# %% [markdown]
# ## Data Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import os
from glob import glob

# %%
# 讀取 Device 對應檔案
device_file = r"C:\Users\slab\Desktop\Stage1\data\Device 對應.csv"
device_df = pd.read_csv(device_file, encoding="utf-8")

# %%
device_df.head()

# %% [markdown]
# ## Data Preprocessing

# %%
import os
import glob
import pandas as pd

# 設定資料夾路徑
data_path = r"C:\Users\slab\Desktop\Stage1\data"

# 讀取 Device 對應表（UTF-8）
device_df = pd.read_csv(os.path.join(data_path, "Device 對應.csv"), encoding="utf-8")
device_df.columns = device_df.columns.str.strip()  # 清理欄位名稱
device_map = dict(zip(device_df["DeviceID"], device_df["MMSI"]))

# 找出所有船舶檔案 (排除對應表)
csv_files = glob.glob(os.path.join(data_path, "Device_*.csv"))

all_data = []

for file in csv_files:
    filename = os.path.basename(file).replace(".csv", "")
    # 去掉前綴 "Device_"
    device_id = filename.replace("Device_", "")
    
    if device_id in device_map:
        mmsi = device_map[device_id]
        df = pd.read_csv(file, encoding="utf-8")
        
        # 保留該船舶的資料
        df = df[df["MMSI"] == mmsi]
        df["DeviceID"] = device_id
        all_data.append(df)
    else:
        print(f"⚠️ 找不到對應的 DeviceID: {device_id}")



# %%
# 清洗函式
def clean_coordinates(df, lat_col='Lat', lon_col='Long'):
    # 先轉成 float
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    # 刪除缺失值
    df = df.dropna(subset=[lat_col, lon_col])
    
    # 經度在 -180 ~ 180
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)]
    
    # 緯度在 -90 ~ 90
    df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90)]
    
    return df

# 套用到所有船舶資料
cleaned_data = [clean_coordinates(df) for df in all_data]

print(f"清洗完成，共 {len(cleaned_data)} 艘船的資料")


# %% [markdown]
# ## DBSCAN

# %%
from sklearn.cluster import DBSCAN
from haversine import haversine

# 假設 cleaned_data 已經有經緯度清洗完成的所有船舶資料
port_list = []

for df_ship in cleaned_data:
    device_id = df_ship["DeviceID"].iloc[0]  # 取得船舶 ID
    
    # 篩選停泊點（假設 SOG 欄位存在）
    stop_df = df_ship[df_ship["Sog"] < 0.5].copy()
    
    if stop_df.empty:
        continue  # 沒有停泊資料就跳過
    
    # 取經緯度
    coords = stop_df[['Lat', 'Long']].to_numpy()
    coords_rad = np.radians(coords)  # DBSCAN 使用 haversine 需轉成弧度
    
    # DBSCAN 聚類
    db = DBSCAN(eps=0.01, min_samples=10, metric='haversine').fit(coords_rad)
    stop_df['cluster'] = db.labels_
    
    # 計算每個 cluster 的中心與半徑
    for cluster_id in stop_df['cluster'].unique():
        if cluster_id == -1:
            continue  # 忽略噪聲點
        
        cluster_points = stop_df[stop_df['cluster']==cluster_id][['Lat','Long']]
        center_lat = cluster_points['Lat'].mean()
        center_lon = cluster_points['Long'].mean()
        
        # 半徑：到中心最遠距離
        distances = cluster_points.apply(lambda row: haversine((center_lat, center_lon), (row['Lat'], row['Long'])), axis=1)
        radius_km = distances.max()
        
        port_list.append({
            'DeviceID': device_id,
            'cluster': cluster_id,
            'lat': center_lat,
            'lon': center_lon,
            'radius_km': radius_km
        })

# 合併成 DataFrame
port_df = pd.DataFrame(port_list)
print(f"共偵測到 {port_df.shape[0]} 個港口")
print(port_df.head())

# %% [markdown]
# ## Folim on the map

# %%
import folium

# 地圖中心取所有港口平均位置
map_center = [port_df['lat'].mean(), port_df['lon'].mean()]
m = folium.Map(location=map_center, zoom_start=6)

# 繪製港口圓圈
for _, row in port_df.iterrows():
    folium.Circle(
        location=[row['lat'], row['lon']],
        radius=row['radius_km'] * 1000,  # km -> m
        color='blue',
        fill=True,
        fill_opacity=0.3,
        popup=f"{row['DeviceID']} Cluster {row['cluster']}"
    ).add_to(m)


# 儲存地圖到指定資料夾
map_file = r"C:\Users\slab\Desktop\test\ports_map.html"
m.save(map_file)
print(f" 港口地圖已生成：{map_file}")



# %% [markdown]
# ## Time Filter to DBSCAN

# %%
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from haversine import haversine
from datetime import timedelta
import folium

# ------------------------------
# 參數設定
# ------------------------------
min_duration = timedelta(minutes=30)   # 最少停留時間
split_interval = timedelta(hours=48)  # 分段間隔
merge_radius_km = 5                    # 空間合併半徑（公里）
eps_haversine = 0.01                   # DBSCAN eps (radians)
min_samples_dbscan = 10                # DBSCAN min_samples

# ------------------------------
# 假設 cleaned_data 已經包含每艘船資料
# cleaned_data 每個元素是一艘船的 DataFrame，包含 ['DeviceID','Timestamp','Lat','Long','SOG']
# ------------------------------

# 儲存每艘船 cluster 經過時間分段後的有效停泊點
cluster_ports = []

for df_ship in cleaned_data:
    device_id = df_ship["DeviceID"].iloc[0]
    
    # Timestamp 確保是 datetime
    df_ship['Timestamp'] = pd.to_datetime(df_ship['Timestamp'],format='%Y%m%d%H%M%S', errors='coerce')
    
    # 篩選停泊點 (怠速判定)
    stop_df = df_ship[df_ship["Sog"] < 0.5].copy()
    if stop_df.empty:
        continue
    
    # DBSCAN 聚類
    coords = stop_df[['Lat', 'Long']].to_numpy()
    coords_rad = np.radians(coords)
    db = DBSCAN(eps=eps_haversine, min_samples=min_samples_dbscan, metric='haversine').fit(coords_rad)
    stop_df['cluster'] = db.labels_
    
    # 處理每個 cluster
    for cluster_id in stop_df['cluster'].unique():
        if cluster_id == -1:
            continue
        
        cluster_points = stop_df[stop_df['cluster'] == cluster_id].sort_values('Timestamp')
        segment_points = [cluster_points.iloc[0]]
        valid_points = []  # 剔除可疑點後，用於畫圖
        
        for i in range(1, len(cluster_points)):
            current_time = cluster_points['Timestamp'].iloc[i]
            prev_time = cluster_points['Timestamp'].iloc[i-1]
            
            if current_time - prev_time > split_interval:
                # 前一段停留時間
                duration = segment_points[-1]['Timestamp'] - segment_points[0]['Timestamp']
                if duration >= min_duration:
                    valid_points.extend(segment_points)
                segment_points = [cluster_points.iloc[i]]
            else:
                segment_points.append(cluster_points.iloc[i])
        
        # 處理最後一段
        if segment_points:
            duration = segment_points[-1]['Timestamp'] - segment_points[0]['Timestamp']
            if duration >= min_duration:
                valid_points.extend(segment_points)
        
        if valid_points:
            valid_df = pd.DataFrame(valid_points)
            cluster_ports.append(valid_df)

# ------------------------------
# 合併所有 cluster 用於畫地圖
# ------------------------------
all_ports_df = pd.concat(cluster_ports, ignore_index=True)

# 計算每個 cluster 的港口中心與半徑
merged_ports = []
for (device_id, cluster_id), group in all_ports_df.groupby(['DeviceID', 'cluster']):
    center_lat = group['Lat'].mean()
    center_lon = group['Long'].mean()
    distances = group.apply(lambda row: haversine((center_lat, center_lon), (row['Lat'], row['Long'])), axis=1)
    merged_ports.append({
        'DeviceID': device_id,
        'cluster': cluster_id,
        'lat': center_lat,
        'lon': center_lon,
        'radius_km': distances.max(),
        'num_points': len(group)
    })

merged_port_df = pd.DataFrame(merged_ports)
print(f" 最終港口數量: {len(merged_port_df)}")
print(merged_port_df.head())

# ------------------------------
# Folium 畫圖
# ------------------------------
map_center = [merged_port_df['lat'].mean(), merged_port_df['lon'].mean()]
m = folium.Map(location=map_center, zoom_start=6)

for _, row in merged_port_df.iterrows():
    folium.Circle(
        location=[row['lat'], row['lon']],
        radius=row['radius_km'] * 1000,  # 公里轉公尺
        color='blue',
        fill=True,
        fill_opacity=0.3,
        popup=f"DeviceID: {row['DeviceID']}, Points: {row['num_points']}"
    ).add_to(m)

# 儲存地圖
map_file = r"C:\Users\slab\Desktop\test\ports_map_cleaned.html"
m.save(map_file)
print(f"港口地圖已生成：{map_file}")


# %%
# 儲存港口停泊段落資訊
port_segments = []

for df_ship in cleaned_data:
    device_id = df_ship["DeviceID"].iloc[0]
    df_ship['Timestamp'] = pd.to_datetime(df_ship['Timestamp'], format='%Y%m%d%H%M%S', errors='coerce')
    
    stop_df = df_ship[df_ship["Sog"] < 0.5].copy()
    if stop_df.empty:
        continue
    
    coords = stop_df[['Lat', 'Long']].to_numpy()
    coords_rad = np.radians(coords)
    db = DBSCAN(eps=eps_haversine, min_samples=min_samples_dbscan, metric='haversine').fit(coords_rad)
    stop_df['cluster'] = db.labels_
    
    for cluster_id in stop_df['cluster'].unique():
        if cluster_id == -1:
            continue
        
        cluster_points = stop_df[stop_df['cluster'] == cluster_id].sort_values('Timestamp')
        segment_points = [cluster_points.iloc[0]]
        
        for i in range(1, len(cluster_points)):
            current_time = cluster_points['Timestamp'].iloc[i]
            prev_time = cluster_points['Timestamp'].iloc[i-1]
            
            if current_time - prev_time > split_interval:
                # 前一段
                duration = segment_points[-1]['Timestamp'] - segment_points[0]['Timestamp']
                if duration >= min_duration:
                    center_lat = np.mean([p['Lat'] for p in segment_points])
                    center_lon = np.mean([p['Long'] for p in segment_points])
                    port_segments.append({
                        'DeviceID': device_id,
                        'cluster': cluster_id,
                        'lat': center_lat,
                        'lon': center_lon,
                        'duration_minutes': duration.total_seconds()/60
                    })
                segment_points = [cluster_points.iloc[i]]
            else:
                segment_points.append(cluster_points.iloc[i])
        
        # 處理最後一段
        if segment_points:
            duration = segment_points[-1]['Timestamp'] - segment_points[0]['Timestamp']
            if duration >= min_duration:
                center_lat = np.mean([p['Lat'] for p in segment_points])
                center_lon = np.mean([p['Long'] for p in segment_points])
                port_segments.append({
                    'DeviceID': device_id,
                    'cluster': cluster_id,
                    'lat': center_lat,
                    'lon': center_lon,
                    'duration_minutes': duration.total_seconds()/60
                })

# 轉成 DataFrame
port_segments_df = pd.DataFrame(port_segments)

# 計算每個港口靠港次數
port_counts = port_segments_df.groupby(['DeviceID', 'cluster']).size().reset_index(name='num_visits')

# 合併停留時間資訊（可選：平均停留時間或列出每段停留時間）
port_summary = port_segments_df.groupby(['DeviceID', 'cluster']).agg({
    'lat':'mean',
    'lon':'mean',
    'duration_minutes': list  # 每段停留時間列表
}).reset_index()

# 加上靠港次數
port_summary = port_summary.merge(port_counts, on=['DeviceID','cluster'])

print("港口停泊次數與停留時間示例:")
print(port_summary.head())


# %%
port_summary.head()


