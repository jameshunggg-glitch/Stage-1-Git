# %% [markdown]
# ## Load Data

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from geopy.distance import geodesic


# %%
file_dir = r"C:\Users\slab\Desktop\Stage1\data\Device_AB00035.csv"
df = pd.read_csv(file_dir, low_memory=False)

## Data Preprocessing
# 只留下目標船資料
mmsi_target = 416426000

df_filtered = df[df['MMSI'] == mmsi_target].copy()

# %%

df_filtered = df_filtered[
    (df_filtered['Lat'] >= -90) & (df_filtered['Lat'] <= 90) &
    (df_filtered['Long'] >= -180) & (df_filtered['Long'] <= 180)
].copy()
df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'],format='%Y%m%d%H%M%S', errors='coerce')

# 將 lat, long 轉 float，ReceiveTime 轉 datetime64
df_filtered['Lat'] = df_filtered['Lat'].astype(float)
df_filtered['Long'] = df_filtered['Long'].astype(float)
df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'], errors='coerce')

# 丟掉缺失資料
df_filtered = df_filtered.dropna(subset=['Lat', 'Long', 'Timestamp'])
df_filtered['Long_360'] = df_filtered['Long'] % 360
df_filtered['Rot'] = pd.to_numeric(df_filtered['Rot'], errors='coerce')

# 確保時間排序
df_filtered = df_filtered.sort_values('Timestamp').reset_index(drop=True)
#df_sorted = df_filtered.sort_values('ReceiveTime', kind="mergesort")


# %%
df_filtered.head(20)

# %% [markdown]
# ## Type I Error

# %%
def mark_same_second_anomaly(df):
    df = df.copy()
    df['anomaly_label'] = df.get('anomaly_label', 0)  # 若已有欄位就保留
    
    for t, group in df.groupby('Timestamp'):
        if len(group) == 1:
            continue  # 單筆資料不可能同秒異常
        
        for coord in ['Lat', 'Long']:
            vals = group[coord].to_numpy()
            Q1 = np.percentile(vals, 25)
            Q3 = np.percentile(vals, 75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue  # 沒有變化，跳過
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outlier_idx = group.index[(vals < lower) | (vals > upper)]
            df.loc[outlier_idx, 'anomaly_label'] = 1
    
    return df

# %%
df_marked = mark_same_second_anomaly(df_filtered)

# %%
df_marked['anomaly_label'].value_counts()


# %% [markdown]
# ### Type I error verification

# %%
import matplotlib.pyplot as plt

# 找出第一類異常的索引（前五個）
anomaly_idx = df_marked[df_marked['anomaly_label'] == 1].index[:5]

for idx in anomaly_idx:
    anomaly_time = df_marked.loc[idx, 'Timestamp']
    # 找出該秒內的所有資料
    same_second = df_marked[df_marked['Timestamp'] == anomaly_time]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(same_second['Long'], same_second['Lat'], color='blue', label='正常點')
    # 標記被判定為異常的點
    plt.scatter(df_marked.loc[idx, 'Long'], df_marked.loc[idx, 'Lat'], color='red', label='異常點', s=100)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'AIS Type I Error - {anomaly_time}')
    plt.legend()
    plt.show()


# %% [markdown]
# ## Type II Error 

# %% [markdown]
# ### 先收起來等等用

# %%
def mark_short_interval_anomaly_simple(df_mark, max_interval=300, lat_tol=0.001, lon_tol=0.001):
    """
    標記第二類短時間內異常點 (簡化版)
    ----------
    df_mark : DataFrame
        需要包含 ['ReceiveTime', 'Lat', 'Long', 'anomaly_label'] 欄位
    max_interval : int
        判定短時間異常的最大秒, 例如5分鐘 = 300秒
    lat_tol : float
        緯度允許變化範圍，小於此值算正常
    lon_tol : float
        經度允許變化範圍，小於此值算正常
    """
    df_mark = df_mark.sort_values('Timestamp').copy()
    df_mark['anomaly_label'] = df_mark['anomaly_label'].astype(int)

    times = df_mark['Timestamp'].to_numpy()
    lats = df_mark['Lat'].to_numpy()
    lons = df_mark['Long'].to_numpy()

    for i in range(1, len(df_mark)):
        delta_t = (times[i] - times[i-1]) / np.timedelta64(1, 's')  # 秒

        if delta_t <= 0 or delta_t > max_interval:
            continue  # 只處理時間間隔在 0~max_interval 的相鄰點

        lat_diff = abs(lats[i] - lats[i-1])/delta_t
        lon_diff = abs(lons[i] - lons[i-1])/delta_t

        if lat_diff > lat_tol or lon_diff > lon_tol:
            # 只標記原本 anomaly_label == 0 的點
            if df_mark.at[i, 'anomaly_label'] == 0:
                df_mark.at[i, 'anomaly_label'] = 2

    return df_mark


# %%
# 標記第二類錯誤
df_marked = mark_short_interval_anomaly_simple(df_marked, max_interval=300, lat_tol=0.001, lon_tol=0.001)


# %%
df_marked[df_marked['anomaly_label'] == 2][['PKY','Timestamp','Lat','Long']].head()


# %%
# 找出第二類異常點及其前一筆資料
def show_short_interval_anomalies(df):
    # 先找出所有 anomaly_label == 2 的索引
    anomaly_idx = df.index[df['anomaly_label'] == 2].tolist()
    
    rows_to_show = []
    for idx in anomaly_idx:
        if idx > 0:
            # 包含前一筆資料
            rows_to_show.append(idx-1)
        rows_to_show.append(idx)
    
    # 移除重複的索引
    rows_to_show = sorted(list(set(rows_to_show)))

    # 顯示 Timestamp, Lat, Long, anomaly_label
    return df.loc[rows_to_show, ['Timestamp', 'Lat', 'Long', 'anomaly_label']]

# 使用範例
df_anomaly_check = show_short_interval_anomalies(df_marked)
print(df_anomaly_check)


# %% [markdown]
# ### Type II Error verification

# %%
import matplotlib.pyplot as plt

# 找出第二類異常的索引（前五個）
anomaly_idx = df_marked[df_marked['anomaly_label'] == 2].index[:5]

for idx in anomaly_idx:
    prev_idx = idx - 1
    if prev_idx < 0:
        continue  # 如果沒有前一筆就跳過
    
    subset = df_marked.loc[[prev_idx, idx], ['Timestamp', 'Lat', 'Long']]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # ---- 緯度 ----
    axes[0].plot(subset['Timestamp'], subset['Lat'], marker='o', color='blue', label='資料點')
    axes[0].scatter(subset.loc[idx, 'Timestamp'], subset.loc[idx, 'Lat'], 
                    color='red', s=100, label='異常點')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title(f'第二類異常 - Index {idx}')
    axes[0].legend()

    # ---- 經度 ----
    axes[1].plot(subset['Timestamp'], subset['Long'], marker='o', color='blue', label='資料點')
    axes[1].scatter(subset.loc[idx, 'Timestamp'], subset.loc[idx, 'Long'], 
                    color='red', s=100, label='異常點')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Longitude')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# %%
df_marked.head(10)
print(len(df_marked))

# %% [markdown]
# ## Type III Error

# %%
# 標記第三類錯誤
def mark_long_interval_anomaly(df_mark, max_interval=6*3600, max_speed=15):
    """
    標記第三類長時間間隔異常點
    ----------
    df_mark : DataFrame
        需要包含 ['ReceiveTime', 'Lat', 'Long', 'anomaly_label'] 欄位
    max_interval : int
        判定長時間異常的最小秒數，例如6小時 = 21600秒
    max_speed : float
        長時間間隔下合理最大航速 (m/s)，超過標記異常
    """
    df_mark = df_mark.sort_values('Timestamp').copy()
    df_mark['anomaly_label'] = df_mark['anomaly_label'].astype(int)

    times = df_mark['Timestamp'].to_numpy()
    coords = df_mark[['Lat', 'Long']].to_numpy()

    for i in range(1, len(df_mark)):
        delta_t = (times[i] - times[i-1]) / np.timedelta64(1, 's')  # 秒

        if delta_t <= max_interval:
            continue  # 只處理超過 max_interval 的相鄰點

        dist = geodesic(coords[i-1], coords[i]).meters
        sog = dist / delta_t

        if sog > max_speed:
            if df_mark.at[i, 'anomaly_label'] == 0:
                df_mark.at[i, 'anomaly_label'] = 3

    return df_mark


# %%
df_marked = mark_long_interval_anomaly(df_marked, max_interval=6*3600, max_speed=15)


# %%
df_marked['anomaly_label'].value_counts()


# %%
df_marked.head(20)


