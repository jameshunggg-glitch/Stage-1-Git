## Load the Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

file_dir = r"C:\Users\slab\Desktop\Stage1\data\Device_AB00035.csv"
df = pd.read_csv(file_dir, low_memory=False)

## Data Preprocessing
# 只留下目標船資料
mmsi_target = 416426000

df_filtered = df[df['MMSI'] == mmsi_target].copy()

# 去掉不合理經緯度
df_filtered = df_filtered[
    (df_filtered['Lat'] >= -90) & (df_filtered['Lat'] <= 90) &
    (df_filtered['Long'] >= -180) & (df_filtered['Long'] <= 180)
].copy()

# 將 lat, long 轉 float，ReceiveTime 轉 datetime64
df_filtered['Lat'] = df_filtered['Lat'].astype(float)
df_filtered['Long'] = df_filtered['Long'].astype(float)
df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'],format='%Y%m%d%H%M%S', errors='coerce')

# 丟掉缺失資料
df_filtered = df_filtered.dropna(subset=['Lat', 'Long', 'Timestamp'])
df_filtered['Long_360'] = df_filtered['Long'] % 360
df_filtered['Rot'] = pd.to_numeric(df_filtered['Rot'], errors='coerce')

# 確保時間排序
df_filtered = df_filtered.sort_values('Timestamp').reset_index(drop=True)

# 計算經緯度差分
df_filtered['Delta_Lat'] = df_filtered['Lat'].diff().abs()
df_filtered['Delta_Long'] = df_filtered['Long_360'].diff().abs()

# 計算時間差（秒）
df_filtered['Delta_Time'] = df_filtered['Timestamp'].diff().dt.total_seconds()

# 設置短時間閥值（秒）與 delta 閥值
short_time_thresh = 1.0   # 秒
sog_threshold = 0.5       # knots
rate_threshold = 0.001   # 經緯度差閾值，約100m，可調整

# 修正 delta 經緯度
df_filtered['Adj_Delta_Lat'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Lat'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Lat']
)
df_filtered['Adj_Delta_Long'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Long'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Long']
)

# 停泊判斷
df_filtered['Is_Stop'] = (
    (df_filtered['Sog'] < sog_threshold) &
    (df_filtered['Adj_Delta_Lat'].abs() < rate_threshold) &
    (df_filtered['Adj_Delta_Long'].abs() < rate_threshold)
)

# 檢查 True/False 分布
print(df_filtered['Is_Stop'].value_counts())
print(df_filtered['Is_Stop'].value_counts(normalize=True))

# 視覺化
df_filtered['Is_Stop'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
plt.xticks([0,1], ['False (航行)','True (停泊)'], rotation=0)
plt.ylabel('Count')
plt.title('停泊狀態 True/False 分布')
plt.show()

plt.figure(figsize=(15,3))
plt.plot(df_filtered['Timestamp'], df_filtered['Is_Stop'].astype(int), marker='o', linestyle='-', markersize=2)
plt.ylabel('Is_Stop (1=True, 0=False)')
plt.xlabel('Timestamp')
plt.title('停泊狀態時間序列')
plt.show()

# -----------------------------
# 找停泊區段（改良版，容忍短暫中斷）
# -----------------------------
max_gap_sec = 120  # 允許 2 分鐘內 False 不切斷停泊

stop_segments = []
current_start = None
last_stop_time = None

for i in range(len(df_filtered)):
    if df_filtered.loc[i, 'Is_Stop']:
        if current_start is None:
            current_start = df_filtered.loc[i, 'Timestamp']
        last_stop_time = df_filtered.loc[i, 'Timestamp']
    else:
        if current_start is not None:
            gap = (df_filtered.loc[i, 'Timestamp'] - last_stop_time).total_seconds()
            if gap > max_gap_sec:
                # 結束停泊段
                current_end = last_stop_time
                duration = (current_end - current_start).total_seconds()
                stop_segments.append((current_start, current_end, duration))
                current_start = None
                last_stop_time = None

# 最後一段停泊若持續到資料尾端
if current_start is not None:
    current_end = df_filtered['Timestamp'].iloc[-1]
    duration = (current_end - current_start).total_seconds()
    stop_segments.append((current_start, current_end, duration))

# 轉成 DataFrame
stops_df = pd.DataFrame(stop_segments, columns=['StartTime','EndTime','Duration_sec'])

#過濾短暫停泊 (<30分鐘)
stops_df = stops_df[stops_df['Duration_sec'] >= 1800].reset_index(drop=True)

print("停泊區段：")
print(stops_df)

# 計算總停泊時間與航行時間
total_stop_time = stops_df['Duration_sec'].sum()
total_time = (df_filtered['Timestamp'].max() - df_filtered['Timestamp'].min()).total_seconds()
sailing_time = total_time - total_stop_time

print(f"總停泊時間 (hr): {total_stop_time/3600:.2f}")
print(f"航行時間 (hr): {sailing_time/3600:.2f}")
print(f"停泊比例: {total_stop_time/total_time:.2%}")
print(f"航行比例: {sailing_time/total_time:.2%}")

# 畫停泊/航行時間比例圖
plt.figure(figsize=(6,6))
plt.pie([total_stop_time, sailing_time], labels=['停泊', '航行'], autopct='%1.1f%%', colors=['salmon','skyblue'], startangle=90)
plt.title('停泊/航行時間比例')
plt.show()
