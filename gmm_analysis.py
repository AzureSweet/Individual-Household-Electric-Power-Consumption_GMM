import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load dữ liệu
filepath = "processed_power_data.csv"
df = pd.read_csv(filepath, parse_dates=['Datetime'])
df = df.dropna(subset=['Global_active_power'])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_power = scaler.fit_transform(df[['Global_active_power']])

# Áp dụng Gaussian Mixture Model
n_components = 3  # có thể điều chỉnh
model = GaussianMixture(n_components=n_components, random_state=42)
labels = model.fit_predict(scaled_power)

# Gán nhãn vào dataframe
df['GMM_Label'] = labels

# Vẽ kết quả phân cụm
df['Datetime'] = pd.to_datetime(df['Datetime'])
plt.figure(figsize=(12, 5))
for i in range(n_components):
    cluster = df[df['GMM_Label'] == i]
    plt.plot(cluster['Datetime'], cluster['Global_active_power'], '.', label=f'Cluster {i}', alpha=0.6)
plt.title('GMM Clustering on Global Active Power')
plt.xlabel('Datetime')
plt.ylabel('Global Active Power')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Thành công")