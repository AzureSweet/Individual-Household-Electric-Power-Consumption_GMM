import pandas as pd
import numpy as np

df = pd.read_csv("household_power_consumption.csv", sep=",", low_memory=False)

# Ghép Date + Time thành Datetime
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

# Xử lý dữ liệu bị lỗi thời gian
df = df.dropna(subset=["Datetime"])

# Ép kiểu dữ liệu
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

# Loại bỏ các dòng thiếu dữ liệu
df = df.dropna(subset=["Global_active_power"])

# Sắp xếp theo thời gian
df = df.set_index("Datetime").sort_index()

# Lấy mẫu theo giờ bằng trung bình
df_hourly = df["Global_active_power"].resample("h").mean()

# Lưu kết quả tiền xử lý
df_hourly.to_csv("processed_power_data.csv")

print("Thành công")
