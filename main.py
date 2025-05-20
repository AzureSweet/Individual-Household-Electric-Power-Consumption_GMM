import os
# Tiền xử lý dữ liệu 
print("\n=== Data Cleaning ===")
os.system("data_cleaning.py")

# Chạy phân tích GMM
print("\n=== Running GMM Analysis ===")
os.system("gmm_analysis.py")

# Huấn luyện Informer
print("\n=== Training Informer Model ===")
os.system("transformer_forecasting.py")

# Đánh giá mô hình
print("\n=== Evaluating Informer Model ===")
os.system("evaluation.py")
