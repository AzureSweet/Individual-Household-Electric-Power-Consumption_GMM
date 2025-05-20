import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import load_and_prepare_data, plot_predictions
from transformer_forecasting import InformerModel, INPUT_LEN, OUTPUT_LEN

# ==== Load dữ liệu ====
FILEPATH = "processed_power_data.csv"
_, test_dataset, scaler = load_and_prepare_data(FILEPATH, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
test_loader = DataLoader(test_dataset, batch_size=1)

# ==== Load mô hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InformerModel(INPUT_LEN, OUTPUT_LEN).to(device)
model.load_state_dict(torch.load("informer_model.pth", map_location=device))
model.eval()

# ==== Dự báo ====
preds, trues = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        output = model(X).cpu().numpy().flatten()
        label = y.numpy().flatten()
        preds.extend(output)
        trues.extend(label)

# ==== Đưa về giá trị gốc ====
preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
trues_inv = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()

# ==== Đánh giá ====
mae = mean_absolute_error(trues_inv, preds_inv)
mse = mean_squared_error(trues_inv, preds_inv)
rmse = np.sqrt(mse)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# ==== Vẽ kết quả ====
plot_predictions(trues_inv[:100], preds_inv[:100], title='Informer Forecast vs Actual')
