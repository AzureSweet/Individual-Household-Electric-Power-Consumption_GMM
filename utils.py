import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_len=24, output_len=3):
        self.X, self.y = [], []
        for i in range(len(series) - input_len - output_len):
            self.X.append(series[i:i+input_len])
            self.y.append(series[i+input_len:i+input_len+output_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_prepare_data(filepath, col_name="Global_active_power", input_len=24, output_len=3, train_ratio=0.8):
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    df = df.dropna(subset=[col_name])
    series = df[col_name].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series).flatten()

    split_point = int(len(scaled_series) * train_ratio)
    train_series = scaled_series[:split_point]
    test_series = scaled_series[split_point - input_len - output_len:]

    train_dataset = TimeSeriesDataset(train_series, input_len, output_len)
    test_dataset = TimeSeriesDataset(test_series, input_len, output_len)

    return train_dataset, test_dataset, scaler

def plot_predictions(true, pred, title='Forecast vs Actual'):
    plt.figure(figsize=(10, 4))
    plt.plot(true, label='Actual')
    plt.plot(pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
