import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_and_prepare_data

# ==== Cấu hình ====
FILEPATH = "processed_power_data.csv"
INPUT_LEN = 24
OUTPUT_LEN = 3
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

# ==== Định nghĩa mô hình Informer tối giản ====
class InformerBlock(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class InformerModel(nn.Module):
    def __init__(self, input_len, output_len, input_dim=1, model_dim=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder = InformerBlock(input_dim, model_dim, num_heads)
        self.decoder = nn.GRU(model_dim, model_dim, batch_first=True)
        self.projection = nn.Linear(model_dim, 1)
        self.output_len = output_len

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.embedding(x)
        x = self.encoder(x)
        dec_input = x[:, -1:, :].repeat(1, self.output_len, 1)
        out, _ = self.decoder(dec_input)
        return self.projection(out).squeeze(-1)

# ==== Dữ liệu ====
train_dataset, test_dataset, scaler = load_and_prepare_data(FILEPATH, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==== Huấn luyện ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InformerModel(INPUT_LEN, OUTPUT_LEN).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ==== Lưu mô hình ====
torch.save(model.state_dict(), "informer_model.pth")
