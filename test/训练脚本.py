import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# ================================
# 1. Dataset
# ================================
class WaterQualityDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_length):
            self.X.append(data[i:i + seq_length])
            self.y.append(data[i + seq_length])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ================================
# 2. LSTM Model
# ================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # use last time step
        return out

# ================================
# 3. Training & Evaluation
# ================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds.append(outputs.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = mean_squared_error(trues, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return mse, rmse, mae, r2

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mse, rmse, mae, r2 = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {total_loss/len(train_loader):.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

# ================================
# 4. Main
# ================================
def main():
    # 自动检测 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 读取数据
    df = pd.read_csv('water_quality.csv')
    df = df[['COD', 'DO', 'PH', 'NH3_N', 'TP', 'TN', 'Turbidity']]

    # 缺失值处理
    df = df.interpolate(method='linear').fillna(method='bfill')

    # 归一化
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    seq_length = 10
    dataset = WaterQualityDataset(scaled, seq_length)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=7, hidden_size=64, output_size=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50)

    torch.save(model.state_dict(), "water_quality_lstm_gpu.pth")
    print("✅ 模型已保存：water_quality_lstm_gpu.pth")

if __name__ == "__main__":
    main()
