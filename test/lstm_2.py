import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import math

# ----------------------------
# 1. Dataset Definition
# ----------------------------
class WaterQualityDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ----------------------------
# 2. LSTM Model Definition
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ----------------------------
# 3. Train and Evaluate Functions
# ----------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            trues.extend(y.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)

    # Compute metrics
    mse = mean_squared_error(trues, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    return total_loss / len(dataloader), mse, rmse, mae, r2

# ----------------------------
# 4. Main Function
# ----------------------------
def main(args):
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate example data
    np.random.seed(42)
    data = np.sin(np.linspace(0, 100, 500)) + np.random.normal(0, 0.1, 500)
    data = data.reshape(-1, 1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    seq_length = 10
    dataset = WaterQualityDataset(data, seq_length)
    train_size = int(len(dataset) * 0.8)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, mse, rmse, mae, r2 = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    torch.save(model.state_dict(), 'lstm_water_quality.pth')
    print("Model saved as lstm_water_quality.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Water Quality Prediction')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    args = parser.parse_args()
    main(args)
