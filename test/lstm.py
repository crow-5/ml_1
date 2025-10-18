"""
pytorch_lstm_water_quality.py

说明：
- 一个可运行的、注释清晰的 PyTorch LSTM 时间序列回归示例，用于预测未来水质指标（多变量输入 -> 标量或多变量输出）。
- 同一脚本支持 CPU 和 GPU（通过 --device 参数选择）。
- 包含数据集类、模型定义、训练/验证循环、保存/加载模型、以及如何使用自己的数据的提示。

依赖：
- torch, numpy, pandas, scikit-learn

示例运行：
# CPU
python pytorch_lstm_water_quality.py --device cpu
# GPU
python pytorch_lstm_water_quality.py --device cuda

"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """建立滑动窗口序列数据集。
    X: (N_timesteps, n_features) 的原始时间序列
    将其切分为多个样本：每个样本长度 = seq_len，目标为接下来的 pred_len 步（可为 1）
    返回：x: (seq_len, n_features)， y: (pred_len, target_dim)
    """

    def __init__(self, data: np.ndarray, seq_len: int = 24, pred_len: int = 1, target_cols: list = None):
        assert data.ndim == 2, "data 应该为二维数组 (timesteps, features)"
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_cols = target_cols
        self.n_timesteps = data.shape[0]

        # 计算可生成样本数
        self.indices = []
        end = self.n_timesteps - seq_len - pred_len + 1
        for i in range(end):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i : i + self.seq_len]  # (seq_len, n_features)
        y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        if self.target_cols is not None:
            # 假设 target_cols 是特征索引列表
            y = y[:, self.target_cols]
        # 如果 pred_len == 1，返回 (target_dim,) 便于 loss 计算
        if self.pred_len == 1:
            y = y.reshape(-1)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = False, pred_len: int = 1, target_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pred_len = pred_len
        self.target_dim = target_dim

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True,
                            bidirectional=bidirectional)

        # 全连接层把最后时刻的隐藏状态映射到预测输出
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * target_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # 取最后时间步的输出（如果双向应该拼接两端）
        last = out[:, -1, :]  # (batch, hidden_dim * num_directions)
        out = self.fc(last)   # (batch, pred_len * target_dim)
        out = out.view(-1, self.pred_len, self.target_dim)
        if self.pred_len == 1:
            out = out.squeeze(1)  # (batch, target_dim) 或 (batch,) if target_dim == 1
        return out


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def prepare_data_from_csv(csv_path: str, feature_cols: list = None) -> Tuple[np.ndarray, StandardScaler]:
    """示例：从 CSV 加载数据，要求 CSV 行为时间步，列为特征（如 DO, pH, Turbidity ...）
    返回标准化后的 numpy 数组和 scaler（用于预测时反归一化）
    """
    df = pd.read_csv(csv_path)
    if feature_cols is None:
        feature_cols = df.columns.tolist()
    data = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


def create_dataloaders(data: np.ndarray, seq_len: int, pred_len: int, target_cols: list,
                       batch_size: int = 64, val_ratio: float = 0.2):
    n = data.shape[0]
    split = int(n * (1 - val_ratio))
    train_data = data[:split]
    val_data = data[split:]

    train_ds = TimeSeriesDataset(train_data, seq_len=seq_len, pred_len=pred_len, target_cols=target_cols)
    val_ds = TimeSeriesDataset(val_data, seq_len=seq_len, pred_len=pred_len, target_cols=target_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_training(csv_path: str = None,
                 device: torch.device = torch.device('cpu'),
                 seq_len: int = 24,
                 pred_len: int = 1,
                 target_cols: list = None,
                 input_dim: int = None,
                 epochs: int = 30,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 save_path: str = 'model.pth'):

    # 如果没有传 csv_path，则生成模拟数据用于演示
    if csv_path is None:
        print("没有提供 CSV，生成随机演示数据（请用真实水质数据替换）")
        rng = np.random.RandomState(42)
        # 1000 个时间步、五个特征（例如 DO, pH, Temp, Turbidity, EC）
        data = rng.normal(size=(2000, 5)).astype(np.float32)
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
        feature_cols = list(range(data.shape[1]))
    else:
        data, scaler = prepare_data_from_csv(csv_path)
        feature_cols = list(range(data.shape[1]))

    if input_dim is None:
        input_dim = data.shape[1]

    if target_cols is None:
        # 默认预测第一列（水质目标指标），可修改为多列
        target_cols = [0]

    train_loader, val_loader = create_dataloaders(data, seq_len, pred_len, target_cols, batch_size=batch_size)

    target_dim = len(target_cols)

    model = LSTMForecaster(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2,
                          bidirectional=False, pred_len=pred_len, target_dim=target_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        t1 = time.time()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'args': {
                    'seq_len': seq_len,
                    'pred_len': pred_len,
                    'target_cols': target_cols,
                    'input_dim': input_dim
                }
            }, save_path)
        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}  time: {t1-t0:.1f}s")

    print(f"训练结束。最佳验证 loss={best_val:.6f}，发生在 epoch {best_epoch}。模型保存在 {save_path}")


def load_model(path: str, device: torch.device = torch.device('cpu')):
    ckpt = torch.load(path, map_location=device)
    args = ckpt.get('args', {})
    model = LSTMForecaster(input_dim=args.get('input_dim', 5),
                           hidden_dim=128, num_layers=2, dropout=0.2,
                           bidirectional=False, pred_len=args.get('pred_len', 1),
                           target_dim=len(args.get('target_cols', [0])))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    scaler = ckpt.get('scaler', None)
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='训练数据 CSV 文件路径（时间步为行，列为特征）')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default='water_lstm.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    run_training(csv_path=args.csv, device=device, seq_len=args.seq_len, pred_len=args.pred_len,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save)
