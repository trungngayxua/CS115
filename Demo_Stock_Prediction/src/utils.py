import csv
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data: np.ndarray) -> None:
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        eps = 1e-8
        return (data - self.min_) / (self.max_ - self.min_ + eps)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        eps = 1e-8
        return data * (self.max_ - self.min_ + eps) + self.min_

    def load_params(self, min_vals: np.ndarray, max_vals: np.ndarray) -> None:
        self.min_ = np.array(min_vals)
        self.max_ = np.array(max_vals)

    def get_params(self):
        return {"min_": self.min_, "max_": self.max_}


def ensure_demo_data(train_path: Path, test_path: Path, seed: int = 0, force_regen: bool = False) -> None:
    """Tao du lieu synthetic. Mac dinh khong ghi de neu file da ton tai."""
    rng = np.random.default_rng(seed)
    if train_path.exists() and test_path.exists() and not force_regen:
        return

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = 420
    train_points = 320
    start_price = 100.0
    t = np.arange(total_points)
    seasonal = 3.0 * np.sin(t / 12) + 1.5 * np.sin(t / 25 + 0.5)
    cycle = 0.8 * np.sin(t / 6 + 1.2)
    drift = 0.02 * np.sin(t / 90)
    noise = rng.normal(scale=1.0, size=total_points)
    prices = start_price + seasonal + cycle + drift + noise

    start_date = datetime(2010, 1, 1)

    def write_csv(path: Path, values: np.ndarray, offset: int) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Close"])
            for i, price in enumerate(values):
                date = start_date + timedelta(days=offset + i)
                writer.writerow([date.strftime("%Y-%m-%d"), f"{price:.4f}"])

    write_csv(train_path, prices[:train_points], 0)
    write_csv(test_path, prices[train_points:], train_points)


def load_stock_data(path: Path) -> np.ndarray:
    """Doc CSV voi cot Date,Close. Tra ve mang (N,1) gia da sort theo ngay."""
    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get("Date", "")
            try:
                dt = datetime.fromisoformat(date_str)
            except ValueError:
                dt = date_str
            close = float(row["Close"])
            records.append((dt, close))
    records.sort(key=lambda r: r[0])
    values = np.array([r[1] for r in records], dtype=float).reshape(-1, 1)
    return values


def create_sequences(series: np.ndarray, lookback: int):
    """Tao (X, y) cho sliding window. X: (n, lookback, 1), y: (n, 1)."""
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def plot_loss_curve(losses, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_gradients(raw_norms, clipped_norms, tau: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(raw_norms, label="Raw grad norm")
    plt.plot(clipped_norms, label="Clipped grad norm")
    plt.axhline(tau, color="red", linestyle="--", label=f"tau={tau}")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient L2 norm")
    plt.title("Gradient clipping log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_predictions(targets, preds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(targets, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.xlabel("Timestep")
    plt.ylabel("Close price")
    plt.title("Test prediction vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_metrics(targets: np.ndarray, preds: np.ndarray):
    mse = np.mean((targets - preds) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(targets - preds))
    return rmse, mae


def save_metrics(rmse: float, mae: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
