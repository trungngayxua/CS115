"""
Utility helpers for data loading, preprocessing, and batching.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class MinMaxScaler:
    """Simple Min-Max scaler (fit/transform/inverse_transform) using numpy."""

    def __init__(self) -> None:
        self.min_ = None
        self.max_ = None

    def fit(self, data: np.ndarray) -> None:
        self.min_ = data.min(axis=0, keepdims=True)
        self.max_ = data.max(axis=0, keepdims=True)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fitted yet.")
        denom = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return (data - self.min_) / denom

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        denom = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return data * denom + self.min_


def maybe_generate_synthetic_csv(data_path: Path, num_points: int = 250) -> None:
    """
    Generate a lightweight synthetic price series if stock_price.csv is missing.
    The series has a mild upward trend and weekly seasonality to mimic real prices.
    """
    if data_path.exists():
        return

    days = np.arange(num_points)
    trend = 50 + 0.1 * days
    seasonality = 1.8 * np.sin(2 * np.pi * days / 7)
    noise = np.random.normal(0, 1.2, size=num_points)
    close = trend + seasonality + noise

    open_price = close + np.random.normal(0, 0.7, size=num_points)
    high = np.maximum(open_price, close) + np.abs(np.random.normal(0, 0.4, size=num_points))
    low = np.minimum(open_price, close) - np.abs(np.random.normal(0, 0.4, size=num_points))
    volume = np.random.randint(800_000, 1_800_000, size=num_points)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_points)
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": open_price.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"[info] Generated synthetic dataset at {data_path}")


def load_stock_data(data_path: Path) -> pd.DataFrame:
    maybe_generate_synthetic_csv(data_path)
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    return df.reset_index(drop=True)


def create_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pairs where X is a window of length `lookback` and y is the next value.
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

