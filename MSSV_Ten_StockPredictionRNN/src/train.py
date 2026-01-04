"""
Train a handcrafted Traditional RNN on stock prices and visualize
loss + gradient norms (with clipping) to illustrate BPTT and exploding gradients.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from optimizer import apply_gradients, clip_gradients
from rnn_core import TraditionalRNN
from utils import MinMaxScaler, create_sequences, load_stock_data


SEED = 42


def train_one_epoch(
    model: TraditionalRNN,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    tau: float,
) -> Tuple[float, float, float]:
    """Train over all sequences (full-batch) and return loss + gradient norms."""
    total_loss = 0.0
    raw_norms, clipped_norms = [], []

    for seq, target in zip(X, y):
        outputs, _ = model.forward(seq)
        pred = outputs[-1]
        loss = 0.5 * float(np.mean((pred - target) ** 2))
        total_loss += loss

        grads = model.backward(y_true=target, y_pred=outputs)
        clipped_grads, raw_norm, clipped_norm = clip_gradients(grads, tau=tau)

        apply_gradients(model.parameters(), clipped_grads, lr)
        raw_norms.append(raw_norm)
        clipped_norms.append(clipped_norm)

    n_samples = max(len(X), 1)
    avg_loss = total_loss / n_samples
    return avg_loss, float(np.mean(raw_norms)), float(np.mean(clipped_norms))


def evaluate(model: TraditionalRNN, X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    preds = []
    for seq in X:
        outputs, _ = model.forward(seq)
        preds.append(outputs[-1, 0])
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Train Loss (clipped)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss with Gradient Clipping")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gradients(raw_norms: list[float], clipped_norms: list[float], tau: float, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(raw_norms, label="Raw grad norm", color="tomato")
    plt.plot(clipped_norms, label="Clipped grad norm", color="steelblue")
    plt.axhline(y=tau, color="gray", linestyle="--", linewidth=0.8, label="Tau")
    plt.xlabel("Epoch")
    plt.ylabel("L2 norm")
    plt.title("Gradient Norms (Clipping to avoid exploding gradients)")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(actual, label="Actual", color="forestgreen")
    plt.plot(predicted, label="Predicted", color="darkorange")
    plt.xlabel("Time step (test set)")
    plt.ylabel("Close price")
    plt.title("Stock Price Prediction (Traditional RNN)")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(SEED)
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "stock_price.csv"
    results_dir = project_root / "results"
    parameters_dir = project_root / "parameters"
    results_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir.mkdir(parents=True, exist_ok=True)

    # Data
    df = load_stock_data(data_path)
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(prices)
    scaled_prices = scaler.transform(prices)

    X, y = create_sequences(scaled_prices, lookback=args.lookback)

    split_idx = int(len(X) * args.train_ratio)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    model = TraditionalRNN(
        input_size=1, hidden_size=args.hidden_size, output_size=1, rng=rng
    )

    losses, raw_norms, clipped_norms = [], [], []

    for epoch in range(args.epochs):
        loss, raw_norm, clipped_norm = train_one_epoch(
            model, X_train, y_train, lr=args.lr, tau=args.tau
        )
        losses.append(loss)
        raw_norms.append(raw_norm)
        clipped_norms.append(clipped_norm)

        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            print(
                f"[epoch {epoch+1:03d}] loss={loss:.6f} "
                f"raw_grad_norm={raw_norm:.4f} clipped_grad_norm={clipped_norm:.4f}"
            )

    # Evaluate on test set
    y_test_inv = scaler.inverse_transform(y_test).flatten()
    preds_inv = evaluate(model, X_test, scaler)

    # Save artifacts
    np.save(parameters_dir / "weights.npy", model.parameters())
    np.save(parameters_dir / "biases.npy", {"b_h": model.b_h, "b_q": model.b_q})
    plot_loss_curve(losses, results_dir / "loss_curve_clipped.png")
    plot_gradients(raw_norms, clipped_norms, args.tau, results_dir / "gradients_norm.png")
    plot_predictions(y_test_inv, preds_inv, results_dir / "prediction_chart.png")

    print(f"[done] Saved weights/biases to {parameters_dir}")
    print(f"[done] Plots written to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Traditional RNN with gradient clipping.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden state size")
    parser.add_argument("--lookback", type=int, default=20, help="Sequence length (window size)")
    parser.add_argument(
        "--tau", type=float, default=0.12, help="Gradient clipping threshold (L2 norm)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train split ratio for sequences"
    )
    main(parser.parse_args())
