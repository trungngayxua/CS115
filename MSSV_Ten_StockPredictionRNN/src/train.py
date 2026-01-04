import argparse
import pickle
from pathlib import Path

import numpy as np

from optimizer import clip_gradients
from rnn_core import RNN
from utils import (
    MinMaxScaler,
    create_sequences,
    ensure_demo_data,
    load_stock_data,
    plot_gradients,
    plot_loss_curve,
)


def init_grad_acc(model: RNN):
    return {
        "W_hx": np.zeros_like(model.W_hx),
        "W_hh": np.zeros_like(model.W_hh),
        "W_qh": np.zeros_like(model.W_qh),
        "b_h": np.zeros_like(model.b_h),
        "b_q": np.zeros_like(model.b_q),
    }


def train_one_epoch(model: RNN, X, y, lr: float, tau: float):
    total_loss = 0.0
    grad_acc = init_grad_acc(model)

    for i in range(len(X)):
        outputs, _, cache = model.forward(X[i], return_cache=True)
        loss, grads = model.backward(cache, y[i])
        total_loss += loss
        for k in grad_acc:
            grad_acc[k] += grads[k]

    total_loss /= len(X)
    for k in grad_acc:
        grad_acc[k] /= len(X)

    clipped_grads, raw_norm, clipped_norm = clip_gradients(grad_acc, tau)
    model.apply_grads(clipped_grads, lr)
    return total_loss, raw_norm, clipped_norm


def main():
    parser = argparse.ArgumentParser(description="Train traditional RNN with manual BPTT.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=16, dest="hidden_size")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=5.0, help="Gradient clipping threshold")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    parameters_dir = base_dir / "parameters"
    results_dir = base_dir / "results"

    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data.csv"
    ensure_demo_data(train_path, test_path, seed=args.seed)

    prices_train = load_stock_data(train_path)
    scaler = MinMaxScaler()
    scaler.fit(prices_train)
    scaled_train = scaler.transform(prices_train)

    X_train, y_train = create_sequences(scaled_train, args.lookback)
    if len(X_train) == 0:
        raise ValueError("Khong du du lieu de tao sequence, hay giam lookback hoac them du lieu.")

    model = RNN(input_size=1, hidden_size=args.hidden_size, output_size=1, seed=args.seed)

    losses, raw_norms, clipped_norms = [], [], []
    for epoch in range(args.epochs):
        loss, raw_norm, clipped_norm = train_one_epoch(model, X_train, y_train, args.lr, args.tau)
        losses.append(loss)
        raw_norms.append(raw_norm)
        clipped_norms.append(clipped_norm)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d} | loss={loss:.6f} | grad_norm={raw_norm:.4f} | clipped={clipped_norm:.4f}"
            )

    parameters_dir.mkdir(parents=True, exist_ok=True)
    weights, biases = model.get_parameters()
    np.save(parameters_dir / "rnn_weights.npy", weights, allow_pickle=True)
    np.save(parameters_dir / "rnn_biases.npy", biases, allow_pickle=True)
    with open(parameters_dir / "scaler_params.pkl", "wb") as f:
        pickle.dump({"min_": scaler.min_, "max_": scaler.max_, "lookback": args.lookback}, f)

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(losses, results_dir / "training_loss.png")
    plot_gradients(raw_norms, clipped_norms, args.tau, results_dir / "gradient_norm_log.png")
    print("Da luu tham so vao", parameters_dir)
    print("Da luu bieu do vao", results_dir)


if __name__ == "__main__":
    main()
