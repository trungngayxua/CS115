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
    download_and_split_stock_data,
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
    parser.add_argument("--symbol", type=str, default="aapl", help="Ma co phieu tren stooq (vd: aapl, msft, goog)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Ti le train/test tren du lieu tai ve")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, mac dinh lay toan bo")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, mac dinh lay toan bo")
    parser.add_argument("--force-download", action="store_true", help="Tai lai du lieu va ghi de train/test CSV")
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Neu tai du lieu that bai thi sinh du lieu synthetic de khong chan training",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    parameters_dir = base_dir / "parameters"
    results_dir = base_dir / "results"

    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data.csv"
    data_source = f"stooq:{args.symbol.lower()}"
    try:
        download_and_split_stock_data(
            train_path=train_path,
            test_path=test_path,
            symbol=args.symbol,
            split_ratio=args.split_ratio,
            start_date=args.start_date,
            end_date=args.end_date,
            force_download=args.force_download,
        )
    except Exception as exc:
        if args.allow_synthetic_fallback:
            print(f"Khong tai duoc du lieu thuc ({exc}). Sinh du lieu synthetic de tiep tuc.")
            ensure_demo_data(train_path, test_path, seed=args.seed, force_regen=True)
            data_source = "synthetic"
        else:
            raise

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
        pickle.dump(
            {
                "min_": scaler.min_,
                "max_": scaler.max_,
                "lookback": args.lookback,
                "data_source": data_source,
                "symbol": args.symbol,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "split_ratio": args.split_ratio,
            },
            f,
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(losses, results_dir / "training_loss.png")
    plot_gradients(raw_norms, clipped_norms, args.tau, results_dir / "gradient_norm_log.png")
    print("Da luu tham so vao", parameters_dir)
    print("Da luu bieu do vao", results_dir)


if __name__ == "__main__":
    main()
