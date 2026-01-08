import argparse
import pickle
from pathlib import Path

import numpy as np

from rnn_core import RNN
from utils import (
    MinMaxScaler,
    compute_metrics,
    create_sequences,
    load_stock_data,
    plot_predictions,
    save_metrics,
)


def load_model(parameters_dir: Path) -> RNN:
    weights = np.load(parameters_dir / "rnn_weights.npy", allow_pickle=True).item()
    biases = np.load(parameters_dir / "rnn_biases.npy", allow_pickle=True).item()
    model = RNN(
        input_size=weights["W_hx"].shape[0],
        hidden_size=weights["W_hx"].shape[1],
        output_size=weights["W_qh"].shape[1],
    )
    model.load_parameters(weights, biases)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate RNN on independent test set.")
    parser.add_argument("--lookback", type=int, default=None, help="Lookback window. Mac dinh lay tu scaler_params.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    parameters_dir = base_dir / "parameters"
    results_dir = base_dir / "results"

    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Khong tim thay data/train_data.csv va data/test_data.csv. Hay chay train.py de tai du lieu tu mang."
        )

    with open(parameters_dir / "scaler_params.pkl", "rb") as f:
        scaler_params = pickle.load(f)
    lookback = args.lookback or int(scaler_params.get("lookback", 20))
    data_source = scaler_params.get("data_source", "unknown")

    scaler = MinMaxScaler()
    scaler.load_params(scaler_params["min_"], scaler_params["max_"])

    df_test = load_stock_data(test_path)
    scaled_test = scaler.transform(df_test)
    X_test, y_test = create_sequences(scaled_test, lookback=lookback)
    if len(X_test) == 0:
        raise ValueError("Khong du du lieu test de tao sequence.")

    model = load_model(parameters_dir)

    preds_scaled = []
    for seq in X_test:
        outputs, _, _ = model.forward(seq, return_cache=False)
        preds_scaled.append(outputs[-1])
    preds_scaled = np.array(preds_scaled)

    preds = scaler.inverse_transform(preds_scaled).flatten()
    targets = scaler.inverse_transform(y_test).flatten()

    rmse, mae = compute_metrics(targets, preds)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(rmse, mae, results_dir / "metrics.txt")
    plot_predictions(targets, preds, results_dir / "test_prediction_chart.png")

    print(f"Test RMSE: {rmse:.6f} | MAE: {mae:.6f}")
    print(f"Da luu metrics va bieu do vao {results_dir} (data_source={data_source})")


if __name__ == "__main__":
    main()
