# Traditional RNN Stock Prediction (BPTT + Gradient Clipping)

Demo hiện thực kiến trúc **Traditional RNN** thủ công (không dùng LSTM/GRU) đúng theo công thức trong CS115.tex, kèm cơ chế **Gradient Clipping** để tránh Exploding Gradients.

## Cấu trúc
```
MSSV_Ten_StockPredictionRNN/
├── data/stock_price.csv          # Dữ liệu giá (sinh mẫu nếu thiếu)
├── parameters/                   # Trọng số & bias sau train (weights.npy, biases.npy)
├── results/                      # Biểu đồ: loss, gradient norm, prediction
└── src/
    ├── rnn_core.py               # Forward + BPTT thủ công (TraditionalRNN)
    ├── optimizer.py              # clip_gradients() + apply_gradients()
    ├── utils.py                  # Load data, MinMaxScaler, tạo cửa sổ chuỗi
    └── train.py                  # Vòng lặp huấn luyện, lưu kết quả
```

## Cách chạy nhanh
```bash
cd MSSV_Ten_StockPredictionRNN/src
python3 train.py --epochs 50 --lr 0.01 --hidden-size 16 --lookback 20 --tau 0.12
```
- Nếu `data/stock_price.csv` chưa có, script sẽ tự sinh dữ liệu mẫu.
- Kết quả lưu tại:
  - `parameters/weights.npy` (W_xh, W_hh, W_qh)
  - `parameters/biases.npy` (b_h, b_q)
  - `results/loss_curve_clipped.png` (loss vs epoch)
  - `results/gradients_norm.png` (L2 norm trước/sau clipping)
  - `results/prediction_chart.png` (Actual vs Predicted trên tập test)

## Liên hệ với slide CS115
- Forward: `h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)`, `o_t = h_t W_qh + b_q` (rnn_core.py).
- Backward: BPTT cộng dồn gradient qua thời gian để cập nhật `W_hx, W_hh, W_qh`, bias (rnn_core.py::backward).
- Gradient Clipping: `g <- g / max(1, ||g|| / tau)` để kìm exploding gradient (optimizer.py).
- Huấn luyện: Forward → tính MSE → backward → clip → update (train.py).
