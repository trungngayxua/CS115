# Stock Price Prediction with Traditional RNN (Hand-coded BPTT)

Thư mục này chứa demo RNN truyền thống tự code forward/backward, tách biệt train/test. Mặc định dùng dữ liệu synthetic (có thể thay bằng dữ liệu thực của bạn).

## Cấu trúc
- `data/train_data.csv`: dùng để fit scaler + huấn luyện (BPTT + clipping).
- `data/test_data.csv`: độc lập để chấm điểm, **không** dùng để tính gradient.
- `parameters/`: `rnn_weights.npy`, `rnn_biases.npy`, `scaler_params.pkl` (tạo sau khi train).
- `results/`: biểu đồ `training_loss.png`, `gradient_norm_log.png`, `test_prediction_chart.png`, file `metrics.txt`.
- `src/`: mã nguồn `rnn_core.py`, `optimizer.py`, `utils.py`, `train.py`, `evaluate.py`.

## Cách chạy
1) Huấn luyện trên train set, lưu model và scaler:
```bash
python src/train.py --epochs 300 --lookback 20 --hidden-size 16 --lr 0.01 --tau 5.0
```
2) Kiểm thử trên test set (chỉ forward, không update trọng số):
```bash
python src/evaluate.py --lookback 20
```

## Ghi chú
- Có sẵn hàm sinh dữ liệu synthetic nếu thiếu file CSV; thay `data/train_data.csv` và `data/test_data.csv` bằng dữ liệu thật của bạn (cột `Date,Close`).
- Scaler trên test **bắt buộc** dùng min/max đã fit từ train (được lưu trong `scaler_params.pkl`).
- Hình ảnh kết quả sẽ được LaTeX chèn từ `results/` (đã khai báo trong `graphicspath` của slide).
