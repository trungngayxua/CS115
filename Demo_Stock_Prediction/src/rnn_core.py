import numpy as np


class RNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_hx = rng.normal(scale=0.1, size=(input_size, hidden_size))
        self.W_hh = rng.normal(scale=0.1, size=(hidden_size, hidden_size))
        self.W_qh = rng.normal(scale=0.1, size=(hidden_size, output_size))
        self.b_h = np.zeros(hidden_size)
        self.b_q = np.zeros(output_size)

    def forward(self, x_seq, h0=None, return_cache: bool = False):
        # Forward cho mot chuoi (T, input_size). Tra outputs (T, output_size) va h_T.
        x_seq = np.asarray(x_seq, dtype=float)
        if x_seq.ndim == 1:
            x_seq = x_seq.reshape(-1, self.input_size)
        T = x_seq.shape[0]
        h_prev = np.zeros(self.hidden_size) if h0 is None else h0

        hs = [h_prev]
        pre_acts = []
        outputs = []

        for t in range(T):
            a_t = x_seq[t].dot(self.W_hx) + h_prev.dot(self.W_hh) + self.b_h
            h_t = np.tanh(a_t)
            o_t = h_t.dot(self.W_qh) + self.b_q

            pre_acts.append(a_t)
            outputs.append(o_t)
            hs.append(h_t)
            h_prev = h_t

        out_arr = np.vstack(outputs)
        cache = None
        if return_cache:
            cache = {
                "x_seq": x_seq,
                "hs": hs,
                "pre_acts": pre_acts,
                "outputs": out_arr,
            }
        return out_arr, h_prev, cache

    def backward(self, cache: dict, target: np.ndarray):
        # BPTT cho loss 0.5*(o_T - y)^2. Chi dung output cuoi chuoi.
        x_seq = cache["x_seq"]
        hs = cache["hs"]
        pre_acts = cache["pre_acts"]
        outputs = cache["outputs"]

        T = x_seq.shape[0]
        target = np.asarray(target).reshape(-1)
        o_T = outputs[-1].reshape(-1)

        grads = {
            "W_hx": np.zeros_like(self.W_hx),
            "W_hh": np.zeros_like(self.W_hh),
            "W_qh": np.zeros_like(self.W_qh),
            "b_h": np.zeros_like(self.b_h),
            "b_q": np.zeros_like(self.b_q),
        }

        dL_do = o_T - target  # derivative of 0.5*(o - y)^2
        grads["W_qh"] += np.outer(hs[-1], dL_do)
        grads["b_q"] += dL_do

        d_next = dL_do.dot(self.W_qh.T)

        for t in reversed(range(T)):
            h_t = hs[t + 1]
            h_prev = hs[t]
            a_t = pre_acts[t]
            x_t = x_seq[t]

            dh = d_next
            da = dh * (1.0 - h_t**2)  # tanh'

            grads["W_hx"] += np.outer(x_t, da)
            grads["W_hh"] += np.outer(h_prev, da)
            grads["b_h"] += da

            d_next = da.dot(self.W_hh.T)

        loss = 0.5 * np.mean((o_T - target) ** 2)
        return loss, grads

    def apply_grads(self, grads: dict, lr: float) -> None:
        self.W_hx -= lr * grads["W_hx"]
        self.W_hh -= lr * grads["W_hh"]
        self.W_qh -= lr * grads["W_qh"]
        self.b_h -= lr * grads["b_h"]
        self.b_q -= lr * grads["b_q"]

    def get_parameters(self):
        weights = {"W_hx": self.W_hx, "W_hh": self.W_hh, "W_qh": self.W_qh}
        biases = {"b_h": self.b_h, "b_q": self.b_q}
        return weights, biases

    def load_parameters(self, weights: dict, biases: dict) -> None:
        self.W_hx = weights["W_hx"]
        self.W_hh = weights["W_hh"]
        self.W_qh = weights["W_qh"]
        self.b_h = biases["b_h"]
        self.b_q = biases["b_q"]
