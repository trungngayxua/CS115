"""
Traditional RNN core: manual forward pass and BPTT gradients.

The implementation follows the equations in CS115.tex:
    h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
    o_t = h_t @ W_qh + b_q
The backward method unrolls through time (BPTT) to accumulate gradients for
W_xh, W_hh, W_qh and the biases.
"""
from __future__ import annotations

import numpy as np

class TraditionalRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, rng: np.random.Generator):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rng = rng

        # Weight matrices and biases
        self.W_xh = rng.normal(scale=0.1, size=(input_size, hidden_size))
        self.W_hh = rng.normal(scale=0.1, size=(hidden_size, hidden_size))
        self.W_qh = rng.normal(scale=0.1, size=(hidden_size, output_size))
        self.b_h = np.zeros((1, hidden_size))
        self.b_q = np.zeros((1, output_size))

        # Cache used for BPTT
        self._cache = {}

    def forward(self, x_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the forward pass over a full sequence.

        Args:
            x_seq: Array of shape (T, input_size)
        Returns:
            outputs: (T, output_size)
            hidden_states: (T, hidden_size)
        """
        T = x_seq.shape[0]
        h_states = np.zeros((T + 1, self.hidden_size))
        outputs = np.zeros((T, self.output_size))

        for t in range(T):
            # h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
            h_states[t + 1] = np.tanh(
                x_seq[t : t + 1] @ self.W_xh + h_states[t : t + 1] @ self.W_hh + self.b_h
            )
            # o_t = h_t W_qh + b_q
            outputs[t : t + 1] = h_states[t + 1 : t + 2] @ self.W_qh + self.b_q

        # Store cache for BPTT
        self._cache = {"inputs": x_seq, "hidden_states": h_states, "outputs": outputs}
        return outputs, h_states[1:]

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
        """
        Backpropagation Through Time (BPTT) for a single sequence.

        Args:
            y_true: Ground-truth targets. If shape == outputs shape, loss is computed per time step.
                    Otherwise only the final time step is used (sequence-to-one setup).
            y_pred: Model outputs from the last forward pass (shape: T x output_size).
        Returns:
            Dictionary of gradients for each parameter.
        """
        if not self._cache:
            raise RuntimeError("Call forward() before backward().")

        outputs = y_pred
        inputs = self._cache["inputs"]
        h_states = self._cache["hidden_states"]
        T = outputs.shape[0]

        # Decide loss mode: per-step (sequence-to-sequence) or only final step (sequence-to-one)
        use_full_sequence = y_true.shape == outputs.shape
        y_true_arr = y_true
        if not use_full_sequence:
            y_true_arr = np.asarray(y_true).reshape(1, -1)

        grads = {
            "W_xh": np.zeros_like(self.W_xh),
            "W_hh": np.zeros_like(self.W_hh),
            "W_qh": np.zeros_like(self.W_qh),
            "b_h": np.zeros_like(self.b_h),
            "b_q": np.zeros_like(self.b_q),
        }

        dh_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(T)):
            if use_full_sequence:
                dy = (outputs[t] - y_true_arr[t]) / max(T, 1)
            else:
                if t != T - 1:
                    continue
                dy = outputs[t] - y_true_arr

            dy = dy.reshape(1, -1)
            h_t = h_states[t + 1 : t + 2]
            h_prev = h_states[t : t + 1]

            # Gradients for output layer
            grads["W_qh"] += h_t.T @ dy
            grads["b_q"] += dy

            # Propagate to hidden
            dh = dy @ self.W_qh.T + dh_next
            dh_raw = (1.0 - h_t**2) * dh  # derivative of tanh

            grads["W_xh"] += inputs[t : t + 1].T @ dh_raw
            grads["W_hh"] += h_prev.T @ dh_raw
            grads["b_h"] += dh_raw

            dh_next = dh_raw @ self.W_hh.T

        return grads

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "W_xh": self.W_xh,
            "W_hh": self.W_hh,
            "W_qh": self.W_qh,
            "b_h": self.b_h,
            "b_q": self.b_q,
        }

