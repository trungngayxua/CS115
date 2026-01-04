"""
Optimizer utilities: gradient clipping and parameter update.

Gradient clipping uses the formula from the slides:
    g <- g / max(1, ||g|| / tau)
so gradients with norm larger than tau are scaled down to avoid exploding values.
"""
from __future__ import annotations

import numpy as np


def gradients_l2_norm(gradients: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(np.sum(g**2) for g in gradients.values())))


def clip_gradients(gradients: dict[str, np.ndarray], tau: float) -> tuple[dict[str, np.ndarray], float, float]:
    """
    Scale gradients to keep their L2 norm under the threshold tau.

    Returns:
        clipped_gradients: dict with the same keys as gradients
        raw_norm: L2 norm before clipping
        clipped_norm: L2 norm after clipping
    """
    raw_norm = gradients_l2_norm(gradients)
    if raw_norm == 0:
        return {k: v.copy() for k, v in gradients.items()}, raw_norm, raw_norm

    scale = min(1.0, tau / raw_norm)
    clipped = {k: v * scale for k, v in gradients.items()}
    clipped_norm = gradients_l2_norm(clipped)
    return clipped, raw_norm, clipped_norm


def apply_gradients(params: dict[str, np.ndarray], gradients: dict[str, np.ndarray], lr: float) -> None:
    for name, grad in gradients.items():
        params[name] -= lr * grad

