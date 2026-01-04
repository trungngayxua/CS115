import numpy as np


def clip_gradients(grads: dict, tau: float):
    """Scale toan bo gradient neu norm vuot nguong tau."""
    total_norm = 0.0
    for g in grads.values():
        total_norm += np.sum(g ** 2)
    total_norm = float(np.sqrt(total_norm))

    if total_norm > 0 and tau > 0 and total_norm > tau:
        scale = tau / total_norm
    else:
        scale = 1.0

    clipped = {k: v * scale for k, v in grads.items()}
    clipped_norm = float(np.sqrt(sum(np.sum(g ** 2) for g in clipped.values())))
    return clipped, total_norm, clipped_norm
