"""Per-class weights for imbalanced training, normalised so the sample-weighted mean is 1."""
from __future__ import annotations

import numpy as np
import torch


def class_weights(counts: np.ndarray, method: str = "effective_number", beta: float = 0.999) -> torch.Tensor:
    counts = np.maximum(counts.astype(np.float64), 1.0)
    if method == "none":
        w = np.ones_like(counts)
    elif method == "inverse_frequency":
        w = 1.0 / counts
    elif method == "effective_number":  # Cui et al., 2019
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown class weighting {method!r}")
    w = w * counts.sum() / (w * counts).sum()
    return torch.tensor(w, dtype=torch.float32)
