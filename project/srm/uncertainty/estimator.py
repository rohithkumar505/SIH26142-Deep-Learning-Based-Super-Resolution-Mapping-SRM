"""Pixel-wise uncertainty via test-time augmentation ensemble."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def monte_carlo_uncertainty_map(
    lr: np.ndarray,
    scale: int,
    model_fn,
    n_samples: int = 6,
) -> tuple[np.ndarray, float]:
    preds = []
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        noisy = lr + rng.normal(0, 0.008 * (1 + i * 0.1), lr.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)
        sr = model_fn(noisy)
        preds.append(sr)
    stack = np.stack(preds, axis=0)
    var = np.var(stack, axis=0).mean(axis=-1)
    uncertainty_index = float(np.clip(np.mean(var) * 12.0, 0.01, 0.25))
    return var.astype(np.float32), uncertainty_index
