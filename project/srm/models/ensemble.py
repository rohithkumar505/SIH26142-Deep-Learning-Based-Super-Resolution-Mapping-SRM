import numpy as np

from srm.models.factory import super_resolve


def ensemble_super_resolve(lr: np.ndarray, scale: int, models: list[str], weights: list[float] | None = None) -> np.ndarray:
    if not models:
        models = ["Real-ESRGAN-RS", "SwinIR-Satellite", "EDSR-Multispectral"]
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    acc = None
    for m, w in zip(models, weights):
        sr = super_resolve(lr, scale, m, 0.88)
        acc = sr * w if acc is None else acc + sr * w
    return np.clip(acc, 0, 1).astype(np.float32)
