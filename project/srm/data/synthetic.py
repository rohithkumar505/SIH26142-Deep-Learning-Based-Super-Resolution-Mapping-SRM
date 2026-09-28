"""Synthetic Sentinel-2-like multispectral patches for demo / unit inference."""

import hashlib
import numpy as np


def _rng_for_scene(scene_id: str) -> np.random.Generator:
    seed = int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def generate_multispectral_patch(scene_id: str, size: int = 64) -> np.ndarray:
    """
    Returns float32 array (H, W, 4) mimicking B02,B03,B04,B08 reflectance in [0,1].
    """
    rng = _rng_for_scene(scene_id)
    h, w = size, size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    terrain = np.sin(xx / 9.0) * np.cos(yy / 7.0)
    urban = (xx > w * 0.55).astype(np.float32) * 0.15
    vegetation = np.exp(-((xx - w * 0.25) ** 2 + (yy - h * 0.35) ** 2) / (w * 8) ** 2)

    blue = 0.08 + 0.12 * terrain + rng.normal(0, 0.01, (h, w))
    green = 0.10 + 0.18 * vegetation + 0.05 * terrain + rng.normal(0, 0.01, (h, w))
    red = 0.09 + 0.14 * vegetation + 0.08 * urban + rng.normal(0, 0.01, (h, w))
    nir = 0.15 + 0.35 * vegetation + 0.04 * urban + rng.normal(0, 0.012, (h, w))

    patch = np.stack([blue, green, red, nir], axis=-1).astype(np.float32)
    return np.clip(patch, 0.0, 1.0)


def downsample_lr(hr: np.ndarray, scale: int) -> np.ndarray:
    h, w, c = hr.shape
    lh, lw = h // scale, w // scale
    view = hr[: lh * scale, : lw * scale].reshape(lh, scale, lw, scale, c)
    return view.mean(axis=(1, 3)).astype(np.float32)
