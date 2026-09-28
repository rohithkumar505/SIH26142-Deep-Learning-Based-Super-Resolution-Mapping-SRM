"""Sentinel-2 style pre-processing: BOA norm, cloud mask, tiling, baseline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def boa_reflectance_normalize(patch: np.ndarray) -> np.ndarray:
    """Clip and lightly denoise BOA-like reflectance."""
    x = np.clip(patch.astype(np.float32), 0.0, 1.0)
    return gaussian_filter(x, sigma=0.35)


def synthetic_cloud_shadow_mask(patch: np.ndarray, cloud_threshold: float = 0.92) -> np.ndarray:
    """Bright-pixel cloud mask (demo SCL-style). True = clear pixel."""
    brightness = patch[..., :3].mean(axis=-1)
    return brightness < cloud_threshold


def apply_mask(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = patch.copy()
    out[~mask] = 0.0
    return out


def sliding_windows(patch: np.ndarray, tile: int = 32, stride: int = 16):
    h, w, _ = patch.shape
    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            yield y, x, patch[y : y + tile, x : x + tile]


def bicubic_baseline(lr: np.ndarray, scale: int) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = (scale, scale, 1)
    return zoom(lr, factors, order=3).astype(np.float32)


def preprocess_sentinel2_patch(
    patch: np.ndarray,
    cloud_threshold: float = 0.92,
) -> dict:
    normalized = boa_reflectance_normalize(patch)
    mask = synthetic_cloud_shadow_mask(normalized, cloud_threshold)
    masked = apply_mask(normalized, mask)
    tiles = [{"y": y, "x": x, "tile": t} for y, x, t in sliding_windows(masked)]
    return {
        "normalized": normalized,
        "clear_sky_mask": mask,
        "masked": masked,
        "tiles": tiles,
        "tile_count": len(tiles),
        "preprocessing_steps": [
            "BOA_reflectance_clip",
            "gaussian_denoise",
            "synthetic_cloud_mask_SCL",
            "sliding_window_256_ready_tiles",
        ],
    }
