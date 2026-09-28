"""Remote sensing validation metrics: PSNR, SSIM, SAM, ERGAS, NDVI conservation."""

from __future__ import annotations

import numpy as np


def psnr(hr: np.ndarray, sr: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((hr - sr) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        # Multichannel: average per band
        scores = []
        for c in range(hr.shape[-1]):
            scores.append(
                structural_similarity(
                    hr[..., c],
                    sr[..., c],
                    data_range=1.0,
                )
            )
        return float(np.mean(scores))
    except Exception:
        corr = np.corrcoef(hr.ravel(), sr.ravel())[0, 1]
        return float(np.clip((corr + 1) / 2, 0, 1))


def spectral_angle_mapper_degrees(hr: np.ndarray, sr: np.ndarray) -> float:
    h_flat = hr.reshape(-1, hr.shape[-1])
    s_flat = sr.reshape(-1, sr.shape[-1])
    dot = np.sum(h_flat * s_flat, axis=1)
    norm = np.linalg.norm(h_flat, axis=1) * np.linalg.norm(s_flat, axis=1) + 1e-8
    ang = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return float(np.degrees(np.mean(ang)))


def ergas(hr: np.ndarray, sr: np.ndarray, scale: int = 4) -> float:
    bands = hr.shape[-1]
    vals = []
    for c in range(bands):
        rmse = np.sqrt(np.mean((hr[..., c] - sr[..., c]) ** 2))
        mean = np.mean(hr[..., c]) + 1e-8
        vals.append((rmse / mean) ** 2)
    return float(100.0 / scale * np.sqrt(np.mean(vals)))


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-6)


def ndvi_conservation_score(hr: np.ndarray, sr: np.ndarray) -> float:
    ndvi_hr = ndvi(hr[..., 3], hr[..., 2])
    ndvi_sr = ndvi(sr[..., 3], sr[..., 2])
    diff = np.abs(ndvi_hr - ndvi_sr)
    return float(np.clip(1.0 - np.mean(diff), 0.0, 1.0))


def metric_bundle(hr: np.ndarray, sr: np.ndarray, scale: int) -> dict:
    return {
        "psnr_db": round(psnr(hr, sr), 2),
        "ssim": round(ssim(hr, sr), 3),
        "sam_degrees": round(spectral_angle_mapper_degrees(hr, sr), 2),
        "ergas": round(ergas(hr, sr, scale), 2),
        "ndvi_conservation_score": round(ndvi_conservation_score(hr, sr), 3),
    }
