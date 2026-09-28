import numpy as np


def simulate_atmospheric_correction(patch: np.ndarray, aod: float = 0.15) -> dict:
    """Demo 6S-style dark object subtraction + path radiance removal."""
    path = aod * 0.05
    corrected = np.clip(patch.astype(np.float32) - path, 0.0, 1.0)
    return {
        "aod_input": aod,
        "path_radiance_removed": round(path, 4),
        "mean_reflectance_before": round(float(patch.mean()), 4),
        "mean_reflectance_after": round(float(corrected.mean()), 4),
        "corrected_shape": list(corrected.shape),
    }
