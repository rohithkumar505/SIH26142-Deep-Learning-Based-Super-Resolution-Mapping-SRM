import numpy as np
from srm.data.synthetic import generate_multispectral_patch
from srm.metrics.rs_metrics import ndvi


def analyze_forestry(scene_id: str) -> dict:
    p = generate_multispectral_patch(scene_id, 64)
    n = ndvi(p[..., 3], p[..., 2])
    canopy = n > 0.45
    return {
        "scene_id": scene_id,
        "application": "forestry",
        "canopy_density": round(float(np.mean(canopy)), 3),
        "mean_ndvi": round(float(np.mean(n)), 3),
        "deforestation_alert": bool(float(np.mean(n < 0.25)) > 0.4),
    }
