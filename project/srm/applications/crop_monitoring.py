import numpy as np
from srm.data.synthetic import generate_multispectral_patch
from srm.metrics.rs_metrics import ndvi


def analyze_crop_health(scene_id: str) -> dict:
    patch = generate_multispectral_patch(scene_id, 64)
    n = ndvi(patch[..., 3], patch[..., 2])
    stress = float(np.mean(n < 0.35))
    healthy = float(np.mean(n >= 0.55))
    return {
        "scene_id": scene_id,
        "application": "crop_monitoring",
        "mean_ndvi": round(float(np.mean(n)), 3),
        "healthy_fraction": round(healthy, 3),
        "stress_fraction": round(stress, 3),
        "parcel_boundary_confidence": round(0.82 + healthy * 0.15, 3),
        "recommendation": "IRRIGATION_AUDIT" if stress > 0.25 else "NORMAL_MONITORING",
    }
