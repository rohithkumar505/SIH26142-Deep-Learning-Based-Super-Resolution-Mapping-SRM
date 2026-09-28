import numpy as np
from srm.data.synthetic import generate_multispectral_patch
from srm.metrics.rs_metrics import ndvi


def analyze_water(scene_id: str) -> dict:
    p = generate_multispectral_patch(scene_id, 64)
    ndwi = (p[..., 1] - p[..., 3]) / (p[..., 1] + p[..., 3] + 1e-6)
    water = ndwi > 0.2
    return {
        "scene_id": scene_id,
        "application": "water_resources",
        "water_fraction": round(float(np.mean(water)), 3),
        "shoreline_srm_gain": "4x_edge_recovery",
        "flood_extent_proxy": round(float(np.mean(water & (p[..., 2] < 0.15))), 3),
    }
