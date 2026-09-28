import numpy as np
from srm.data.synthetic import generate_multispectral_patch


def analyze_disaster(scene_id: str) -> dict:
    patch = generate_multispectral_patch(scene_id, 64)
    slope = np.gradient(patch[..., 2].mean(axis=0))[0]
    debris = float(np.mean(np.abs(slope) > 0.08))
    return {
        "scene_id": scene_id,
        "application": "disaster_assessment",
        "landslide_debris_proxy": round(debris, 3),
        "damage_localization_confidence": round(0.75 + debris * 0.2, 3),
        "uncertainty_flag": bool(debris > 0.35),
        "response_priority": "HIGH" if debris > 0.35 else "MEDIUM",
    }
