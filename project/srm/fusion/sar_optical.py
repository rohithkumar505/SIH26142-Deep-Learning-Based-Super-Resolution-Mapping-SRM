import numpy as np

from srm.data.synthetic import generate_multispectral_patch


def fuse_sar_optical(scene_id: str) -> dict:
    optical = generate_multispectral_patch(scene_id, 32)
    sar = optical[..., 3] * 0.6 + optical[..., 0] * 0.4 + np.random.default_rng(7).normal(0, 0.02, optical.shape[:2])
    sar = np.clip(sar, 0, 1)
    fused = 0.65 * optical.mean(axis=-1) + 0.35 * sar
    return {
        "scene_id": scene_id,
        "fusion_method": "weighted_optical_sar_coherence",
        "coherence_score": round(float(np.corrcoef(optical[..., 3].ravel(), sar.ravel())[0, 1]), 3),
        "fused_mean": round(float(fused.mean()), 4),
    }
