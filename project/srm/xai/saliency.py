import numpy as np

from srm.data.synthetic import generate_multispectral_patch
from srm.models.factory import super_resolve
from srm.data.synthetic import downsample_lr


def saliency_heatmap(scene_id: str, scale: int = 4) -> dict:
    hr = generate_multispectral_patch(scene_id, 48)
    lr = downsample_lr(hr, scale)
    sr = super_resolve(lr, scale, "Real-ESRGAN-RS", 0.88)
    h, w = min(hr.shape[0], sr.shape[0]), min(hr.shape[1], sr.shape[1])
    delta = np.abs(hr[:h, :w] - sr[:h, :w]).mean(axis=-1)
    delta = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
    return {
        "scene_id": scene_id,
        "heatmap_shape": list(delta.shape),
        "mean_activation": round(float(delta.mean()), 4),
        "top_pixels_fraction": round(float(np.mean(delta > 0.75)), 4),
        "method": "gradient_free_abs_diff_saliency",
    }
