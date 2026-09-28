"""End-to-end SRM inference with real metrics (not random placeholders)."""

from __future__ import annotations

import datetime
import hashlib
from typing import Callable, Optional

import numpy as np

from srm.data.real_cache import has_real_data, load_scene_patch
from srm.data.synthetic import downsample_lr, generate_multispectral_patch
from srm.metrics.rs_metrics import metric_bundle
from srm.models.factory import _has_trained_weights, super_resolve
from srm.preprocessing.pipeline import preprocess_sentinel2_patch
from srm.uncertainty.estimator import monte_carlo_uncertainty_map


def run_srm_pipeline(
    scene_id: str,
    model_architecture: str,
    scale_factor: int,
    spectral_consistency_weight: float,
    enable_uncertainty: bool,
    prev_audit_hash: Optional[str] = None,
) -> dict:
    hr_ref = load_scene_patch(scene_id)
    if hr_ref is None:
        hr_ref = generate_multispectral_patch(scene_id, size=64)
    data_source = "copernicus_real_npy" if has_real_data(scene_id) else "synthetic_curriculum"
    pre = preprocess_sentinel2_patch(hr_ref)
    lr = downsample_lr(pre["masked"], scale_factor)

    def _model_fn(x):
        return super_resolve(
            x,
            scale_factor,
            model_architecture,
            spectral_consistency_weight,
        )

    sr = _model_fn(lr)
    # Align sizes for metrics
    h, w = min(hr_ref.shape[0], sr.shape[0]), min(hr_ref.shape[1], sr.shape[1])
    hr_crop = hr_ref[:h, :w]
    sr_crop = sr[:h, :w]

    metrics = metric_bundle(hr_crop, sr_crop, scale_factor)
    unc_map = None
    uncertainty_index = 0.05
    if enable_uncertainty:
        unc_map, uncertainty_index = monte_carlo_uncertainty_map(
            lr, scale_factor, _model_fn, n_samples=5
        )

    hallucination_level = "LOW_RISK" if uncertainty_index < 0.08 else "MODERATE_INFERRED"
    if uncertainty_index > 0.15:
        hallucination_level = "HIGH_REVIEW_REQUIRED"

    in_res = 10.0
    out_res = round(in_res / scale_factor, 2)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    exec_id = hashlib.sha256(f"{scene_id}{ts}".encode()).hexdigest()[:12].upper()
    exec_id = f"SRM-{exec_id}"

    chain = f"{prev_audit_hash or 'GENESIS'}:{exec_id}:{metrics['psnr_db']}:{ts}"
    sha_hash = hashlib.sha256(chain.encode()).hexdigest()[:16]

    ckpt_name = (
        "EDSR-Multispectral"
        if model_architecture in ("Real-ESRGAN-RS", "SwinIR-Satellite", "Diffusion-SR-Sat")
        else model_architecture
    )
    weights_used = _has_trained_weights(ckpt_name, scale_factor, hr_ref.shape[-1])

    return {
        "execution_id": exec_id,
        "scene_id": scene_id,
        "model_architecture": model_architecture,
        "input_resolution_m": in_res,
        "output_resolution_m": out_res,
        "scale_factor": scale_factor,
        "psnr_db": metrics["psnr_db"],
        "ssim": metrics["ssim"],
        "ergas": metrics["ergas"],
        "sam_degrees": metrics["sam_degrees"],
        "ndvi_conservation_score": metrics["ndvi_conservation_score"],
        "uncertainty_index": round(uncertainty_index, 3),
        "hallucination_risk": hallucination_level,
        "status": "SUPER_RESOLUTION_SYNTHESIS_COMPLETE",
        "sha256_audit_hash": sha_hash,
        "processed_at": ts,
        "developer": "Rohith Kumar (github.com/rohithkumar505)",
        "preprocessing": {
            "tile_count": pre["tile_count"],
            "steps": pre["preprocessing_steps"],
            "clear_sky_ratio": float(pre["clear_sky_mask"].mean()),
        },
        "uncertainty_map_shape": list(unc_map.shape) if unc_map is not None else None,
        "pipeline_mode": "REAL_DATA_AWARE_ENGINE_V5",
        "data_source": data_source,
        "trained_weights_used": weights_used,
    }
