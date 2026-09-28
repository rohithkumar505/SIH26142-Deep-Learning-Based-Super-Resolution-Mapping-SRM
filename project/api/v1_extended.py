"""Extended SIH26142 API: Copernicus, training, validation, applications, platform."""

from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from srm.applications.crop_monitoring import analyze_crop_health
from srm.applications.disaster_assessment import analyze_disaster
from srm.applications.urban_analysis import analyze_urban
from srm.config import DEVELOPER, GITHUB, SUPPORTED_MODELS, TARGET_OUTPUT_GSD_M
from srm.copernicus.stac import search_sentinel2_l2a
from srm.data.real_cache import has_real_data, load_scene_patch
from srm.data.synthetic import downsample_lr, generate_multispectral_patch
from srm.copernicus.download import download_all_scenes, download_scene
from srm.export.geotiff_export import export_sr_array
from srm.inference.engine import run_srm_pipeline
from srm.metrics.rs_metrics import metric_bundle
from srm.models.factory import model_capabilities, super_resolve
from srm.preprocessing.pipeline import preprocess_sentinel2_patch
from srm.scenes import SAMPLE_SCENES, SCENE_BY_ID
from srm.training.jobs import get_job, start_training_job
from srm.training.trainer import train_srm_model

router = APIRouter(prefix="/api/v1", tags=["SIH26142 Extended Platform"])


class PreprocessRequest(BaseModel):
    scene_id: str = "SCENE_HIMALAYA_01"
    cloud_threshold: float = 0.92


class ValidateRequest(BaseModel):
    scene_id: str
    scale_factor: int = Field(default=4, ge=2, le=8)
    model_architecture: str = "Real-ESRGAN-RS"


class TrainingRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=100)
    scale_factor: int = Field(default=4, ge=2, le=8)
    model_architecture: str = "EDSR-Multispectral"
    async_job: bool = True


class ChangeDetectionRequest(BaseModel):
    scene_id_a: str
    scene_id_b: str
    scale_factor: int = 4


@router.get("/compliance/matrix")
async def compliance_matrix():
    """Maps NTRO SIH26142 expected solution bullets to implemented modules."""
    return {
        "problem_id": "SIH26142",
        "developer": DEVELOPER,
        "items": [
            {"requirement": "10m Sentinel-2 input", "status": "IMPLEMENTED", "module": "copernicus/stac + scenes"},
            {"requirement": "Pre-processing pipeline", "status": "IMPLEMENTED", "module": "preprocessing/pipeline.py"},
            {"requirement": "Generative/CNN/Transformer SR models", "status": "IMPLEMENTED", "module": "models/factory.py"},
            {"requirement": "Output <4m products", "status": "IMPLEMENTED", "module": "inference/engine.py (4x default)"},
            {"requirement": "Spectral consistency", "status": "IMPLEMENTED", "module": "metrics SAM + NDVI"},
            {"requirement": "Uncertainty quantification", "status": "IMPLEMENTED", "module": "uncertainty/estimator.py"},
            {"requirement": "HR reference validation", "status": "IMPLEMENTED", "module": "POST /srm/validate"},
            {"requirement": "Model training with paired data", "status": "IMPLEMENTED", "module": "training/trainer.py"},
            {"requirement": "Crop monitoring", "status": "IMPLEMENTED", "module": "applications/crop_monitoring.py"},
            {"requirement": "Urban analysis", "status": "IMPLEMENTED", "module": "applications/urban_analysis.py"},
            {"requirement": "Disaster assessment", "status": "IMPLEMENTED", "module": "applications/disaster_assessment.py"},
            {"requirement": "Copernicus dataset link", "status": "IMPLEMENTED", "module": "GET /copernicus/search"},
            {"requirement": "GeoTIFF export", "status": "PARTIAL", "module": "export/geotiff_export.py (rasterio optional)"},
            {"requirement": "Production-scale HR weights", "status": "ROADMAP", "module": "train on Copernicus+Planet pairs"},
        ],
    }


@router.get("/platform/features")
async def platform_features():
    from srm.platform.feature_registry import full_feature_manifest

    manifest = full_feature_manifest()
    return {
        "unique_capabilities": [f["name"] for f in manifest["features"]],
        "total_features": manifest["total_features"],
        "categories": manifest["categories"],
        "by_category": manifest["by_category"],
        "supported_models": SUPPORTED_MODELS,
        "target_output_gsd_m": TARGET_OUTPUT_GSD_M,
        "github": GITHUB,
        "enterprise_api": "/api/v2/features/full",
    }


@router.get("/benchmark/leaderboard")
async def benchmark_leaderboard(scale_factor: int = Query(4)):
    from srm.benchmark.leaderboard import run_leaderboard

    return run_leaderboard(scale_factor)


@router.get("/copernicus/search")
async def copernicus_search(
    min_lon: float = Query(75.5),
    min_lat: float = Query(28.4),
    max_lon: float = Query(77.5),
    max_lat: float = Query(30.9),
    cloud_cover_lt: float = Query(15.0),
    limit: int = Query(5, ge=1, le=50),
):
    return search_sentinel2_l2a(
        bbox=[min_lon, min_lat, max_lon, max_lat],
        cloud_cover_lt=cloud_cover_lt,
        limit=limit,
    )


@router.post("/srm/preprocess")
async def preprocess(payload: PreprocessRequest):
    if payload.scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    patch = load_scene_patch(payload.scene_id) or generate_multispectral_patch(payload.scene_id)
    result = preprocess_sentinel2_patch(patch, payload.cloud_threshold)
    return {
        "scene_id": payload.scene_id,
        "tile_count": result["tile_count"],
        "clear_sky_ratio": float(result["clear_sky_mask"].mean()),
        "steps": result["preprocessing_steps"],
    }


@router.post("/srm/validate")
async def validate_against_hr_reference(payload: ValidateRequest):
    if payload.scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    hr = load_scene_patch(payload.scene_id) or generate_multispectral_patch(payload.scene_id)
    lr = downsample_lr(hr, payload.scale_factor)
    sr = super_resolve(lr, payload.scale_factor, payload.model_architecture, 0.9)
    h, w = min(hr.shape[0], sr.shape[0]), min(hr.shape[1], sr.shape[1])
    metrics = metric_bundle(hr[:h, :w], sr[:h, :w], payload.scale_factor)
    return {
        "scene_id": payload.scene_id,
        "reference": "synthetic_hr_holdout_patch",
        "validation_passed": metrics["psnr_db"] >= 28 and metrics["sam_degrees"] <= 8,
        "metrics": metrics,
    }


@router.post("/training/start")
async def training_start(payload: TrainingRequest):
    if payload.async_job:
        job_id = start_training_job(payload.epochs, payload.scale_factor, payload.model_architecture)
        return {"job_id": job_id, "status": "RUNNING"}
    return train_srm_model(
        payload.epochs,
        payload.scale_factor,
        model_name=payload.model_architecture,
        use_real_data=True,
    )


@router.post("/copernicus/download/{scene_id}")
async def copernicus_download_scene(scene_id: str, force: bool = Query(False)):
    if scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    try:
        return download_scene(scene_id, force=force)
    except Exception as exc:
        raise HTTPException(502, f"Download failed: {exc}")


@router.post("/copernicus/download-all")
async def copernicus_download_all(force: bool = Query(False)):
    return {"results": download_all_scenes(force=force)}


@router.get("/copernicus/real-data-status")
async def real_data_status():
    return {
        scene_id: has_real_data(scene_id)
        for scene_id in SCENE_BY_ID
    }


@router.post("/srm/upload-image")
async def upload_image(file: UploadFile = File(...), scale_factor: int = Query(4)):
    from io import BytesIO
    from PIL import Image

    raw = await file.read()
    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize((128, 128))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    nir = np.clip(0.5 * rgb[..., 1] + 0.5 * rgb[..., 0], 0, 1)
    patch = np.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0], nir], axis=-1)
    lr = downsample_lr(patch, scale_factor)
    sr = super_resolve(lr, scale_factor, "EDSR-Multispectral", 0.88)
    return {
        "input_shape": list(lr.shape),
        "output_shape": list(sr.shape),
        "status": "PROCESSED",
        "format": file.filename,
    }


@router.get("/training/jobs/{job_id}")
async def training_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/models/zoo")
async def models_zoo():
    return model_capabilities()


@router.get("/applications/crop/{scene_id}")
async def app_crop(scene_id: str):
    if scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    return analyze_crop_health(scene_id)


@router.get("/applications/urban/{scene_id}")
async def app_urban(scene_id: str):
    if scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    return analyze_urban(scene_id)


@router.get("/applications/disaster/{scene_id}")
async def app_disaster(scene_id: str):
    if scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    return analyze_disaster(scene_id)


@router.post("/srm/change-detection")
async def change_detection(payload: ChangeDetectionRequest):
    a = generate_multispectral_patch(payload.scene_id_a)
    b = generate_multispectral_patch(payload.scene_id_b)
    diff = np.abs(a - b).mean(axis=-1)
    return {
        "mean_change_index": round(float(diff.mean()), 4),
        "hotspot_fraction": round(float(np.mean(diff > 0.08)), 4),
        "srm_recommended": True,
    }


@router.post("/srm/export")
async def export_product(scene_id: str = Query("SCENE_HIMALAYA_01"), scale_factor: int = Query(4)):
    if scene_id not in SCENE_BY_ID:
        raise HTTPException(404, "Unknown scene_id")
    hr = load_scene_patch(scene_id) or generate_multispectral_patch(scene_id)
    lr = downsample_lr(hr, scale_factor)
    sr = super_resolve(lr, scale_factor, "Real-ESRGAN-RS", 0.88)
    gsd = 10.0 / scale_factor
    return export_sr_array(sr, scene_id, gsd)


@router.post("/srm/upload-npy")
async def upload_npy(file: UploadFile = File(...), scale_factor: int = Query(4)):
    raw = await file.read()
    try:
        buf = np.frombuffer(raw, dtype=np.float32)
        side = int(np.sqrt(buf.size / 4))
        patch = buf.reshape(side, side, 4)
    except Exception:
        raise HTTPException(400, "Expected float32 .npy-like 4-band square payload")
    lr = patch if patch.max() <= 1.5 else patch / 255.0
    sr = super_resolve(lr.astype(np.float32), scale_factor, "EDSR-Multispectral", 0.85)
    return {"input_shape": list(lr.shape), "output_shape": list(sr.shape), "status": "PROCESSED"}
