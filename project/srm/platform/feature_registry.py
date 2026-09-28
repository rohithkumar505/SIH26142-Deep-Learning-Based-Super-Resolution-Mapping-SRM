"""Central catalog of every platform capability (SIH26142 NTRO SRM)."""

from srm.config import SUPPORTED_MODELS

FEATURE_CATEGORIES = {
    "earth_observation_core": [
        "Sentinel-2 L2A 10m multispectral ingest (B02,B03,B04,B08)",
        "Copernicus Data Space STAC live search + resilient fallback",
        "BOA reflectance normalization & radiometric clip",
        "SCL-style cloud & shadow masking",
        "256px sliding-window tile orchestration",
        "4x/8x super-resolution to sub-4m GSD",
        "Bicubic baseline + spectral-preserving enhancer",
        "Optional PyTorch ESPCN GPU path",
    ],
    "deep_learning_zoo": [
        "Real-ESRGAN-RS (GAN family)",
        "SwinIR-Satellite (Transformer)",
        "EDSR-Multispectral (CNN)",
        "Diffusion-SR-Sat (generative)",
        "Swin2SR-EO",
        "RCAN-Multiband",
        "Model hot-swap at inference without restart",
        "Ensemble fusion inference",
    ],
    "metrics_and_validation": [
        "PSNR / SSIM / SAM / ERGAS",
        "NDVI & NDWI conservation scoring",
        "HR hold-out validation API",
        "Cross-scene benchmark leaderboard",
        "Per-band error budgets",
        "Scientific acceptance thresholds (NTRO)",
    ],
    "uncertainty_and_trust": [
        "Monte-Carlo test-time augmentation maps",
        "Hallucination risk tiers (LOW/MODERATE/HIGH)",
        "SHA-256 hash-chained audit ledger",
        "Explainability saliency (XAI) heatmaps",
        "Confidence-calibrated export metadata",
    ],
    "domain_applications": [
        "Precision agriculture & crop stress",
        "Urban footprint & corridor mapping",
        "Landslide & disaster damage proxy",
        "Water body & shoreline refinement",
        "Forestry canopy density",
        "Defense infrastructure change hints",
        "Multi-temporal change detection",
    ],
    "data_engineering": [
        "Live Copernicus / Planetary Computer STAC download to data/real/",
        "PyTorch ESPCN checkpoint train + load at inference",
        "End-to-end pipeline_e2e.py (download → train → export)",
        "Async training job queue",
        "Paired LR/HR synthetic curriculum",
        "GeoTIFF + NumPy export",
        "Batch tile pipeline API",
        "Custom multispectral upload inference",
        "PostGIS schema (docker-compose)",
        "Redis-ready cache layer stub",
    ],
    "mlops_and_ops": [
        "Docker multi-service compose",
        "Kubernetes deployment manifests",
        "GitHub Actions CI pytest",
        "Health / readiness / metrics probes",
        "WebSocket live inference progress",
        "API key auth (optional header)",
        "Rate limiting middleware",
        "Webhook notification dispatcher",
        "HTML/PDF mission reports",
        "ONNX edge export manifest",
        "Federated learning coordinator stub",
    ],
    "unique_differentiators": [
        "NTRO compliance matrix API for judges",
        "Mission Control multi-panel web console",
        "SAR-optical fusion preview (demo physics)",
        "Atmospheric correction simulation (6S-style)",
        "Spectral unmixing endmember hints",
        "Night-time VIIRS cross-sensor bridge",
        "Blockchain-style audit chain verification API",
        "Zero-trust API gateway pattern",
        "Multi-tenant project workspaces",
        "Natural language mission briefing generator",
    ],
}


def full_feature_manifest() -> dict:
    flat = []
    for cat, items in FEATURE_CATEGORIES.items():
        for item in items:
            flat.append({"category": cat, "name": item})
    return {
        "total_features": len(flat),
        "categories": len(FEATURE_CATEGORIES),
        "supported_models": SUPPORTED_MODELS,
        "features": flat,
        "by_category": FEATURE_CATEGORIES,
    }
