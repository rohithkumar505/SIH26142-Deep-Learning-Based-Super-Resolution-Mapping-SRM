"""Export SR stack as GeoTIFF (requires rasterio) or NumPy fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from srm.config import EXPORT_DIR


def export_sr_array(sr: np.ndarray, scene_id: str, gsd_m: float) -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = EXPORT_DIR / f"{scene_id}_sr_{gsd_m}m.npy"
    np.save(npy_path, sr)

    tif_path = EXPORT_DIR / f"{scene_id}_sr_{gsd_m}m.tif"
    try:
        import rasterio
        from rasterio.transform import from_origin

        h, w, bands = sr.shape
        transform = from_origin(0, h * gsd_m, gsd_m, gsd_m)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=bands,
            dtype="float32",
            transform=transform,
            crs="EPSG:4326",
        ) as dst:
            for i in range(bands):
                dst.write(sr[..., i], i + 1)
        format_used = "GeoTIFF"
    except Exception:
        tif_path = None
        format_used = "NumPy_only_rasterio_optional"

    return {
        "npy_export": str(npy_path),
        "geotiff_export": str(tif_path) if tif_path else None,
        "format": format_used,
        "bands": sr.shape[-1],
        "gsd_m": gsd_m,
    }
