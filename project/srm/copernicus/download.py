"""Download real Sentinel-2 preview assets from STAC (Copernicus CDSE + Planetary Computer fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from srm.config import COPERNICUS_STAC_URL, DATA_DIR

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
REAL_DIR = DATA_DIR / "real"
REAL_DIR.mkdir(parents=True, exist_ok=True)

SCENE_BBOX = {
    "SCENE_HIMALAYA_01": [78.9, 30.6, 79.2, 30.85],
    "SCENE_PUNJAB_02": [75.7, 30.8, 76.0, 31.0],
    "SCENE_DELHI_03": [77.1, 28.5, 77.4, 28.7],
}


def _stac_search(url: str, bbox: list[float], limit: int = 3) -> list[dict]:
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": "2024-01-01T00:00:00Z/2026-12-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 25}},
        "limit": limit,
    }
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        r = client.post(f"{url}/search", json=body)
        if r.status_code != 200:
            return []
        return r.json().get("features", [])


def _pick_asset_href(feature: dict) -> str | None:
    assets = feature.get("assets") or {}
    for key in ("rendered_preview", "thumbnail", "visual", "overview"):
        if key in assets and assets[key].get("href"):
            return assets[key]["href"]
    for a in assets.values():
        href = a.get("href", "")
        if href.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            return href
    return None


def _sign_planetary_href(href: str) -> str:
    try:
        import planetary_computer

        return planetary_computer.sign(href)
    except Exception:
        with httpx.Client(timeout=20.0) as client:
            r = client.get("https://planetarycomputer.microsoft.com/api/sas/v1/sign", params={"href": href})
            if r.status_code == 200:
                return r.json().get("href", href)
    return href


def _download_image_bytes(href: str) -> bytes:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        r = client.get(href)
        r.raise_for_status()
        return r.content


def _bytes_to_multispectral_patch(raw: bytes, target_size: int = 128) -> np.ndarray:
    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    # Pseudo NIR from green+red (heuristic for demo pipeline — replace with real B08 when COG bands wired)
    nir = np.clip(0.45 * rgb[..., 1] + 0.55 * rgb[..., 0], 0, 1)
    patch = np.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0], nir], axis=-1).astype(np.float32)
    return patch


def download_scene(scene_id: str, force: bool = False) -> dict[str, Any]:
    meta_path = REAL_DIR / f"{scene_id}.json"
    npy_path = REAL_DIR / f"{scene_id}.npy"
    if npy_path.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    bbox = SCENE_BBOX.get(scene_id, [77.0, 28.6, 77.3, 28.8])
    features = _stac_search(COPERNICUS_STAC_URL, bbox, limit=5)
    provider = "copernicus_cdse"
    if not features:
        features = _stac_search(PC_STAC, bbox, limit=5)
        provider = "planetary_computer"

    if not features:
        raise RuntimeError("No STAC features found; check network or bbox")

    feature = features[0]
    href = _pick_asset_href(feature)
    if not href:
        raise RuntimeError("STAC item has no downloadable preview asset")

    if provider == "planetary_computer":
        href = _sign_planetary_href(href)

    raw = _download_image_bytes(href)
    patch = _bytes_to_multispectral_patch(raw)
    np.save(npy_path, patch)

    record = {
        "scene_id": scene_id,
        "provider": provider,
        "stac_id": feature.get("id"),
        "bbox": bbox,
        "asset_href": href,
        "patch_shape": list(patch.shape),
        "npy_path": str(npy_path),
        "live_download": True,
    }
    meta_path.write_text(json.dumps(record, indent=2))
    return record


def download_all_scenes(force: bool = False) -> list[dict]:
    results = []
    for sid in SCENE_BBOX:
        try:
            results.append(download_scene(sid, force=force))
        except Exception as exc:
            results.append({"scene_id": sid, "error": str(exc), "live_download": False})
    return results
