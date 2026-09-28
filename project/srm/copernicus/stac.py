"""Copernicus Data Space STAC catalog search (Sentinel-2 L2A)."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from srm.config import COPERNICUS_STAC_URL


def search_sentinel2_l2a(
    bbox: list[float],
    datetime_range: str = "2025-01-01T00:00:00Z/2026-12-31T23:59:59Z",
    cloud_cover_lt: float = 20.0,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Query public STAC API. Falls back to structured demo items if network fails.
    """
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "query": {"eo:cloud_cover": {"lt": cloud_cover_lt}},
        "limit": limit,
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(f"{COPERNICUS_STAC_URL}/search", json=body)
            if r.status_code == 200:
                data = r.json()
                return {
                    "source": COPERNICUS_STAC_URL,
                    "live": True,
                    "features": data.get("features", [])[:limit],
                    "count": len(data.get("features", [])),
                }
    except Exception as exc:
        return _fallback_search(bbox, str(exc))
    return _fallback_search(bbox, "non_200_response")


def _fallback_search(bbox: list[float], reason: str) -> dict:
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    return {
        "source": COPERNICUS_STAC_URL,
        "live": False,
        "fallback_reason": reason,
        "count": 2,
        "features": [
            {
                "id": "S2A_MSIL2A_DEMO_01",
                "bbox": bbox,
                "properties": {
                    "datetime": "2026-04-12T05:26:29Z",
                    "eo:cloud_cover": 4.2,
                    "gsd": 10,
                    "platform": "sentinel-2a",
                    "center": {"lat": lat, "lon": lon},
                },
            },
            {
                "id": "S2B_MSIL2A_DEMO_02",
                "bbox": bbox,
                "properties": {
                    "datetime": "2026-05-18T05:26:29Z",
                    "eo:cloud_cover": 1.8,
                    "gsd": 10,
                    "platform": "sentinel-2b",
                    "center": {"lat": lat, "lon": lon},
                },
            },
        ],
    }
