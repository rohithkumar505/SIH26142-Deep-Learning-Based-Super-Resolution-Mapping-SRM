"""
Automated Pytest Suite for SIH26142 Satellite Super Resolution Mapping (SRM)
Architect & Developer: Rohith Kumar (https://github.com/rohithkumar505)
Sponsoring Organization: National Technical Research Organisation (NTRO)
"""

import os
import sys
import importlib.util
import pytest
from fastapi.testclient import TestClient

app_path = os.path.join(os.path.dirname(__file__), "app.py")
spec = importlib.util.spec_from_file_location("app_sih26142", app_path)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app

client = TestClient(app)

def test_root_metadata_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["problem_id"] == "SIH26142"
    assert data["developer"] == "Rohith Kumar"
    assert "https://github.com/rohithkumar505" in data["github_profile"]
    assert data["sponsoring_organization"] == "National Technical Research Organisation (NTRO)"
    assert data["theme"] == "Space Technology"
    assert data["status"] == "OPERATIONAL"

def test_copernicus_scenes_endpoint():
    response = client.get("/api/v1/srm/scenes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_scenes"] >= 3
    assert "Copernicus" in data["source"]
    assert any("Himalayan" in s["title"] for s in data["scenes"])

def test_super_resolution_inference():
    payload = {
        "scene_id": "SCENE_HIMALAYA_01",
        "model_architecture": "Real-ESRGAN-RS",
        "scale_factor": 4,
        "spectral_consistency_weight": 0.9,
        "enable_uncertainty_quantification": True
    }
    response = client.post("/api/v1/srm/super-resolve", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["scale_factor"] == 4
    assert data["input_resolution_m"] == 10.0
    assert data["output_resolution_m"] == 2.5
    assert data["psnr_db"] >= 30.0
    assert 0.85 <= data["ssim"] <= 1.0
    assert data["sam_degrees"] > 0
    assert "sha256_audit_hash" in data
    assert "Rohith Kumar" in data["developer"]

def test_srm_stats_endpoint():
    response = client.get("/api/v1/srm/telemetry/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["lead_developer"] == "Rohith Kumar"
    assert "Real-ESRGAN-RS" in data["active_models"]
    assert data["copernicus_hub_status"] == "ONLINE_HEALTHY"

def test_audit_logs():
    response = client.get("/api/v1/audit/logs")
    assert response.status_code == 200
    data = response.json()
    assert data["lead_architect"] == "Rohith Kumar"
    assert "records" in data
    assert isinstance(data["records"], list)
