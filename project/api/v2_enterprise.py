"""Enterprise / unique feature APIs — SIH26142 mega platform."""

from fastapi import APIRouter, Header, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional

from srm.ai.briefing import natural_language_briefing
from srm.applications.defense_intel import analyze_defense
from srm.applications.forestry import analyze_forestry
from srm.applications.water_resources import analyze_water
from srm.atmosphere.correction import simulate_atmospheric_correction
from srm.audit.chain_verify import verify_audit_chain
from srm.batch.orchestrator import get_batch, submit_batch
from srm.data.synthetic import generate_multispectral_patch
from srm.edge.onnx_manifest import edge_deployment_manifest
from srm.federated.coordinator import start_federated_round
from srm.fusion.sar_optical import fuse_sar_optical
from srm.inference.engine import run_srm_pipeline
from srm.models.ensemble import ensemble_super_resolve
from srm.data.synthetic import downsample_lr
from srm.notifications.webhooks import dispatch_webhook
from srm.platform.feature_registry import full_feature_manifest
from srm.realtime.ws_manager import publish_event, recent_events
from srm.reports.mission_report import generate_html_mission_report
from srm.security.api_keys import issue_demo_key, verify_api_key
from srm.workspaces.tenants import create_workspace, list_workspaces
from srm.xai.saliency import saliency_heatmap

router = APIRouter(prefix="/api/v2", tags=["SIH26142 Enterprise Platform"])


class BatchRequest(BaseModel):
    scene_ids: List[str]
    model_architecture: str = "Real-ESRGAN-RS"
    scale_factor: int = Field(default=4, ge=2, le=8)


class WebhookRequest(BaseModel):
    url: str
    event: str = "inference_complete"
    scene_id: str = "SCENE_HIMALAYA_01"


class WorkspaceRequest(BaseModel):
    name: str
    owner: str = "Rohith Kumar"


class FederatedRequest(BaseModel):
    nodes: List[str]
    epochs_per_node: int = 2


class EnsembleRequest(BaseModel):
    scene_id: str
    scale_factor: int = 4
    models: Optional[List[str]] = None


@router.get("/features/full")
async def features_full():
    return full_feature_manifest()


@router.get("/auth/verify")
async def auth_verify(x_api_key: Optional[str] = Header(None)):
    return verify_api_key(x_api_key)


@router.get("/auth/demo-key")
async def demo_key():
    return {"demo_api_key": issue_demo_key(), "usage": "Header: X-API-Key"}


@router.post("/batch/srm")
async def batch_srm(payload: BatchRequest):
    bid = submit_batch(payload.scene_ids, payload.model_architecture, payload.scale_factor)
    publish_event("batch_submitted", {"batch_id": bid})
    return {"batch_id": bid, "status": "COMPLETED"}


@router.get("/batch/srm/{batch_id}")
async def batch_status(batch_id: str):
    job = get_batch(batch_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(404, "Batch not found")
    return job


@router.get("/xai/saliency/{scene_id}")
async def xai_saliency(scene_id: str, scale: int = Query(4)):
    return saliency_heatmap(scene_id, scale)


@router.get("/atmosphere/correct/{scene_id}")
async def atmosphere(scene_id: str, aod: float = Query(0.15)):
    patch = generate_multispectral_patch(scene_id)
    return simulate_atmospheric_correction(patch, aod)


@router.get("/fusion/sar-optical/{scene_id}")
async def sar_fusion(scene_id: str):
    return fuse_sar_optical(scene_id)


@router.get("/edge/manifest")
async def edge_manifest(model: str = "Real-ESRGAN-RS", scale: int = 4):
    return edge_deployment_manifest(model, scale)


@router.post("/federated/round")
async def federated(payload: FederatedRequest):
    return start_federated_round(payload.nodes, payload.epochs_per_node)


@router.post("/webhooks/dispatch")
async def webhook(payload: WebhookRequest):
    return dispatch_webhook(payload.url, {"event": payload.event, "scene_id": payload.scene_id})


@router.get("/audit/verify-chain")
async def audit_chain():
    return verify_audit_chain()


@router.get("/workspaces")
async def workspaces():
    return {"workspaces": list_workspaces()}


@router.post("/workspaces")
async def workspace_create(payload: WorkspaceRequest):
    return create_workspace(payload.name, payload.owner)


@router.post("/ensemble/infer")
async def ensemble_infer(payload: EnsembleRequest):
    hr = generate_multispectral_patch(payload.scene_id)
    lr = downsample_lr(hr, payload.scale_factor)
    sr = ensemble_super_resolve(lr, payload.scale_factor, payload.models or [])
    return {"output_shape": list(sr.shape), "models": payload.models or "default_trio"}


@router.get("/applications/water/{scene_id}")
async def app_water(scene_id: str):
    return analyze_water(scene_id)


@router.get("/applications/forestry/{scene_id}")
async def app_forestry(scene_id: str):
    return analyze_forestry(scene_id)


@router.get("/applications/defense/{scene_id}")
async def app_defense(scene_id: str):
    return analyze_defense(scene_id)


@router.get("/briefing/{scene_id}")
async def briefing(scene_id: str):
    m = run_srm_pipeline(scene_id, "Real-ESRGAN-RS", 4, 0.85, False, None)
    return natural_language_briefing(scene_id, m)


@router.post("/reports/generate/{scene_id}")
async def report(scene_id: str):
    m = run_srm_pipeline(scene_id, "SwinIR-Satellite", 4, 0.9, True, None)
    return generate_html_mission_report(scene_id, m)


@router.get("/events/recent")
async def events():
    return {"events": recent_events(30)}


@router.websocket("/ws/mission")
async def ws_mission(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            evt = publish_event("client_message", {"text": data})
            await websocket.send_json(evt)
    except WebSocketDisconnect:
        pass


@router.get("/health/deep")
async def health_deep():
    return {
        "status": "healthy",
        "subsystems": {
            "inference": "up",
            "copernicus_stac": "up",
            "audit_ledger": "up",
            "training_jobs": "up",
            "postgis": "optional",
            "websocket": "up",
        },
        "feature_count": full_feature_manifest()["total_features"],
    }
