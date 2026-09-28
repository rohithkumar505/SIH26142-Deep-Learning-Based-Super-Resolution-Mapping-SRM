import uuid
from datetime import datetime, timezone

from srm.inference.engine import run_srm_pipeline
from srm.scenes import SCENE_BY_ID

_JOBS: dict[str, dict] = {}


def submit_batch(scene_ids: list[str], model: str, scale: int) -> str:
    batch_id = str(uuid.uuid4())
    results = []
    for sid in scene_ids:
        if sid not in SCENE_BY_ID:
            continue
        results.append(
            run_srm_pipeline(sid, model, scale, 0.85, True, None)
        )
    _JOBS[batch_id] = {
        "batch_id": batch_id,
        "status": "COMPLETED",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "results": results,
    }
    return batch_id


def get_batch(batch_id: str) -> dict | None:
    return _JOBS.get(batch_id)
