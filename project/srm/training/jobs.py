import threading
import uuid
from datetime import datetime, timezone

from srm.training.trainer import train_srm_model

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def start_training_job(epochs: int, scale: int, model_name: str) -> str:
    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "epochs": epochs,
            "scale": scale,
            "model": model_name,
        }

    def _run():
        try:
            result = train_srm_model(
                epochs=epochs, scale=scale, model_name=model_name, use_real_data=True
            )
            with _LOCK:
                _JOBS[job_id].update(status="COMPLETED", result=result)
        except Exception as exc:
            with _LOCK:
                _JOBS[job_id].update(status="FAILED", error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _LOCK:
        return _JOBS.get(job_id)
