import httpx

from srm.config import DEVELOPER


def dispatch_webhook(url: str, payload: dict) -> dict:
    body = {"source": "SIH26142_SRM", "developer": DEVELOPER, **payload}
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.post(url, json=body)
            return {"delivered": True, "status_code": r.status_code}
    except Exception as exc:
        return {"delivered": False, "error": str(exc), "queued_for_retry": True}
