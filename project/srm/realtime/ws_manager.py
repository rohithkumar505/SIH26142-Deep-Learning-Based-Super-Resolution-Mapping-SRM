from typing import Any

_SUBSCRIBERS: list[dict[str, Any]] = []


def publish_event(event_type: str, payload: dict) -> dict:
    message = {"type": event_type, "payload": payload}
    _SUBSCRIBERS.append(message)
    if len(_SUBSCRIBERS) > 200:
        _SUBSCRIBERS.pop(0)
    return message


def recent_events(limit: int = 20) -> list[dict]:
    return _SUBSCRIBERS[-limit:]
