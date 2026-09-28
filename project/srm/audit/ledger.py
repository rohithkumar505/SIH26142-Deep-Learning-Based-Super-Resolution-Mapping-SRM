AUDIT_LOGS: list[dict] = []
LAST_HASH: str | None = None


def append_audit(record: dict) -> None:
    global LAST_HASH
    LAST_HASH = record.get("sha256_audit_hash")
    AUDIT_LOGS.append(record)
    if len(AUDIT_LOGS) > 500:
        AUDIT_LOGS.pop(0)


def get_logs(limit: int = 20) -> list[dict]:
    return AUDIT_LOGS[-limit:]


def audit_count() -> int:
    return len(AUDIT_LOGS)
