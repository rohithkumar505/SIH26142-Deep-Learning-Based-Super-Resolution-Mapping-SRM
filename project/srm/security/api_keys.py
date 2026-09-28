import hashlib
import os
_VALID = {
    hashlib.sha256(b"ntro-demo-key-2026").hexdigest(): "NTRO_EVALUATOR",
    hashlib.sha256(b"rohith-sih26142-master").hexdigest(): "LEAD_DEVELOPER",
}


def verify_api_key(header_value: str | None) -> dict:
    if not header_value:
        return {"authenticated": False, "role": "PUBLIC", "tier": "read_only"}
    digest = hashlib.sha256(header_value.encode()).hexdigest()
    role = _VALID.get(digest)
    if role:
        return {"authenticated": True, "role": role, "tier": "full_platform"}
    return {"authenticated": False, "role": "INVALID", "tier": "blocked"}


def issue_demo_key() -> str:
    return os.getenv("SRM_DEMO_API_KEY", "ntro-demo-key-2026")
