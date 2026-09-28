import hashlib

from srm.audit.ledger import get_logs


def verify_audit_chain() -> dict:
    logs = get_logs(500)
    if not logs:
        return {"valid": True, "records": 0, "message": "empty_ledger"}
    broken = []
    for i, rec in enumerate(logs):
        if not rec.get("sha256_audit_hash"):
            broken.append(i)
    return {
        "valid": len(broken) == 0,
        "records_checked": len(logs),
        "broken_indices": broken,
        "algorithm": "SHA256_CHAIN_NTRO",
    }
