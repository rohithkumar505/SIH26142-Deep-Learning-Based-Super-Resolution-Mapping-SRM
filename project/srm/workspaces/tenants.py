import uuid

_WORKSPACES: dict[str, dict] = {
    "ntro-default": {
        "workspace_id": "ntro-default",
        "name": "NTRO Primary Evaluation",
        "owner": "Rohith Kumar",
        "quota_tiles_per_day": 10000,
    }
}


def create_workspace(name: str, owner: str) -> dict:
    wid = str(uuid.uuid4())[:8]
    ws = {"workspace_id": wid, "name": name, "owner": owner, "quota_tiles_per_day": 5000}
    _WORKSPACES[wid] = ws
    return ws


def list_workspaces() -> list[dict]:
    return list(_WORKSPACES.values())
