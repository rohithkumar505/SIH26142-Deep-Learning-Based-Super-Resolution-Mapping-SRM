"""Federated learning round coordinator (demo protocol for multi-node training)."""


def start_federated_round(nodes: list[str], epochs_per_node: int = 2) -> dict:
    return {
        "protocol": "FedAvg-SRM-NTRO",
        "nodes": nodes,
        "epochs_per_node": epochs_per_node,
        "aggregation": "weighted_by_clear_sky_pixels",
        "privacy": "gradient_clipping + secure_aggregation_stub",
        "status": "ROUND_SCHEDULED",
    }
