from pathlib import Path

from srm.config import CHECKPOINT_DIR


def edge_deployment_manifest(model_name: str, scale: int) -> dict:
    onnx_path = CHECKPOINT_DIR / f"{model_name}_x{scale}.onnx"
    return {
        "model": model_name,
        "scale": scale,
        "onnx_path": str(onnx_path),
        "onnx_exists": onnx_path.exists(),
        "target_devices": ["NVIDIA Jetson Orin", "Intel NUC + OpenVINO", "ARM64 edge node"],
        "recommended_batch": 1,
        "fp16_ready": True,
        "export_command": f"python -m srm.edge.export_onnx --model {model_name} --scale {scale}",
    }
