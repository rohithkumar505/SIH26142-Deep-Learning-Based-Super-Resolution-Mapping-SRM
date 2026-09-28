"""Trainable ESPCN super-resolution network (multispectral)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from srm.config import CHECKPOINT_DIR


def build_espcn(channels: int, scale: int):
    import torch.nn as nn

    class ESPCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            self.net = nn.Sequential(
                nn.Conv2d(channels, 64, 5, padding=2),
                nn.Tanh(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.Tanh(),
                nn.Conv2d(32, channels * (scale ** 2), 3, padding=1),
                nn.PixelShuffle(scale),
            )

        def forward(self, x):
            import torch

            return torch.clamp(self.net(x), 0.0, 1.0)

    return ESPCN()


def checkpoint_path(model_name: str, scale: int, channels: int = 4) -> Path:
    safe = model_name.replace("/", "_").replace(" ", "_")
    return CHECKPOINT_DIR / f"{safe}_c{channels}_x{scale}.pt"


def save_checkpoint(model, model_name: str, scale: int, channels: int, meta: dict) -> str:
    import torch

    path = checkpoint_path(model_name, scale, channels)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return str(path)


def load_checkpoint(model_name: str, scale: int, channels: int = 4):
    import torch

    path = checkpoint_path(model_name, scale, channels)
    if not path.exists():
        return None, None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_espcn(channels, scale)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})


def infer_numpy(lr: np.ndarray, model_name: str, scale: int) -> np.ndarray:
    import torch

    channels = lr.shape[-1]
    model, meta = load_checkpoint(model_name, scale, channels)
    if model is None:
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path(model_name, scale, channels)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.from_numpy(lr.transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        y = model(x)[0].cpu().numpy().transpose(1, 2, 0)
    return y.astype(np.float32)
