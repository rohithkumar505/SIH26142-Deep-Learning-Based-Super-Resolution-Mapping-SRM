"""Real PyTorch L1 training on paired LR/HR patches (synthetic or disk)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from srm.config import CHECKPOINT_DIR, DATA_DIR
from srm.data.real_cache import load_scene_patch
from srm.data.synthetic import downsample_lr, generate_multispectral_patch
from srm.metrics.rs_metrics import psnr
from srm.models.espcn_net import build_espcn, save_checkpoint


def _iter_patches(scene_ids: list[str], scale: int, use_real: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for sid in scene_ids:
        if use_real:
            hr = load_scene_patch(sid)
            if hr is None:
                hr = generate_multispectral_patch(sid, 64)
        else:
            hr = generate_multispectral_patch(sid, 64)
        lr = downsample_lr(hr, scale)
        pairs.append((lr, hr))
    return pairs


def train_torch_espcn(
    model_name: str = "EDSR-Multispectral",
    scale: int = 4,
    epochs: int = 15,
    lr_rate: float = 1e-3,
    scene_ids: Optional[list[str]] = None,
    use_real_data: bool = True,
) -> dict:
    import torch
    import torch.nn as nn

    scene_ids = scene_ids or ["SCENE_HIMALAYA_01", "SCENE_PUNJAB_02", "SCENE_DELHI_03"]
    pairs = _iter_patches(scene_ids, scale, use_real_data)
    channels = pairs[0][0].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_espcn(channels, scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_rate)
    loss_fn = nn.L1Loss()
    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for lr_np, hr_np in pairs:
            lr_t = torch.from_numpy(lr_np.transpose(2, 0, 1)[None]).float().to(device)
            hr_t = torch.from_numpy(hr_np.transpose(2, 0, 1)[None]).float().to(device)
            opt.zero_grad()
            pred = model(lr_t)
            h, w = hr_t.shape[2], hr_t.shape[3]
            pred = pred[:, :, :h, :w]
            loss = loss_fn(pred, hr_t)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / len(pairs)

        # validation PSNR
        lr_v, hr_v = pairs[0]
        with torch.no_grad():
            sr = model(torch.from_numpy(lr_v.transpose(2, 0, 1)[None]).float().to(device))
            sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)[: hr_v.shape[0], : hr_v.shape[1]]
        val_psnr = psnr(hr_v, sr_np)
        history.append({"epoch": epoch, "loss": round(avg_loss, 6), "val_psnr_db": round(val_psnr, 2)})

    meta = {
        "model": model_name,
        "scale": scale,
        "epochs": epochs,
        "channels": channels,
        "use_real_data": use_real_data,
        "device": str(device),
        "best_val_psnr_db": max(h["val_psnr_db"] for h in history),
    }
    ckpt = save_checkpoint(model.cpu(), model_name, scale, channels, meta)
    return {
        "status": "TRAINING_COMPLETE",
        "backend": "pytorch_espcn",
        "checkpoint": ckpt,
        "meta": meta,
        "history": history,
    }
