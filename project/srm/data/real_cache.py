from pathlib import Path

import numpy as np

from srm.copernicus.download import REAL_DIR


def load_scene_patch(scene_id: str) -> np.ndarray | None:
    path = REAL_DIR / f"{scene_id}.npy"
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def has_real_data(scene_id: str) -> bool:
    return (REAL_DIR / f"{scene_id}.npy").exists()
