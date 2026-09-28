import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

for _d in (DATA_DIR, EXPORT_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COPERNICUS_STAC_URL = os.getenv(
    "COPERNICUS_STAC_URL",
    "https://catalogue.dataspace.copernicus.eu/stac",
)
DEFAULT_INPUT_GSD_M = 10.0
TARGET_OUTPUT_GSD_M = 2.5
SUPPORTED_MODELS = [
    "Real-ESRGAN-RS",
    "SwinIR-Satellite",
    "EDSR-Multispectral",
    "Diffusion-SR-Sat",
    "Swin2SR-EO",
    "RCAN-Multiband",
]
DEVELOPER = "Rohith Kumar"
GITHUB = "https://github.com/rohithkumar505"
