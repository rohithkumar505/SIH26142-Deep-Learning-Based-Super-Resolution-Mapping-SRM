#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || pip install torch
pip install planetary-computer rasterio 2>/dev/null || true
echo "Downloading real Sentinel-2 previews from STAC..."
python download_sentinel.py
echo "Training PyTorch ESPCN weights..."
python train.py --epochs 8
echo "Done. Run: python app.py"
