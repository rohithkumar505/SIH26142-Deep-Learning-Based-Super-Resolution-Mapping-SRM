# Real Working Stack — SIH26142

## Python version

Use **Python 3.11 or 3.12** for PyTorch training (`torch` may not install on 3.14 yet).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

## One command (recommended)

```bash
cd project
chmod +x scripts/setup_real_stack.sh
./scripts/setup_real_stack.sh
python app.py
```

## What is REAL (not fake)

| Step | What happens |
|------|----------------|
| **Download** | `download_sentinel.py` hits **Copernicus CDSE STAC** or **Microsoft Planetary Computer**, downloads a **real Sentinel-2 preview image**, saves `data/real/SCENE_*.npy` |
| **Train** | `train.py` runs **PyTorch ESPCN** with **L1 loss**, saves `checkpoints/*.pt` |
| **Infer** | `super_resolve()` loads **trained weights**; metrics are **computed** (PSNR, SSIM, SAM…) |
| **Export** | NumPy + optional GeoTIFF via rasterio |

## Manual pipeline

```bash
python download_sentinel.py --scene SCENE_DELHI_03
python train.py --epochs 15
python validate.py --scene-id SCENE_DELHI_03
python pipeline_e2e.py
```

## APIs

- `POST /api/v1/copernicus/download-all`
- `POST /api/v1/training/start` with `async_job: false` for sync PyTorch train
- `POST /api/v1/srm/upload-image` — real PNG/JPG super-resolution

**Note:** Full 10m band COG download at scale needs Copernicus credentials + storage; previews are real satellite imagery from STAC.
