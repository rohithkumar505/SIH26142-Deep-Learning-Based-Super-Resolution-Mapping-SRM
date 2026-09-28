#!/usr/bin/env python3
"""
End-to-end REAL pipeline:
  1) Download Sentinel-2 previews from STAC
  2) Train PyTorch ESPCN weights
  3) Run inference + export GeoTIFF
"""

import argparse

from srm.copernicus.download import download_all_scenes
from srm.data.real_cache import load_scene_patch
from srm.data.synthetic import downsample_lr
from srm.export.geotiff_export import export_sr_array
from srm.metrics.rs_metrics import metric_bundle
from srm.models.espcn_net import infer_numpy
from srm.training.torch_trainer import train_torch_espcn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    if not args.skip_download:
        print("Step 1: Downloading real Sentinel-2 STAC previews...")
        print(download_all_scenes())

    print("Step 2: Training PyTorch ESPCN on real+cached patches...")
    train_result = train_torch_espcn(
        model_name="EDSR-Multispectral",
        scale=args.scale,
        epochs=args.epochs,
        use_real_data=True,
    )
    print(train_result["meta"])

    scene_id = "SCENE_HIMALAYA_01"
    hr = load_scene_patch(scene_id)
    if hr is None:
        raise SystemExit("No real patch — download failed")
    lr = downsample_lr(hr, args.scale)
    sr = infer_numpy(lr, "EDSR-Multispectral", args.scale)
    h, w = min(hr.shape[0], sr.shape[0]), min(hr.shape[1], sr.shape[1])
    metrics = metric_bundle(hr[:h, :w], sr[:h, :w], args.scale)
    print("Step 3: Metrics", metrics)

    export = export_sr_array(sr, scene_id, 10.0 / args.scale)
    print("Step 4: Export", export)


if __name__ == "__main__":
    main()
