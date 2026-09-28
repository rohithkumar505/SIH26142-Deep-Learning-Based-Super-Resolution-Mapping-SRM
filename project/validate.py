#!/usr/bin/env python3
"""CLI: Validate SR output against HR reference patch."""

import argparse

from srm.data.synthetic import downsample_lr, generate_multispectral_patch
from srm.metrics.rs_metrics import metric_bundle
from srm.models.factory import super_resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-id", default="SCENE_HIMALAYA_01")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--model", default="Real-ESRGAN-RS")
    args = p.parse_args()
    hr = generate_multispectral_patch(args.scene_id)
    lr = downsample_lr(hr, args.scale)
    sr = super_resolve(lr, args.scale, args.model, 0.9)
    h, w = min(hr.shape[0], sr.shape[0]), min(hr.shape[1], sr.shape[1])
    metrics = metric_bundle(hr[:h, :w], sr[:h, :w], args.scale)
    print(metrics)


if __name__ == "__main__":
    main()
