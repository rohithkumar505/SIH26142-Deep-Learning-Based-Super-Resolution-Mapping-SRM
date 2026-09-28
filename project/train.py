#!/usr/bin/env python3
"""CLI: Train SRM model on paired synthetic Sentinel-2 patches."""

import argparse

from srm.training.trainer import train_srm_model


def main():
    p = argparse.ArgumentParser(description="SIH26142 SRM training")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--model", default="EDSR-Multispectral")
    p.add_argument("--no-real-data", action="store_true")
    args = p.parse_args()
    result = train_srm_model(
        epochs=args.epochs,
        scale=args.scale,
        model_name=args.model,
        use_real_data=not args.no_real_data,
    )
    print(result)


if __name__ == "__main__":
    main()
