#!/usr/bin/env python3
"""Download real Sentinel-2 preview patches from Copernicus / Planetary Computer STAC."""

import argparse

from srm.copernicus.download import download_all_scenes, download_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="SCENE_HIMALAYA_01 etc; omit for all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.scene:
        print(download_scene(args.scene, force=args.force))
    else:
        print(download_all_scenes(force=args.force))


if __name__ == "__main__":
    main()
