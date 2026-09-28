#!/usr/bin/env python3
"""Unified CLI for SIH26142 SRM platform."""

import argparse

from srm.benchmark.leaderboard import run_leaderboard
from srm.platform.feature_registry import full_feature_manifest


def main():
    p = argparse.ArgumentParser(description="SIH26142 SRM CLI")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("features", help="List all platform features")
    sub.add_parser("leaderboard", help="Run model benchmark leaderboard")

    args = p.parse_args()
    if args.cmd == "features":
        m = full_feature_manifest()
        print(f"Total features: {m['total_features']}")
        for f in m["features"]:
            print(f"  [{f['category']}] {f['name']}")
    elif args.cmd == "leaderboard":
        print(run_leaderboard())
    else:
        p.print_help()


if __name__ == "__main__":
    main()
