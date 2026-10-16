"""Build the processed cache for one dataset.

    python scripts/prepare_data.py --config configs/data/edge_iiotset.yaml
    python scripts/prepare_data.py --config configs/data/ciciot2023.yaml --root D:/datasets/CICIoT2023
"""
from __future__ import annotations

import argparse

from eids.data.pipeline import prepare_dataset
from eids.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", help="override the raw data folder")
    parser.add_argument("--label-level", choices=["binary", "category", "fine"])
    args = parser.parse_args()

    overrides = {}
    if args.root:
        overrides["root"] = args.root
    if args.label_level:
        overrides["label_level"] = args.label_level
    prepare_dataset(load_config(args.config, **overrides))


if __name__ == "__main__":
    main()
