"""Select the operating points of every method for every dataset (BLE system-energy profile).

The energy-profile run holds the main methods and the energy_baselines run the tuned two-tier cascade and greedy
search; their fronts are merged so that all methods are selected on the same profile, and the result is written
next to the main run, where scripts/make_paper_artifacts.py reads it.

    python scripts/operating_points_all.py --config configs/paper_artifacts.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from eids.utils.config import load_config, resolve_path

PROFILE = "system_ble"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_artifacts.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, d in cfg.datasets.items():
        fronts = [pd.read_csv(resolve_path(d.cost) / "fronts.csv")]
        if d.get("energy_baselines"):
            fronts.append(pd.read_csv(resolve_path(d.energy_baselines) / "fronts.csv"))
        merged = pd.concat(fronts, ignore_index=True)
        merged_path = resolve_path(d.cost) / f"fronts_all_methods_{PROFILE}.csv"
        merged[merged.profile == PROFILE].to_csv(merged_path, index=False)
        out = resolve_path(d.cost) / f"operating_points_profile{PROFILE}.csv"
        subprocess.run([sys.executable, "experiments/operating_points.py", "--fronts", str(merged_path),
                        "--out", str(out)], check=True)
        print(f"{key}: wrote {out}")


if __name__ == "__main__":
    main()
