"""Operating points: pick one deployment per method and operator scenario (on validation), report on test.

Works on the fronts.csv of a joint_search run (one study/tx) or a cost_sensitivity run (one profile).

    python experiments/operating_points.py --fronts results/runs/final_joint_ciciiot2025/<run>/fronts.csv \
        --filter study=tx_sweep tx=1000
    python experiments/operating_points.py --fronts results/runs/final_cost_ciciiot2025/<run>/fronts.csv \
        --filter profile=system_ble
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eids.cascade.selection import SCENARIOS, knee, reference_detector, select
from eids.utils.logging import get_logger

log = get_logger()
REPORT = ["macro_f1", "minority_recall", "fpr", "detection_rate", "mean_cost"]


def operating_points(fronts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, per_seed in fronts.groupby("seed"):
        ref = reference_detector(per_seed[per_seed["method"] == "single_model"])
        for method, front in per_seed.groupby("method"):
            picks = [(s.name, select(front, s, ref)) for s in SCENARIOS] + [("knee", knee(front))]
            for scenario, pick in picks:
                row = {"seed": seed, "method": method, "scenario": scenario, "feasible": pick is not None}
                if pick is not None:
                    row |= {m: pick[m] for m in REPORT} | {"deployment": pick["deployment"]}
                row["reference_cost"] = ref["mean_cost"]
                row["reference_macro_f1"] = ref["macro_f1"]
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fronts", required=True)
    parser.add_argument("--filter", nargs="*", default=[], help="column=value pairs, e.g. study=tx_sweep tx=1000")
    parser.add_argument("--out")
    args = parser.parse_args()
    fronts = pd.read_csv(args.fronts)
    for f in args.filter:
        col, val = f.split("=", 1)
        fronts = fronts[fronts[col].astype(str) == val]
    if "val_macro_f1" not in fronts.columns:
        raise SystemExit("fronts have no validation metrics; re-run the search with the current code")
    ops = operating_points(fronts)
    out = Path(args.out) if args.out else Path(args.fronts).with_name(
        "operating_points" + ("_" + "_".join(a.replace("=", "") for a in args.filter) if args.filter else "") + ".csv")
    ops.to_csv(out, index=False)
    summary = ops[ops.feasible].groupby(["scenario", "method"])[REPORT].mean()
    infeasible = ops[~ops.feasible].groupby(["scenario", "method"]).size()
    with pd.option_context("display.width", 220, "display.max_rows", None):
        log.info(f"wrote {out}\n" + summary.round(4).to_string())
        if len(infeasible):
            log.info("no feasible deployment (scenario, method -> seeds):\n" + infeasible.to_string())


if __name__ == "__main__":
    main()
