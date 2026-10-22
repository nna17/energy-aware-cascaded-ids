"""Paired statistics behind the significance statements of the paper (Results, RQ1).

Tests joint optimisation (NSGA-II, per-category thresholds) against each baseline on the same seeds:
  * transmission-cost sweep (runs `joint`): against the single detectors, the fixed two-tier design and random search,
    at every transmission cost, on both datasets (2 x 4 x 3 = 24 tests);
  * energy profiles (runs `cost`): against the best other method of each profile, i.e. the one with the highest mean
    hypervolume (2 x 10 = 20 tests).
For each comparison: per-seed hypervolume differences, a two-sided paired t-test over seeds, Holm correction over all
tests, and the paired effect size d_z = mean(diff) / sd(diff).

    PYTHONPATH=src python scripts/paired_stats.py [--config configs/paper_artifacts.yaml] [--out stats.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_artifacts as mpa  # noqa: E402

JOINT_SWEEP, JOINT_COST = "nsga2_category", "joint_nsga2"
SWEEP_BASELINES = ["single_model", "fixed_two_tier", "random_category"]


def pairs(joint: pd.Series, other: pd.Series) -> np.ndarray:
    both = pd.concat([joint.rename("j"), other.rename("o")], axis=1).dropna()
    return (both.j - both.o).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_artifacts.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = mpa.load_config(args.config)
    rows = []
    for key, d in cfg.datasets.items():
        s = mpa.run_csv(d.joint, "summary.csv")
        s = s[s.study == "tx_sweep"]
        for tx, g in s.groupby("tx"):
            hv = g.pivot(index="seed", columns="method", values="hv_log_cost")
            for b in SWEEP_BASELINES:
                rows.append(dict(dataset=key, setting=f"tx={tx}", baseline=b, diff=pairs(hv[JOINT_SWEEP], hv[b])))
        c = mpa.run_csv(d.cost, "summary.csv")
        for prof, g in c.groupby("profile"):
            hv = g.pivot(index="seed", columns="method", values="hv_log_cost")
            others = hv.drop(columns=[JOINT_COST]).mean().idxmax()
            rows.append(dict(dataset=key, setting=prof, baseline=others, diff=pairs(hv[JOINT_COST], hv[others])))
    for r in rows:
        dlt = r.pop("diff")
        r.update(n=len(dlt), mean_diff=dlt.mean(), min_diff=dlt.min(), wins=int((dlt > 0).sum()),
                 p=stats.ttest_rel(dlt, np.zeros_like(dlt)).pvalue if len(dlt) > 1 else np.nan,
                 d_z=dlt.mean() / dlt.std(ddof=1) if len(dlt) > 1 and dlt.std(ddof=1) > 0 else np.inf)
    out = pd.DataFrame(rows)
    # Holm step-down correction over all tests
    order = out.p.sort_values().index
    m, running = len(out), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out.at[i, "p"]))
        out.at[i, "p_holm"] = running
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\ntests: {m}; seed-level comparisons: {int(out.n.sum())}; joint ahead: {int(out.wins.sum())}; "
          f"significant after Holm (5%): {int((out.p_holm < 0.05).sum())}; "
          f"d_z range: {out.d_z.min():.3g} to {out.d_z.max():.3g}, median {out.d_z.median():.3g}")
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
