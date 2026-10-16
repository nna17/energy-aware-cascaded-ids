"""Variance of the joint search alone: several search seeds on one fixed detector pool (pool seed 0).

Compared with the variance across pool seeds (which retrains every detector and reruns the search), this separates
the optimiser's randomness from the detectors' training randomness.

    python experiments/search_seed_variance.py --config configs/final/joint_ciciiot2025.yaml --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse

import pandas as pd

from eids.cascade.algorithms import run_search
from eids.cascade.pool import load_pool
from eids.cascade.search import CascadeEvaluator, Device, front_on, hypervolume
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir
from eids.utils.logging import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tx", type=float, nargs="+", default=[0, 1000])
    args = parser.parse_args()
    exp = load_config(args.config)
    run_dir = make_run_dir(f"search_seed_variance_{exp.name.removeprefix('final_joint_')}")
    get_logger(log_file=run_dir / "log.txt")
    pool = load_pool(resolve_path(exp.pool_dir.format(seed=0)))
    devices = [Device(**d) for d in exp.devices]
    s = exp.search
    rows = []
    for tx in args.tx:
        ev = CascadeEvaluator(pool, devices, [tx, tx])
        for seed in args.seeds:
            res = run_search(ev, s.algorithm, "category", ordered=True, budget=s.budget, seed=seed,
                             pop_size=s.pop_size, seed_singles=s.get("seed_singles", False))
            F, _ = front_on(ev, [res.encoding.decode(res.X[i]) for i in res.front_indices(s.budget)], "test")
            rows.append({"pool_seed": 0, "search_seed": seed, "tx": tx, "hv_log_cost": hypervolume(F)})
            log.info(f"tx {tx} search seed {seed}: HV={rows[-1]['hv_log_cost']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "results.csv", index=False)
    log.info("\n" + df.groupby("tx").hv_log_cost.agg(["mean", "std", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
