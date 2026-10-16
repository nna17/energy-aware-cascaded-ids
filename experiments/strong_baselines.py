"""Stronger baselines than the fixed designs: does joint optimisation with NSGA-II beat simple, cost-aware designs?

  tuned_two_tier  every device-eligible detector followed by the server detector with the best validation
                  macro-F1, with one global escalation threshold from a grid (0.05 ... 1.0): a hand-designed
                  cascade whose detector and threshold are tuned on validation
  greedy          weighted-Chebyshev hill climbing in the same encoding as NSGA-II (per-category thresholds,
                  cost-ordered detectors), one weight vector and one random start per run; the runs share the
                  NSGA-II evaluation budget
  nsga2_category  the joint search, for reference

Every method is scored like the others: validation-non-dominated deployments, hypervolume on test.

    python experiments/strong_baselines.py --config configs/final/joint_ciciiot2025.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from eids.cascade.algorithms import OrderedEncoding, run_search
from eids.cascade.pool import load_pool
from eids.cascade.search import CascadeEvaluator, Deployment, Device, front_on, hypervolume
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()
GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
N_WEIGHTS = 20


def tuned_two_tier(ev: CascadeEvaluator) -> list[Deployment]:
    server = max(ev.eligible[2], key=lambda m: ev.evaluate(Deployment([m], [2], []), "val")[1]["macro_f1"])
    n = ev.pool.n_classes
    return [Deployment([d, server], [0, 2], [np.full(n, t)]) for d in ev.eligible[0] for t in GRID]


def greedy(ev: CascadeEvaluator, budget: int, seed: int) -> list[Deployment]:
    """Hill climbing on scalarised validation objectives; all evaluated deployments form the archive."""
    enc = OrderedEncoding(ev, "category", ordered=True)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(4), size=N_WEIGHTS)
    per_run = budget // N_WEIGHTS
    steps = np.r_[[1 / len(enc.options[0]), 1 / (len(enc.options[1]) + 1), 1 / (len(enc.options[2]) + 1)],
                  np.full(enc.n_var - 3, 0.1)]
    archive, cache = [], {}

    def objectives(x: np.ndarray) -> np.ndarray:
        key = tuple(np.round(x, 6))
        if key not in cache:
            dep = enc.decode(x)
            cache[key] = ev.evaluate(dep, "val")[0]
            archive.append(dep)
        return cache[key]

    for w in weights:
        x = rng.random(enc.n_var)  # multi-start: each weight vector climbs from its own random deployment
        best = np.max(w * objectives(x))
        used = 1
        improved = True
        while improved and used < per_run:
            improved = False
            for i in rng.permutation(enc.n_var):
                for sign in (1, -1):
                    y = x.copy()
                    y[i] = np.clip(y[i] + sign * steps[i], 0, 1 - 1e-9)
                    used += 1
                    v = np.max(w * objectives(y))
                    if v < best - 1e-12:
                        x, best, improved = y, v, True
                        break
                if improved or used >= per_run:
                    break
    log.info(f"  greedy: {len(archive)} distinct deployments evaluated")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    exp = load_config(args.config)
    run_dir = make_run_dir(f"strong_baselines_{exp.name.removeprefix('final_joint_')}")
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")
    devices = [Device(**d) for d in exp.devices]
    s = exp.search
    rows = []
    for seed in exp.seeds:
        pool = load_pool(resolve_path(exp.pool_dir.format(seed=seed)))
        for tx in exp.tx_sweep:
            ev = CascadeEvaluator(pool, devices, [tx, tx])
            res = run_search(ev, s.algorithm, "category", ordered=True, budget=s.budget, seed=seed,
                             pop_size=s.pop_size, seed_singles=s.get("seed_singles", False))
            methods = {"tuned_two_tier": tuned_two_tier(ev), "greedy": greedy(ev, s.budget, seed),
                       "nsga2_category": [res.encoding.decode(res.X[i]) for i in res.front_indices(s.budget)]}
            for name, deps in methods.items():
                F, front = front_on(ev, deps, "test")
                rows.append({"seed": seed, "tx": tx, "method": name, "hv_log_cost": hypervolume(F),
                             "front_size": len(front), "evaluated": len(deps)})
                log.info(f"seed {seed} tx {tx} {name:<15} HV={rows[-1]['hv_log_cost']:.4f}")
            pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    log.info("\n" + pd.DataFrame(rows).groupby(["tx", "method"]).hv_log_cost.mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
