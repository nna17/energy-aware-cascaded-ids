"""Search-algorithm study: NSGA-II (pool-order vs cost-ordered encoding), NSGA-III, SMS-EMOA and random
search, with equal evaluation budgets, full archives, hypervolume at several budgets, several seeds.

    python experiments/search_study.py --config configs/experiments/search_study.yaml
    python experiments/search_study.py --only smsemoa --seeds 0 --tx 0 --budget 1000   # timing check
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from eids.cascade.algorithms import run_search
from eids.cascade.pool import load_pool
from eids.cascade.search import CascadeEvaluator, Device, hypervolume
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()


def _test_fronts(ev: CascadeEvaluator, res, indices: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Test objectives (log-cost and linear-cost versions) of archive members, cached by index."""
    F_log, F_lin = [], []
    for i in indices:
        if i not in cache:
            f, extra = ev.evaluate(res.encoding.decode(res.X[i]), "test")
            lin = f.copy()
            lin[3] = extra["mean_cost"] / ev.cost_ref
            cache[i] = (f, lin)
        F_log.append(cache[i][0])
        F_lin.append(cache[i][1])
    return np.array(F_log), np.array(F_lin)


def run_job(job: dict) -> list[dict]:
    pool = load_pool(resolve_path(job["pool_dir"]))
    ev = CascadeEvaluator(pool, [Device(**d) for d in job["devices"]], [job["tx"], job["tx"]])
    res = run_search(ev, job["algorithm"], job["granularity"], job["ordered"], job["budget"],
                     job["seed"], job["pop_size"])
    variant = f"{job['algorithm']}_{'ordered' if job['ordered'] else 'unordered'}"
    base = {"variant": variant, "seed": job["seed"], "tx": job["tx"], "seconds": res.seconds}
    rows, cache = [], {}
    for k in job["checkpoints"]:
        if k > len(res.F):
            continue
        F_log, F_lin = _test_fronts(ev, res, res.front_indices(k), cache)
        rows.append({**base, "selection": "archive", "evaluations": k, "hv_log_cost": hypervolume(F_log),
                     "hv_linear_cost": hypervolume(F_lin), "front_size": len(F_log)})
    if len(res.population):  # what the first experiments reported: the final population only
        val_F = np.array([ev.evaluate(res.encoding.decode(x), "val")[0] for x in res.population])
        keep = NonDominatedSorting().do(val_F, only_non_dominated_front=True)
        tests = [ev.evaluate(res.encoding.decode(res.population[i]), "test") for i in keep]
        F_log = np.array([t[0] for t in tests])
        F_lin = F_log.copy()
        F_lin[:, 3] = [t[1]["mean_cost"] / ev.cost_ref for t in tests]
        rows.append({**base, "selection": "final_population", "evaluations": len(res.F),
                     "hv_log_cost": hypervolume(F_log), "hv_linear_cost": hypervolume(F_lin),
                     "front_size": len(F_log)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/search_study.yaml")
    parser.add_argument("--only", nargs="*", help="run only these algorithms")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--tx", nargs="*", type=float)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    exp = load_config(args.config)
    budget = args.budget or exp.budget
    run_dir = make_run_dir(exp.name)
    get_logger(log_file=run_dir / "log.txt")
    save_json({**exp, "overrides": vars(args)}, run_dir / "config.json")

    jobs = [
        {"pool_dir": exp.pool_dir, "devices": [dict(d) for d in exp.devices], "granularity": exp.granularity,
         "budget": budget, "checkpoints": [c for c in exp.checkpoints if c <= budget] or [budget],
         "pop_size": exp.pop_size, "algorithm": v.algorithm, "ordered": v.ordered, "seed": s, "tx": tx}
        for v in exp.variants if not args.only or v.algorithm in args.only
        for s in (args.seeds if args.seeds is not None else exp.search_seeds)
        for tx in (args.tx if args.tx is not None else exp.tx_per_hop)
    ]
    # Slowest algorithms first so the workers finish together.
    order = {"smsemoa": 0, "nsga3": 1, "nsga2": 2, "random": 3}
    jobs.sort(key=lambda j: order.get(j["algorithm"], 9))
    log.info(f"{len(jobs)} jobs on {exp.workers} workers -> {run_dir}")

    rows, start = [], time.perf_counter()
    with mp.get_context("spawn").Pool(exp.workers) as pool:
        for i, job_rows in enumerate(pool.imap_unordered(run_job, jobs), 1):
            rows += job_rows
            final = max((r for r in job_rows if r["selection"] == "archive"), key=lambda r: r["evaluations"])
            log.info(f"[{i}/{len(jobs)}] {final['variant']} seed={final['seed']} tx={final['tx']:.0f}: "
                     f"{final['seconds']:.0f}s, HV(log) at {final['evaluations']} evals = {final['hv_log_cost']:.4f}")
            pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["tx", "selection", "variant", "evaluations"])[["hv_log_cost", "hv_linear_cost", "seconds"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(run_dir / "summary.csv")
    with pd.option_context("display.width", 220, "display.max_rows", None):
        log.info(f"done in {(time.perf_counter() - start) / 60:.0f} min\n" + summary.to_string())


if __name__ == "__main__":
    main()
