"""Fair-baseline ablation ladder (same tiers, same detector pool):

  A device_only          each device-eligible detector alone
  B device_gateway_fixed device keeps predicted benign, forwards predicted attacks to the gateway
  C models_only          NSGA-II over detectors and tiers; escalation thresholds fixed
  D thresholds_only      NSGA-II over per-category thresholds; detectors fixed to the best per tier (validation)
  E joint                NSGA-II over detectors, tiers and per-category thresholds
  F gateway_server_c2t   literature reference: gateway keeps predicted benign, server classifies the rest

    python experiments/ablation_ladder.py --config configs/experiments/ablation_ladder.yaml
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from eids.cascade.algorithms import OrderedEncoding, run_search
from eids.cascade.pool import load_pool
from eids.cascade.search import (CascadeEvaluator, Device, device_only_deployments, fixed_device_gateway_deployments,
                                 fixed_two_tier_deployments, front_on, hypervolume)
from eids.eval.metrics import fast_metrics
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()
SEARCHED = ("models_only", "thresholds_only", "joint")


def best_per_tier(ev: CascadeEvaluator) -> list[int]:
    """Detector with the highest validation macro-F1 among those eligible on each tier."""
    y = ev.pool.y["val"].astype(np.int64)
    best = []
    for opts in ev.eligible:
        scores = [fast_metrics(y, ev.pool.pred["val"][m].astype(np.int64), ev.pool.n_classes,
                               ev.pool.minority, ev.pool.benign_index)[0] for m in opts]
        best.append(opts[int(np.argmax(scores))])
    return best


def run_job(job: dict) -> list[dict]:
    pool = load_pool(resolve_path(job["pool_dir"].format(seed=max(job["seed"], 0))))
    ev = CascadeEvaluator(pool, [Device(**d) for d in job["devices"]], [job["tx"], job["tx"]])
    start = time.perf_counter()
    variant, extra = job["variant"], {}
    if variant == "device_only":
        deps = device_only_deployments(ev)
    elif variant == "device_gateway_fixed":
        deps = fixed_device_gateway_deployments(ev)
    elif variant == "gateway_server_c2t":
        deps = fixed_two_tier_deployments(ev)
    else:
        probe = OrderedEncoding(ev, job["granularity"], ordered=True)
        g = probe.n_groups
        if variant == "models_only":
            freeze = {i: job["fixed_threshold"] for i in range(3, 3 + 2 * g)}
        elif variant == "thresholds_only":
            best = best_per_tier(ev)
            freeze = {t: probe.gene_for_model(t, m) for t, m in enumerate(best)}
            extra["fixed_models"] = " -> ".join(pool.names[m] for m in best)
        else:
            freeze = None
        res = run_search(ev, "nsga2", job["granularity"], True, job["budget"], job["seed"], job["pop_size"], freeze,
                         seed_singles=job.get("seed_singles", False) and variant == "joint")
        deps = [res.encoding.decode(res.X[i]) for i in res.front_indices(job["budget"])]
    F, rows = front_on(ev, deps, "test")
    F_lin = F.copy()
    F_lin[:, 3] = [r["mean_cost"] / ev.cost_ref for r in rows]
    summary = {"variant": variant, "seed": job["seed"], "tx": job["tx"], "hv_log_cost": hypervolume(F),
               "hv_linear_cost": hypervolume(F_lin), "front_size": len(rows),
               "seconds": time.perf_counter() - start, **extra}
    fronts = [{"variant": variant, "seed": job["seed"], "tx": job["tx"], **r} for r in rows]
    return [summary, fronts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/ablation_ladder.yaml")
    args = parser.parse_args()
    exp = load_config(args.config)
    run_dir = make_run_dir(exp.name)
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")

    base = {"pool_dir": exp.pool_dir, "devices": [dict(d) for d in exp.devices], "granularity": exp.granularity,
            "budget": exp.budget, "pop_size": exp.pop_size, "fixed_threshold": exp.fixed_threshold,
            "seed_singles": exp.get("seed_singles", False)}
    jobs = []
    for tx in exp.tx_per_hop:
        for v in ("device_only", "device_gateway_fixed", "gateway_server_c2t"):
            jobs.append({**base, "variant": v, "seed": -1, "tx": tx})
        for v in SEARCHED:
            jobs += [{**base, "variant": v, "seed": s, "tx": tx} for s in exp.search_seeds]
    jobs.sort(key=lambda j: j["variant"] not in SEARCHED)  # long jobs first
    log.info(f"{len(jobs)} jobs on {exp.workers} workers -> {run_dir}")

    summary, fronts = [], []
    with mp.get_context("spawn").Pool(exp.workers) as pool:
        for i, (s, f) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            summary.append(s)
            fronts += f
            log.info(f"[{i}/{len(jobs)}] {s['variant']:<22} seed={s['seed']:>2} tx={s['tx']:>6.0f} "
                     f"HV(log)={s['hv_log_cost']:.4f} front={s['front_size']} {s['seconds']:.0f}s")
            pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
            pd.DataFrame(fronts).to_csv(run_dir / "fronts.csv", index=False)

    df = pd.DataFrame(summary)
    table = df.pivot_table(index="variant", columns="tx", values="hv_log_cost", aggfunc=["mean", "std"]).round(4)
    with pd.option_context("display.width", 220):
        log.info("Hypervolume (log cost, test):\n" + table.to_string())


if __name__ == "__main__":
    main()
