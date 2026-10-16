"""Cost-sensitivity analysis: re-run the joint search and the baselines under model-based energy profiles
(nJ per flow) for several radios, two perspectives (system energy vs device battery energy) and scaled
compute/radio costs, and check whether the ranking of designs holds.

    python experiments/cost_sensitivity.py --config configs/experiments/cost_sensitivity.yaml
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import pandas as pd

from eids.cascade.algorithms import run_search
from eids.cascade.pool import load_pool
from eids.cascade.search import (CascadeEvaluator, Device, device_only_deployments, fixed_device_gateway_deployments,
                                 fixed_two_tier_deployments, front_on, hypervolume, single_model_deployments)
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger
from strong_baselines import greedy, tuned_two_tier

log = get_logger()
TIERS = ("mcu", "gateway", "server")


def profile_costs(exp: dict, p: dict) -> tuple[list[Device], list[float]]:
    cs, rs = p.get("compute_scale", 1.0), p.get("radio_scale", 1.0)
    rs *= exp.get("record_scale", 1.0)  # dataset's flow-record size relative to the 200-byte reference record
    device_only = p["perspective"] == "device"
    devices = []
    for tier in TIERS:
        op = exp["op_nj"][tier] * cs
        if device_only and tier != "mcu":
            op = 0.0  # gateway/server energy does not drain the device battery
        devices.append(Device(tier, cost_per_op=op, **exp["budgets"][tier]))
    hop1 = exp["radio_nj"][p["radio"]] * rs
    hop2 = 0.0 if device_only else exp["backhaul_nj"] * rs
    return devices, [hop1, hop2]


def run_job(job: dict) -> list[dict]:
    exp, p = job["exp"], job["profile"]
    pool = load_pool(resolve_path(exp["pool_dir"].format(seed=job["seed"])))
    devices, tx = profile_costs(exp, p)
    ev = CascadeEvaluator(pool, devices, tx)
    start = time.perf_counter()

    def joint() -> list:
        res = run_search(ev, "nsga2", exp["granularity"], True, exp["budget"], job["seed"], exp["pop_size"],
                         seed_singles=exp.get("seed_singles", False))
        return [res.encoding.decode(res.X[i]) for i in res.front_indices(exp["budget"])]

    builders = {
        "single_model": lambda: single_model_deployments(ev),
        "device_only": lambda: device_only_deployments(ev),
        "device_gateway_fixed": lambda: fixed_device_gateway_deployments(ev),
        "gateway_server_c2t": lambda: fixed_two_tier_deployments(ev),
        "joint_nsga2": joint,
        # stronger baselines (experiments/strong_baselines.py), only when listed in `methods`
        "tuned_two_tier": lambda: tuned_two_tier(ev),
        "greedy": lambda: greedy(ev, exp["budget"], job["seed"]),
    }
    names = exp.get("methods") or ["single_model", "device_only", "device_gateway_fixed", "gateway_server_c2t",
                                   "joint_nsga2"]
    methods = {name: builders[name]() for name in names}
    summary, fronts = [], []
    for name, deps in methods.items():
        F, rows = front_on(ev, deps, "test")
        summary.append({"profile": p["name"], "perspective": p["perspective"], "radio": p["radio"],
                        "seed": job["seed"], "method": name, "hv_log_cost": hypervolume(F), "front_size": len(rows),
                        "min_cost_nj": min(r["mean_cost"] for r in rows),
                        "best_macro_f1": max(r["macro_f1"] for r in rows)})
        fronts += [{"profile": p["name"], "seed": job["seed"], "method": name, **r} for r in rows]
    log.info(f"{p['name']} seed={job['seed']} done in {time.perf_counter() - start:.0f}s")
    return [summary, fronts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/cost_sensitivity.yaml")
    args = parser.parse_args()
    exp = load_config(args.config)
    run_dir = make_run_dir(exp.name)
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")
    plain = {k: v for k, v in exp.items() if k != "profiles"}
    jobs = [{"exp": plain, "profile": dict(p), "seed": s} for p in exp.profiles for s in exp.search_seeds]
    log.info(f"{len(jobs)} jobs on {exp.workers} workers -> {run_dir}")
    summary, fronts = [], []
    with mp.get_context("spawn").Pool(exp.workers) as pool:
        for s, f in pool.imap_unordered(run_job, jobs):
            summary += s
            fronts += f
            pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
            pd.DataFrame(fronts).to_csv(run_dir / "fronts.csv", index=False)
    table = pd.DataFrame(summary).pivot_table(index="profile", columns="method", values="hv_log_cost").round(4)
    with pd.option_context("display.width", 220):
        log.info("Hypervolume (log energy, test):\n" + table.to_string())


if __name__ == "__main__":
    main()
