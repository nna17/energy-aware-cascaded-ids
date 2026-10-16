"""A2-A5: joint optimisation of tier models, tiers and escalation thresholds vs baselines.

Search and selection use the validation split only; every front is then scored once on the test split.
Hypervolume (higher is better) is computed on test objectives with reference point (1, 1, 1, 1), once with
log-scaled cost (what the search optimises) and once with linear cost (the originally planned metric).

    python experiments/joint_search.py --config configs/experiments/joint_search.yaml
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from eids.cascade.algorithms import run_search
from eids.cascade.pool import load_pool
from eids.cascade.search import (CascadeEvaluator, Deployment, Device, fixed_two_tier_deployments, front_on,
                                 hypervolume, single_model_deployments)
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()


def searched(ev: CascadeEvaluator, exp, algorithm: str, granularity: str, seed: int) -> list[Deployment]:
    """Non-dominated deployments of the algorithm's full archive (cost-ordered encoding, equal budget)."""
    s = exp.search
    start = time.perf_counter()
    res = run_search(ev, algorithm, granularity, ordered=s.get("ordered", True), budget=s.budget,
                     seed=seed, pop_size=s.pop_size, objectives=s.get("objectives"),
                     seed_singles=s.get("seed_singles", False) and algorithm != "random")
    deps = [res.encoding.decode(res.X[i]) for i in res.front_indices(s.budget)]
    log.info(f"  {algorithm}_{granularity}: {len(deps)} archive-front deployments in {time.perf_counter() - start:.0f}s")
    return deps


def run_methods(ev: CascadeEvaluator, exp, seed: int, granularities: list[str]) -> dict[str, list]:
    alg = exp.search.algorithm
    methods = {"single_model": single_model_deployments(ev), "fixed_two_tier": fixed_two_tier_deployments(ev)}
    for g in granularities:
        methods[f"{alg}_{g}"] = searched(ev, exp, alg, g, seed)
    g = exp.random_search_granularity
    methods[f"random_{g}"] = searched(ev, exp, "random", g, seed)
    return methods


def score(ev: CascadeEvaluator, methods: dict[str, list], tag: dict) -> tuple[list[dict], list[dict]]:
    """HV with the search's log-scaled cost (primary) and with linear cost (as first planned)."""
    summary, fronts = [], []
    for name, deps in methods.items():
        F, rows = front_on(ev, deps, "test")
        F_linear = F.copy()
        F_linear[:, 3] = [r["mean_cost"] / ev.cost_ref for r in rows]
        summary.append({**tag, "method": name, "hv_log_cost": hypervolume(F),
                        "hv_linear_cost": hypervolume(F_linear), "front_size": len(rows)})
        fronts += [{**tag, "method": name, **r} for r in rows]
        log.info(f"  {name:<18} HV(log cost)={summary[-1]['hv_log_cost']:.4f} "
                 f"HV(linear cost)={summary[-1]['hv_linear_cost']:.4f} front={len(rows)}")
    return summary, fronts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/joint_search.yaml")
    args = parser.parse_args()
    exp = load_config(args.config)
    run_dir = make_run_dir(exp.name)
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")
    devices = [Device(**d) for d in exp.devices]

    summary, fronts = [], []
    for seed in exp.seeds:
        pool = load_pool(resolve_path(exp.pool_dir.format(seed=seed)))
        log.info(f"seed {seed}: pool of {pool.n_models} models")

        log.info(f"main comparison (tx per hop = {list(exp.tx_cost_per_hop)})")
        score_name = exp.get("score", "confidence")
        ev = CascadeEvaluator(pool, devices, exp.tx_cost_per_hop, score=score_name)
        s, f = score(ev, run_methods(ev, exp, seed, exp.granularities),
                     {"seed": seed, "study": "main", "tx": exp.tx_cost_per_hop[0]})
        summary += s
        fronts += f

        for tx in exp.get("tx_sweep") or []:
            log.info(f"transmission sweep: tx per hop = {tx}")
            ev = CascadeEvaluator(pool, devices, [tx, tx], score=score_name)
            alg = exp.search.algorithm
            methods = {"single_model": single_model_deployments(ev),
                       "fixed_two_tier": fixed_two_tier_deployments(ev),
                       f"{alg}_category": searched(ev, exp, alg, "category", seed),
                       "random_category": searched(ev, exp, "random", "category", seed)}
            s, f = score(ev, methods, {"seed": seed, "study": "tx_sweep", "tx": tx})
            summary += s
            fronts += f

        pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)  # checkpoint per seed
        pd.DataFrame(fronts).to_csv(run_dir / "fronts.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", None):
        log.info("Hypervolume on test (higher is better):\n" + pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
