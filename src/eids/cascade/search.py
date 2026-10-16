"""Joint search over deployments: which model on each tier, which tiers exist, and escalation thresholds.

A deployment is up to three tiers (device 0 = MCU, 1 = gateway, 2 = edge server). Tier 1 always
exists; tiers 2 and 3 are optional, so "MCU -> server" and "MCU only" are deployments too.
Each tier except the last has escalation thresholds at a chosen granularity (one global value, one per
attack category, or one per class), applied to the class the tier predicted.

Objectives (all minimised), evaluated from the candidate pool without re-running any model:
    1 - macro-F1, 1 - minority recall, benign false-positive rate, mean cost per flow (log-scaled, normalised)
Cost per flow = model ops x device cost-per-op on every tier it visits + transmission per hop it is sent
(including the hops from the IoT device to the first tier).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from eids.cascade.pool import Pool
from eids.cascade.simulate import Tier, run_cascade
from eids.eval.metrics import fast_metrics

OBJECTIVES = ["1-macro_f1", "1-minority_recall", "fpr", "cost"]


@dataclass
class Device:
    name: str
    max_size_kb: float | None = None   # model must fit (flash budget)
    max_features: int | None = None    # flow features the device can extract
    cost_per_op: float = 1.0           # ops -> cost units (measured energy or latency later)


@dataclass
class Deployment:
    models: list[int]                  # pool index per tier
    devices: list[int]                 # device index per tier (increasing)
    thresholds: list[np.ndarray] = field(default_factory=list)  # per-class arrays, one per tier but the last

    def describe(self, pool: Pool, devices: list[Device]) -> str:
        return " -> ".join(f"{devices[d].name}:{pool.names[m]}" for m, d in zip(self.models, self.devices))


class CascadeEvaluator:
    """`score` is the uncertainty score compared with escalation thresholds: "confidence" (max class
    probability) or "margin" (top-1 minus top-2 probability)."""

    def __init__(self, pool: Pool, devices: list[Device], tx_cost_per_hop: list[float], score: str = "confidence"):
        self.pool, self.devices, self.tx, self.score = pool, devices, list(tx_cost_per_hop), score
        source = {"confidence": pool.conf, "margin": pool.margin}[score]
        if source is None:
            raise ValueError(f"pool has no {score!r} scores; rebuild it")
        self.conf = {s: c.astype(np.float32) for s, c in source.items()}
        self.eligible = [self._eligible(d) for d in devices]
        max_op = max(d.cost_per_op for d in devices)
        self.cost_ref = float(pool.cost.max() * max_op * len(devices) + sum(self.tx))
        # Costs span orders of magnitude (a 7-comparison tree to an 84k-comparison booster), so the cost
        # objective is log-scaled; on a linear scale nearly every deployment sits at ~0 and cost exerts
        # almost no selection pressure (joint_search run of 2026-10-14).
        self.log_cost_ref = float(np.log10(1 + self.cost_ref))

    def cost_objective(self, mean_cost: float) -> float:
        return float(np.log10(1 + mean_cost) / self.log_cost_ref)

    def _eligible(self, d: Device) -> list[int]:
        ok = np.ones(self.pool.n_models, dtype=bool)
        if d.max_size_kb is not None:
            ok &= self.pool.size_kb <= d.max_size_kb
        if d.max_features is not None:
            ok &= self.pool.n_features <= d.max_features
        return np.flatnonzero(ok).tolist()

    def tiers(self, dep: Deployment, split: str) -> list[Tier]:
        """Flows originate at device 0 (the IoT device). A deployment whose first tier runs elsewhere pays
        the transmission to reach it for every flow, so a server-only detector is not free of communication."""
        tiers = []
        for i, (m, d) in enumerate(zip(dep.models, dep.devices)):
            nxt = dep.devices[i + 1] if i + 1 < len(dep.devices) else d
            hop_cost = sum(self.tx[d:nxt])  # relayed through intermediate devices
            ingress = sum(self.tx[:d]) if i == 0 else 0.0
            tiers.append(Tier(f"t{i + 1}", self.pool.pred[split][m], self.conf[split][m],
                              cost=self.pool.cost[m] * self.devices[d].cost_per_op + ingress,
                              escalation_cost=hop_cost))
        return tiers

    def evaluate(self, dep: Deployment, split: str) -> tuple[np.ndarray, dict]:
        pred, decided_at, cost = run_cascade(self.tiers(dep, split), dep.thresholds)
        f1, min_rec, fpr = fast_metrics(self.pool.y[split].astype(np.int64), pred, self.pool.n_classes,
                                        self.pool.minority, self.pool.benign_index)
        mean_cost = float(cost.mean())
        y = self.pool.y[split]
        attack = y != self.pool.benign_index
        detection = float(np.mean(pred[attack] != self.pool.benign_index))
        objectives = np.array([1 - f1, 1 - min_rec, fpr, self.cost_objective(mean_cost)])
        extra = {"macro_f1": f1, "minority_recall": min_rec, "fpr": fpr, "detection_rate": detection,
                 "mean_cost": mean_cost,
                 **{f"share_t{i + 1}": float(np.mean(decided_at == i)) for i in range(len(dep.models))}}
        return objectives, extra


class DeploymentEncoding:
    """Real vector in [0,1]^n -> Deployment.  x = [tier1, tier2|none, tier3|none, thresholds tier1, thresholds tier2]."""

    def __init__(self, ev: CascadeEvaluator, granularity: str):
        self.ev = ev
        self.group = ev.pool.class_group[granularity]
        self.n_groups = int(self.group.max()) + 1
        self.n_var = 3 + 2 * self.n_groups

    @staticmethod
    def _pick(u: float, options: list) -> object:
        return options[min(int(u * len(options)), len(options) - 1)]

    def decode(self, x: np.ndarray) -> Deployment:
        e = self.ev.eligible
        models = [self._pick(x[0], e[0])]
        devices = [0]
        for tier, u in ((1, x[1]), (2, x[2])):
            m = self._pick(u, [None] + e[tier])
            if m is not None:
                models.append(m)
                devices.append(tier)
        g = self.n_groups
        th = [x[3:3 + g][self.group], x[3 + g:3 + 2 * g][self.group]]
        return Deployment(models, devices, th[: len(models) - 1])


class _Problem(Problem):
    def __init__(self, enc: DeploymentEncoding):
        super().__init__(n_var=enc.n_var, n_obj=len(OBJECTIVES), xl=0.0, xu=1.0)
        self.enc = enc

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.enc.ev.evaluate(self.enc.decode(x), "val")[0] for x in X])


def nsga2_search(ev: CascadeEvaluator, granularity: str, pop_size: int, n_gen: int, seed: int) -> list[Deployment]:
    enc = DeploymentEncoding(ev, granularity)
    res = minimize(_Problem(enc), NSGA2(pop_size=pop_size), ("n_gen", n_gen), seed=seed, verbose=False)
    X = np.atleast_2d(res.X)
    return [enc.decode(x) for x in X]


def random_search(ev: CascadeEvaluator, granularity: str, n_evals: int, seed: int) -> list[Deployment]:
    enc = DeploymentEncoding(ev, granularity)
    rng = np.random.default_rng(seed)
    deps = [enc.decode(x) for x in rng.random((n_evals, enc.n_var))]
    F = np.array([ev.evaluate(d, "val")[0] for d in deps])
    return [deps[i] for i in NonDominatedSorting().do(F, only_non_dominated_front=True)]


def single_model_deployments(ev: CascadeEvaluator) -> list[Deployment]:
    """Every model alone, on every device it fits (no escalation)."""
    return [Deployment([m], [d], []) for d, opts in enumerate(ev.eligible) for m in opts]


def fixed_two_tier_deployments(ev: CascadeEvaluator) -> list[Deployment]:
    """C2T-IDS-style fixed design: gateway model keeps flows it calls benign, sends every predicted attack
    to the server model. No MCU tier and no tuned thresholds."""
    n = ev.pool.n_classes
    keep_benign = np.full(n, 2.0)                # confidence < 2 always -> escalate
    keep_benign[ev.pool.benign_index] = 0.0      # predicted benign -> decide locally
    return [Deployment([g, s], [1, 2], [keep_benign]) for g in ev.eligible[1] for s in ev.eligible[2]]


def fixed_device_gateway_deployments(ev: CascadeEvaluator) -> list[Deployment]:
    """Device-first fixed design: the device model keeps flows it predicts as benign and forwards every
    predicted attack to a gateway model. Same fixed rule as the C2T-style design, one tier lower."""
    n = ev.pool.n_classes
    keep_benign = np.full(n, 2.0)
    keep_benign[ev.pool.benign_index] = 0.0
    return [Deployment([d, g], [0, 1], [keep_benign]) for d in ev.eligible[0] for g in ev.eligible[1]]


def device_only_deployments(ev: CascadeEvaluator) -> list[Deployment]:
    return [Deployment([m], [0], []) for m in ev.eligible[0]]


def front_on(ev: CascadeEvaluator, deps: list[Deployment], split: str) -> tuple[np.ndarray, list[dict]]:
    """Objectives and details of deployments on a split, keeping only the val-non-dominated ones.

    Each row carries the split's metrics and the validation metrics (prefixed `val_`), so that a single
    deployment can later be selected on validation and reported on `split` (see cascade/selection.py)."""
    val = [ev.evaluate(d, "val") for d in deps]
    val_F = np.array([v[0] for v in val])
    keep = NonDominatedSorting().do(val_F, only_non_dominated_front=True)
    rows, F = [], []
    for i in keep:
        f, extra = ev.evaluate(deps[i], split)
        F.append(f)
        rows.append({"deployment": deps[i].describe(ev.pool, ev.devices), **extra,
                     **{f"val_{k}": v for k, v in val[i][1].items() if not k.startswith("share_")},
                     "thresholds": json.dumps([np.round(t, 4).tolist() for t in deps[i].thresholds])})
    return np.array(F), rows


def hypervolume(F: np.ndarray, ref: np.ndarray | None = None) -> float:
    ref = np.ones(F.shape[1]) if ref is None else ref
    return float(HV(ref_point=ref)(F)) if len(F) else 0.0
