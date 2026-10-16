"""Search algorithms over deployments with a full archive and hypervolume checkpoints.

Fairness rules for comparing algorithms:
- every algorithm gets the same number of deployment evaluations (on the validation split);
- every evaluated deployment goes into an archive, and an algorithm's result at a budget of k
  evaluations is the non-dominated set of its first k evaluations (not only its final population);
- fronts are then scored once on the test split.

Encodings: a tier's model is chosen by rounding a gene in [0, 1] to a position in that tier's list of
eligible models. `ordered=True` sorts each list by cost, so neighbouring genes are similar-sized models
and crossover/mutation act locally; `ordered=False` keeps pool order (as in the first experiments).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from eids.cascade.search import OBJECTIVES, CascadeEvaluator, Deployment, DeploymentEncoding

ALGORITHMS = ("nsga2", "nsga3", "smsemoa", "random")


class OrderedEncoding(DeploymentEncoding):
    """DeploymentEncoding whose per-tier model lists are sorted by cost (ties by pool index).

    `freeze` maps gene index -> fixed value; frozen genes ignore the optimiser (used for ablations:
    thresholds fixed while models are searched, or models fixed while thresholds are searched).
    """

    def __init__(self, ev: CascadeEvaluator, granularity: str, ordered: bool,
                 freeze: dict[int, float] | None = None):
        super().__init__(ev, granularity)
        self.options = [sorted(opts, key=lambda m: (ev.pool.cost[m], m)) if ordered else list(opts)
                        for opts in ev.eligible]
        self.freeze = freeze or {}

    def gene_for_model(self, tier: int, model: int | None) -> float:
        """Gene value that decodes to `model` on `tier` (None = tier unused, tiers 2 and 3 only)."""
        opts = self.options[tier] if tier == 0 else [None] + self.options[tier]
        return (opts.index(model) + 0.5) / len(opts)

    def decode(self, x: np.ndarray) -> Deployment:
        if self.freeze:
            x = np.array(x, dtype=float)
            for i, v in self.freeze.items():
                x[i] = v
        models = [self._pick(x[0], self.options[0])]
        devices = [0]
        for tier, u in ((1, x[1]), (2, x[2])):
            m = self._pick(u, [None] + self.options[tier])
            if m is not None:
                models.append(m)
                devices.append(tier)
        g = self.n_groups
        th = [x[3:3 + g][self.group], x[3 + g:3 + 2 * g][self.group]]
        return Deployment(models, devices, th[: len(models) - 1])


@dataclass
class SearchResult:
    algorithm: str
    X: np.ndarray                 # every evaluated genome, in evaluation order
    F: np.ndarray                 # their validation objectives
    population: np.ndarray        # final population genomes (empty for random search)
    seconds: float
    encoding: OrderedEncoding = field(repr=False)

    def front_indices(self, budget: int) -> np.ndarray:
        """Indices of the validation-non-dominated deployments among the first `budget` evaluations."""
        F = self.F[:budget]
        return NonDominatedSorting().do(F, only_non_dominated_front=True)


class _ArchiveProblem(Problem):
    def __init__(self, enc: OrderedEncoding, budget: int, objectives: list[int]):
        super().__init__(n_var=enc.n_var, n_obj=len(objectives), xl=0.0, xu=1.0)
        self.enc, self.budget, self.objectives = enc, budget, objectives
        self.X: list[np.ndarray] = []
        self.F: list[np.ndarray] = []

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.array([self.enc.ev.evaluate(self.enc.decode(x), "val")[0] for x in X])[:, self.objectives]
        room = self.budget - len(self.X)
        self.X.extend(X[:max(room, 0)])
        self.F.extend(F[:max(room, 0)])
        out["F"] = F


def initial_population(enc: OrderedEncoding, pop_size: int, seed: int, seed_singles: bool) -> np.ndarray:
    """Uniform random genomes; with `seed_singles`, the first rows encode every device-only single-detector
    deployment, so the search starts from (and its archive contains) every single-detector design in its space."""
    X = np.random.default_rng(seed).random((pop_size, enc.n_var))
    if seed_singles:
        singles = enc.options[0]
        for i, m in enumerate(singles[:pop_size]):
            X[i, 0] = enc.gene_for_model(0, m)
            X[i, 1] = enc.gene_for_model(1, None)
            X[i, 2] = enc.gene_for_model(2, None)
    return X


def run_search(ev: CascadeEvaluator, algorithm: str, granularity: str, ordered: bool, budget: int,
               seed: int, pop_size: int = 100, freeze: dict[int, float] | None = None,
               objectives: list[int] | None = None, seed_singles: bool = False) -> SearchResult:
    """`objectives` selects which of OBJECTIVES the search optimises (default: all four).
    `seed_singles` puts every device-only single-detector deployment into the initial population."""
    objectives = list(range(len(OBJECTIVES))) if objectives is None else list(objectives)
    enc = OrderedEncoding(ev, granularity, ordered, freeze)
    start = time.perf_counter()
    sampling = initial_population(enc, pop_size, seed, seed_singles)
    if algorithm == "random":
        X = np.random.default_rng(seed).random((budget, enc.n_var))
        F = np.array([ev.evaluate(enc.decode(x), "val")[0] for x in X])[:, objectives]
        return SearchResult(algorithm, X, F, np.empty((0, enc.n_var)), time.perf_counter() - start, enc)

    if algorithm == "nsga2":
        algo = NSGA2(pop_size=pop_size, sampling=sampling)
    elif algorithm == "nsga3":
        ref_dirs = get_reference_directions("das-dennis", len(objectives), n_partitions=6)
        n = max(pop_size, len(ref_dirs))
        algo = NSGA3(ref_dirs=ref_dirs, pop_size=n, sampling=initial_population(enc, n, seed, seed_singles))
    elif algorithm == "smsemoa":
        algo = SMSEMOA(pop_size=pop_size, sampling=sampling)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}; known: {ALGORITHMS}")

    problem = _ArchiveProblem(enc, budget, objectives)
    res = minimize(problem, algo, ("n_eval", budget), seed=seed, verbose=False)
    pop = np.atleast_2d(res.pop.get("X")) if res.pop is not None else np.empty((0, enc.n_var))
    return SearchResult(algorithm, np.array(problem.X), np.array(problem.F), pop,
                        time.perf_counter() - start, enc)
