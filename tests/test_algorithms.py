import numpy as np
import pytest

from eids.cascade.algorithms import ALGORITHMS, OrderedEncoding, run_search
from eids.cascade.search import CascadeEvaluator, Device
from tests.test_search import pool  # noqa: F401  (module fixture: small synthetic pool)


@pytest.fixture
def ev(pool):  # noqa: F811
    return CascadeEvaluator(pool, [Device("mcu"), Device("gateway"), Device("server")], [10, 10])


def test_ordered_encoding_sorts_models_by_cost(ev, pool):  # noqa: F811
    enc = OrderedEncoding(ev, "category", ordered=True)
    for opts in enc.options:
        assert list(pool.cost[opts]) == sorted(pool.cost[opts])


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_respects_the_budget_and_archives_all(ev, algorithm):
    res = run_search(ev, algorithm, "category", ordered=True, budget=120, seed=0, pop_size=20)
    assert len(res.X) == len(res.F) == 120
    front = res.front_indices(60)
    assert len(front) >= 1 and front.max() < 60


def test_frozen_genes_fix_models_or_thresholds(ev):  # noqa: F811
    enc = OrderedEncoding(ev, "category", ordered=True)
    target = enc.options[0][-1]
    frozen = OrderedEncoding(ev, "category", ordered=True,
                             freeze={0: enc.gene_for_model(0, target), 1: enc.gene_for_model(1, None)})
    for x in np.random.default_rng(0).random((20, enc.n_var)):
        dep = frozen.decode(x)
        assert dep.models[0] == target and 1 not in dep.devices
    th = OrderedEncoding(ev, "category", ordered=True, freeze={i: 0.8 for i in range(3, enc.n_var)})
    dep = th.decode(np.zeros(enc.n_var) + 0.99)
    assert all(np.allclose(t, 0.8) for t in dep.thresholds)


def test_seeded_search_archive_contains_every_device_only_single(ev):  # noqa: F811
    res = run_search(ev, "nsga2", "category", True, budget=60, seed=0, pop_size=40, seed_singles=True)
    singles = {(m,) for m in res.encoding.options[0]}
    found = {tuple(res.encoding.decode(x).models) for x in res.X}
    assert singles <= found
