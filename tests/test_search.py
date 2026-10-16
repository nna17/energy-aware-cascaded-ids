import numpy as np
import pytest

from eids.cascade import Tier, run_cascade
from eids.cascade.pool import build_pool, load_pool, model_grid
from eids.cascade.search import (CascadeEvaluator, Device, fixed_two_tier_deployments, front_on, hypervolume,
                                 nsga2_search, random_search, single_model_deployments)
from eids.eval.metrics import compute_metrics, fast_metrics
from eids.utils.config import Config


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    pred = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 5, 2000))
    pred[pred == 4] = 3  # class 4 never predicted -> zero precision handling
    ref = compute_metrics(y, pred, 5, minority=[3, 4], benign_index=0)
    f1, min_rec, fpr = fast_metrics(y, pred, 5, minority=[3, 4], benign_index=0)
    assert f1 == pytest.approx(ref["macro_f1"])
    assert min_rec == pytest.approx(ref["minority_recall"])
    assert fpr == pytest.approx(ref["false_positive_rate"])


def test_per_class_thresholds_use_the_predicted_class():
    t1 = Tier("mcu", pred=np.array([0, 1, 1]), conf=np.array([0.6, 0.6, 0.95]), cost=1)
    t2 = Tier("gw", pred=np.array([2, 2, 2]), conf=np.ones(3), cost=10)
    # class 0 kept at 0.5, class 1 must be >= 0.9 to stay local
    pred, decided_at, _ = run_cascade([t1, t2], [np.array([0.5, 0.9, 0.5])])
    assert decided_at.tolist() == [0, 1, 0]
    assert pred.tolist() == [0, 2, 1]


@pytest.fixture(scope="module")
def pool(synthetic_dm, synthetic_cfg, tmp_path_factory):
    specs = model_grid([Config(kind="decision_tree", max_leaves=[8, 64], n_features=[4, 16]),
                        Config(kind="random_forest", n_estimators=[5], max_leaves=[32]),
                        Config(kind="mlp", hidden=[[8]], n_features=[4], epochs=2)], synthetic_dm.n_features)
    out = build_pool(specs, synthetic_dm, synthetic_cfg, seed=0, out_dir=tmp_path_factory.mktemp("pool"))
    return load_pool(out)


def test_joint_search_end_to_end(pool):
    devices = [Device("mcu", max_size_kb=4, max_features=4), Device("gateway"), Device("server")]
    ev = CascadeEvaluator(pool, devices, tx_cost_per_hop=[10, 10])
    assert all(pool.n_features[m] <= 4 and pool.size_kb[m] <= 4 for m in ev.eligible[0])

    found = nsga2_search(ev, "category", pop_size=12, n_gen=3, seed=0)
    rand = random_search(ev, "class", n_evals=36, seed=0)
    for deps in (found, rand, single_model_deployments(ev), fixed_two_tier_deployments(ev)):
        F, rows = front_on(ev, deps, "test")
        assert F.shape[1] == 4 and len(rows) == len(F) >= 1
        assert 0.0 <= hypervolume(F) <= 1.0
        assert all(0.0 <= f <= 1.0 for f in F[:, 3])  # log-scaled cost stays normalised
    # tier 1 always runs on the MCU and only MCU-eligible models are chosen
    assert all(d.devices[0] == 0 and d.models[0] in ev.eligible[0] for d in found)


def test_detectors_away_from_the_device_pay_to_receive_every_flow(pool):
    devices = [Device("mcu"), Device("gateway"), Device("server")]
    ev = CascadeEvaluator(pool, devices, tx_cost_per_hop=[10, 100])
    from eids.cascade.search import Deployment
    _, on_mcu = ev.evaluate(Deployment([0], [0], []), "test")
    _, on_server = ev.evaluate(Deployment([0], [2], []), "test")
    assert on_server["mean_cost"] == pytest.approx(on_mcu["mean_cost"] + 110)


def test_pool_has_margin_and_mlp_and_fronts_keep_thresholds(pool):
    assert pool.margin is not None and any(n.startswith("mlp_") for n in pool.names)
    assert (pool.margin["val"] <= pool.conf["val"]).all()
    devices = [Device("mcu"), Device("gateway"), Device("server")]
    ev = CascadeEvaluator(pool, devices, [0, 0], score="margin")
    F, rows = front_on(ev, nsga2_search(ev, "category", pop_size=8, n_gen=2, seed=0), "test")
    assert all("thresholds" in r and 0.0 <= r["detection_rate"] <= 1.0 for r in rows)


def test_search_on_a_subset_of_objectives(pool):
    from eids.cascade.algorithms import run_search
    ev = CascadeEvaluator(pool, [Device("mcu"), Device("gateway"), Device("server")], [0, 0])
    res = run_search(ev, "nsga2", "category", True, budget=60, seed=0, pop_size=12, objectives=[0, 2, 3])
    assert res.F.shape == (60, 3)
