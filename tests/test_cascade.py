import numpy as np
import pytest

from eids.cascade import Tier, run_cascade, thresholds_for_rates


def test_cascade_escalates_only_uncertain_flows_and_adds_costs():
    t1 = Tier.from_proba("mcu", np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]]), cost=1, escalation_cost=10)
    t2 = Tier.from_proba("gw", np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]]), cost=100)
    pred, decided_at, cost = run_cascade([t1, t2], thresholds=[0.7])
    assert decided_at.tolist() == [0, 1, 0]
    assert pred.tolist() == [0, 1, 1]           # flow 1 takes the gateway's answer
    assert cost.tolist() == [1, 111, 1]         # 1 local + 10 transmit + 100 gateway


def test_threshold_for_rate_escalates_that_fraction():
    conf = np.linspace(0, 1, 1001)
    th = thresholds_for_rates(conf, 0.2)
    assert np.mean(conf < th) == pytest.approx(0.2, abs=0.002)
    assert thresholds_for_rates(conf, 0.0) == -np.inf
