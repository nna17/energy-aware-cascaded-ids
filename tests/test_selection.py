import pandas as pd

from eids.cascade.selection import SCENARIOS, Scenario, knee, reference_detector, select


def _front():
    return pd.DataFrame({
        "val_macro_f1": [0.60, 0.80, 0.78, 0.50], "val_minority_recall": [0.5, 0.8, 0.85, 0.4],
        "val_fpr": [0.30, 0.04, 0.08, 0.50], "val_detection_rate": [0.99, 0.97, 0.96, 0.90],
        "val_mean_cost": [10.0, 1000.0, 200.0, 5.0], "deployment": ["a", "b", "c", "d"],
    })


def test_select_respects_constraints_and_optimises():
    f = _front()
    ref = f.iloc[1]  # macro-F1 0.80, cost 1000
    battery = next(s for s in SCENARIOS if s.name == "battery")       # F1 >= 0.76 -> b or c, cheapest c
    assert select(f, battery, ref)["deployment"] == "c"
    low_fa = next(s for s in SCENARIOS if s.name == "low_false_alarms")  # FPR <= 0.05 -> only b
    assert select(f, low_fa, ref)["deployment"] == "b"
    impossible = Scenario("x", "", "macro_f1", max_values={"fpr": 0.01})
    assert select(f, impossible, ref) is None


def test_reference_and_knee():
    f = _front()
    assert reference_detector(f)["deployment"] == "b"
    assert knee(f)["deployment"] in {"b", "c"}
