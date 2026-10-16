"""Choosing one deployment from a Pareto front: operator scenarios (constraints) and the knee point.

Every choice is made on validation metrics (`val_*` columns written by search.front_on) and reported on
the test metrics of the same deployment, so selection never looks at test data.

A scenario is: hard constraints on validation metrics, then one metric to optimise. Constraints may refer
to the reference detector: the single detector with the best validation macro-F1 (the strongest model an
operator would otherwise deploy alone).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"fpr", "mean_cost"}


@dataclass
class Scenario:
    name: str
    description: str
    optimise: str                                     # metric name (without val_ prefix)
    max_values: dict[str, float] = field(default_factory=dict)       # metric <= value
    min_values: dict[str, float] = field(default_factory=dict)       # metric >= value
    min_fraction_of_reference: dict[str, float] = field(default_factory=dict)  # metric >= f * reference
    max_fraction_of_reference: dict[str, float] = field(default_factory=dict)  # metric <= f * reference


SCENARIOS = [
    Scenario("battery", "lowest cost with macro-F1 >= 95% of the reference detector",
             optimise="mean_cost", min_fraction_of_reference={"macro_f1": 0.95}),
    Scenario("low_false_alarms", "highest macro-F1 with benign FPR <= 5%",
             optimise="macro_f1", max_values={"fpr": 0.05}),
    Scenario("rare_attacks", "highest minority recall at <= 10% of the reference detector's cost",
             optimise="minority_recall", max_fraction_of_reference={"mean_cost": 0.10}),
    Scenario("balanced", "lowest cost with benign FPR <= 10% and attack detection >= 95%",
             optimise="mean_cost", max_values={"fpr": 0.10}, min_values={"detection_rate": 0.95}),
    Scenario("strict_false_alarms", "highest macro-F1 with benign FPR <= 1%",
             optimise="macro_f1", max_values={"fpr": 0.01}),
    Scenario("secure_budget", "highest macro-F1 with benign FPR <= 5%, attack detection >= 95% and at most 10% "
             "of the reference detector's cost",
             optimise="macro_f1", max_values={"fpr": 0.05}, min_values={"detection_rate": 0.95},
             max_fraction_of_reference={"mean_cost": 0.10}),
]


def reference_detector(single_front: pd.DataFrame) -> pd.Series:
    """Single-detector deployment with the best validation macro-F1."""
    return single_front.loc[single_front["val_macro_f1"].idxmax()]


def select(front: pd.DataFrame, scenario: Scenario, reference: pd.Series) -> pd.Series | None:
    ok = pd.Series(True, index=front.index)
    for m, v in scenario.max_values.items():
        ok &= front[f"val_{m}"] <= v
    for m, v in scenario.min_values.items():
        ok &= front[f"val_{m}"] >= v
    for m, f in scenario.min_fraction_of_reference.items():
        ok &= front[f"val_{m}"] >= f * reference[f"val_{m}"]
    for m, f in scenario.max_fraction_of_reference.items():
        ok &= front[f"val_{m}"] <= f * reference[f"val_{m}"]
    feasible = front[ok]
    if feasible.empty:
        return None
    col = f"val_{scenario.optimise}"
    return feasible.loc[feasible[col].idxmin() if scenario.optimise in LOWER_IS_BETTER else feasible[col].idxmax()]


def knee(front: pd.DataFrame) -> pd.Series:
    """Deployment closest to the ideal point after min-max normalising the four validation objectives
    (1 - macro-F1, 1 - minority recall, FPR, log cost) over the front."""
    obj = np.c_[1 - front["val_macro_f1"], 1 - front["val_minority_recall"], front["val_fpr"],
                np.log1p(front["val_mean_cost"])]
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    norm = (obj - lo) / np.where(hi > lo, hi - lo, 1)
    return front.iloc[int(np.argmin(np.linalg.norm(norm, axis=1)))]
