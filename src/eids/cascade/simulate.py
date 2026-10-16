"""Offline simulation of an escalation cascade from per-flow predictions and confidences.

Each tier either decides a flow locally or escalates it to the next tier when its confidence
(max class probability) is below the threshold for the class it predicted. A threshold is a scalar
(one for all classes) or an array indexed by predicted class. The last tier always decides.
Costs are per-tier unit costs (ops now; measured energy/latency and transmission later), so any
configuration can be evaluated without re-running models — this is what the optimizer searches over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eids.eval.metrics import compute_metrics

Threshold = float | np.ndarray


@dataclass
class Tier:
    name: str
    pred: np.ndarray        # (n_flows,) predicted class
    conf: np.ndarray        # (n_flows,) max class probability
    cost: float             # cost per flow processed at this tier (e.g. ops or joules)
    escalation_cost: float = 0.0  # cost per flow sent from this tier to the next (e.g. transmission)

    @classmethod
    def from_proba(cls, name: str, proba: np.ndarray, cost: float, escalation_cost: float = 0.0) -> "Tier":
        return cls(name, proba.argmax(axis=1), proba.max(axis=1), cost, escalation_cost)

    @property
    def confidence(self) -> np.ndarray:
        return self.conf


def _escalates(tier: Tier, threshold: Threshold) -> np.ndarray:
    th = threshold[tier.pred] if isinstance(threshold, np.ndarray) else threshold
    return tier.conf < th


def run_cascade(tiers: list[Tier], thresholds: list[Threshold]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predictions, deciding tier per flow, total cost per flow).

    `thresholds[i]` applies to tier i (one per tier except the last): escalate if confidence < threshold.
    """
    if len(thresholds) != len(tiers) - 1:
        raise ValueError("need one threshold per tier except the last")
    n = len(tiers[0].pred)
    pred = np.empty(n, dtype=np.int64)
    decided_at = np.full(n, len(tiers) - 1, dtype=np.int64)
    cost = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)

    for i, tier in enumerate(tiers):
        cost[active] += tier.cost
        last = i == len(tiers) - 1
        decide = active if last else active & ~_escalates(tier, thresholds[i])
        pred[decide] = tier.pred[decide]
        decided_at[decide] = i
        active &= ~decide
        cost[active] += tier.escalation_cost
    return pred, decided_at, cost


def thresholds_for_rates(confidence: np.ndarray, escalation_rate: float) -> float:
    """Threshold that escalates the given fraction of flows (chosen on validation data)."""
    if escalation_rate <= 0:
        return -np.inf
    if escalation_rate >= 1:
        return np.inf
    return float(np.quantile(confidence, escalation_rate))


def cascade_metrics(y: np.ndarray, tiers: list[Tier], thresholds: list[Threshold], n_classes: int,
                    minority: list[int], benign_index: int | None) -> dict:
    pred, decided_at, cost = run_cascade(tiers, thresholds)
    m = compute_metrics(y, pred, n_classes, minority, benign_index)
    out = {k: m[k] for k in ("macro_f1", "minority_recall", "false_positive_rate", "detection_rate", "accuracy")}
    out["mean_cost"] = float(cost.mean())
    for i, tier in enumerate(tiers):
        out[f"share_{tier.name}"] = float(np.mean(decided_at == i))
    return out
