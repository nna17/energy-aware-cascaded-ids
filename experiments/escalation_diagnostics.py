"""Escalation diagnostics for device-tier detectors (reviewer questions on confidence and calibration).

For each detector, on the validation split:
1. how well uncertainty scores separate wrong from correct predictions (AUROC): max probability,
   margin (top-1 minus top-2), normalised entropy;
2. calibration: expected calibration error (ECE) before and after isotonic calibration of the confidence;
3. confidently missed attacks: for a benign-keep threshold tau, the share of attack flows that the device
   predicts as benign with confidence >= tau (kept locally, never escalated), and the share of benign flows
   escalated.

    python experiments/escalation_diagnostics.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from eids.data.datamodule import load_datamodule
from eids.models.baselines.trees import TreeModel
from eids.utils.config import load_config
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()
DETECTORS = [
    {"kind": "decision_tree", "n_estimators": 1, "max_leaves": 256, "n_features": 20},
    {"kind": "decision_tree", "n_estimators": 1, "max_leaves": 1024, "n_features": 20},
    {"kind": "random_forest", "n_estimators": 10, "max_leaves": 128},
]
TAUS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def uncertainty_scores(p: np.ndarray) -> dict[str, np.ndarray]:
    """Higher = more certain, for every score."""
    top2 = np.sort(p, axis=1)[:, -2:]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1) / np.log(p.shape[1])
    return {"max_probability": top2[:, 1], "margin": top2[:, 1] - top2[:, 0], "neg_entropy": -ent}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data/ciciot2023.yaml")
    args = parser.parse_args()
    run_dir = make_run_dir("escalation_diagnostics")
    get_logger(log_file=run_dir / "log.txt")
    dm = load_datamodule(load_config(args.data))
    benign = dm.labels.benign_index
    rows, miss_rows = [], []

    for spec in DETECTORS:
        params = {k: v for k, v in spec.items() if k != "kind"}
        model = TreeModel(spec["kind"], params, seed=0).fit(dm.X["train"], dm.y["train"])
        # calibration is fitted on the validation split and evaluated on the test split
        p_val, p_test = model.predict_proba(dm.X["val"]), model.predict_proba(dm.X["test"])
        y_val, y_test = dm.y["val"], dm.y["test"]
        pred_val, pred_test = p_val.argmax(1), p_test.argmax(1)
        ok_val, ok_test = pred_val == y_val, pred_test == y_test

        row = {"detector": model.name, "accuracy_test": float(ok_test.mean())}
        for name, s in uncertainty_scores(p_test).items():
            row[f"auroc_{name}"] = float(roc_auc_score(ok_test, s))  # wrong vs correct separation
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_val.max(1), ok_val.astype(float))
        row["ece_raw"] = ece(p_test.max(1), ok_test)
        row["ece_isotonic"] = ece(iso.predict(p_test.max(1)), ok_test)
        rows.append(row)

        conf = p_test.max(1)
        attack, is_benign = y_test != benign, y_test == benign
        for tau in TAUS:
            kept_benign = (pred_test == benign) & (conf >= tau)
            escalate = conf < tau
            miss_rows.append({
                "detector": model.name, "tau": tau,
                "attacks_kept_as_benign": float(kept_benign[attack].mean()),
                "benign_escalated": float(escalate[is_benign].mean()),
                "wrong_escalated": float(escalate[~ok_test].mean()),
                "correct_escalated": float(escalate[ok_test].mean()),
                "flows_escalated": float(escalate.mean()),
            })
        log.info(f"{model.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "detector"))

    pd.DataFrame(rows).to_csv(run_dir / "scores.csv", index=False)
    pd.DataFrame(miss_rows).to_csv(run_dir / "missed_attacks.csv", index=False)
    with pd.option_context("display.width", 220):
        log.info("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
        log.info("\n" + pd.DataFrame(miss_rows).round(3).to_string(index=False))
    save_json({"data": args.data, "detectors": DETECTORS}, run_dir / "config.json")


if __name__ == "__main__":
    main()
