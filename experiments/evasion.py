"""Evasion of the device tier, and two defences (random audit escalation, a stricter benign threshold).

The deployment under attack is the joint deployment selected for a BLE sensor at a share of the energy of sending
every flow (device-battery profile, seed 0, chosen on validation). Its detectors are retrained from their specs.

Attacker (black box, feature space): a mimicry attacker moves every attack flow a fraction `alpha` of the way
towards the median benign flow on the features the device detector reads. The server reads all features, of which
the perturbed ones are a subset. For each alpha we report, on the test split:
  escape      share of attack flows accepted on the device as benign (never seen by the server)
  missed      share of attack flows that the whole cascade labels benign
  energy      device energy per flow (uJ), including audit escalations
Defences: audit a random share r of the flows the device decides locally (they are escalated anyway), and raise
the benign threshold so that more benign-looking flows are escalated.

    python experiments/evasion.py --data configs/data/ciciiot2025.yaml --pool-config configs/experiments/pool.yaml \
        --strong configs/experiments/pool_strong_ciciiot2025.yaml --fronts results/runs/final_cost_.../fronts.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eids.cascade.pool import make_detector, model_grid
from eids.data.datamodule import load_datamodule
from eids.utils.config import load_config
from eids.utils.io import make_run_dir
from eids.utils.logging import get_logger

log = get_logger()
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
AUDITS = [0.0, 0.01, 0.05, 0.1]
BENIGN_THRESHOLDS = [None, 0.9, 0.99]  # None: the threshold chosen by the search


def selected(fronts: pd.DataFrame, profile: str, budget: float, seed: int = 0) -> pd.Series:
    g = fronts[(fronts.profile == profile) & (fronts.seed == seed)]
    s = g[g.method == "single_model"]
    ref = s.loc[s.val_macro_f1.idxmax()]
    j = g[(g.method == "joint_nsga2") & (g.val_mean_cost <= budget * ref.val_mean_cost)]
    return j.loc[j.val_macro_f1.idxmax()]


def spec_for(name: str, configs: list[Path], n_features: int) -> dict:
    for c in configs:
        for spec in model_grid(load_config(c).models, n_features):
            if make_detector(spec, 0).name == name:
                return spec
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pool-config", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--fronts", required=True)
    parser.add_argument("--profile", default="device_ble")
    parser.add_argument("--budget", type=float, default=0.3)
    parser.add_argument("--radio-uj", type=float, default=400.0, help="device energy per transmitted flow (uJ)")
    args = parser.parse_args()

    dm = load_datamodule(load_config(args.data))
    dep = selected(pd.read_csv(args.fronts), args.profile, args.budget)
    parts = [p.split(":") for p in dep.deployment.split(" -> ")]
    if len(parts) != 2:
        raise SystemExit(f"expected a two-tier deployment, got {dep.deployment}")
    (_, dev_name), (_, up_name) = parts
    thresholds = np.array(json.loads(dep.thresholds)[0])
    configs = [Path(args.pool_config), Path(args.strong)]
    dev_spec, up_spec = spec_for(dev_name, configs, dm.n_features), spec_for(up_name, configs, dm.n_features)
    log.info(f"deployment {dep.deployment}; device ops {dep.mean_cost:.0f} nJ/flow mean")
    device = make_detector(dev_spec, 0).fit(dm.X["train"], dm.y["train"], balanced=dev_spec.get("balanced", True))
    upper = make_detector(up_spec, 0).fit(dm.X["train"], dm.y["train"], balanced=up_spec.get("balanced", True))

    X, y = dm.X["test"], dm.y["test"]
    benign = dm.labels.benign_index
    attack = y != benign
    k = dev_spec.get("n_features") or dm.n_features  # features the device reads (ranked first)
    target = np.median(dm.X["train"][dm.y["train"] == benign], axis=0)
    rng = np.random.default_rng(0)
    audit_draw = rng.random(len(y))

    rows = []
    for alpha in ALPHAS:
        Xa = X.copy()
        Xa[np.ix_(attack, np.arange(k))] += alpha * (target[:k] - Xa[np.ix_(attack, np.arange(k))])
        pd_dev = device.predict_proba(Xa)
        pred, conf = pd_dev.argmax(1), pd_dev.max(1)
        up_pred = upper.predict_proba(Xa).argmax(1)
        for tb in BENIGN_THRESHOLDS:
            th = thresholds.copy()
            if tb is not None:
                th[benign] = max(th[benign], tb)
            local = conf >= th[pred]
            for r in AUDITS:
                decided_local = local & (audit_draw >= r)
                final = np.where(decided_local, pred, up_pred)
                escalated = ~decided_local
                rows.append({
                    "alpha": alpha, "benign_threshold": "searched" if tb is None else tb, "audit": r,
                    "escape": float(np.mean(decided_local[attack] & (pred[attack] == benign))),
                    "missed": float(np.mean(final[attack] == benign)),
                    "benign_fpr": float(np.mean(final[~attack] != benign)),
                    "escalated": float(np.mean(escalated)),
                    "device_uj": float(np.mean(escalated) * args.radio_uj),
                })
        log.info(f"alpha={alpha}: escape {rows[-len(BENIGN_THRESHOLDS) * len(AUDITS)]['escape']:.4f}")
    run = make_run_dir(f"evasion_{Path(args.data).stem}")
    df = pd.DataFrame(rows)
    df.to_csv(run / "results.csv", index=False)
    (run / "deployment.json").write_text(json.dumps({"deployment": dep.deployment, "device_features": k,
                                                     "profile": args.profile, "budget": args.budget}), "utf-8")
    log.info(f"wrote {run}\n" + df[(df.audit == 0) & (df.benign_threshold == "searched")].round(4).to_string())


if __name__ == "__main__":
    main()
