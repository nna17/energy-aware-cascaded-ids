"""Tune strong server-tier detectors on the validation split, and write a pool-extension config.

LightGBM is tuned with Optuna (TPE) on a stratified training sample with early stopping on validation;
class weighting (balanced or none) is part of the search. Two settings are kept: the best validation
macro-F1 and the best validation minority recall. Unrestricted forests with and without class weights are
added untuned. The test split is never used.

    python experiments/tune_detectors.py --data configs/data/ciciiot2025.yaml --trials 30
Writes configs/experiments/pool_strong_<dataset>.yaml (extends results/pools_v2/<dataset>/seed<n>).
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import optuna
import yaml

from eids.data.datamodule import load_datamodule
from eids.eval.metrics import fast_metrics
from eids.utils.config import load_config, resolve_path
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()


def stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        k = max(1, int(round(n * len(members) / len(y))))
        idx.append(rng.choice(members, size=min(k, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--sample", type=int, default=200_000)
    args = parser.parse_args()
    cfg = load_config(args.data)
    dm = load_datamodule(cfg)
    run_dir = make_run_dir(f"tune_{dm.name}")
    get_logger(log_file=run_dir / "log.txt")
    tr = stratified_sample(dm.y["train"], args.sample, 0)
    Xtr, ytr, Xva, yva = dm.X["train"][tr], dm.y["train"][tr], dm.X["val"], dm.y["val"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        balanced = trial.suggest_categorical("balanced", [True, False])
        model = lgb.LGBMClassifier(n_estimators=1000, min_child_weight=1.0, class_weight="balanced" if balanced else None,
                                   random_state=0, n_jobs=-1, verbose=-1, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = model.predict(Xva)
        f1, min_rec, fpr = fast_metrics(yva, np.asarray(pred).astype(np.int64), dm.n_classes,
                                        dm.labels.minority, dm.labels.benign_index)
        trial.set_user_attr("n_estimators", int(model.best_iteration_ or 1000))
        trial.set_user_attr("minority_recall", min_rec)
        trial.set_user_attr("fpr", fpr)
        log.info(f"trial {trial.number}: macroF1={f1:.4f} minRec={min_rec:.4f} FPR={fpr:.4f} {params} balanced={balanced} "
                 f"rounds={trial.user_attrs['n_estimators']}")
        return f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials)
    by_f1 = study.best_trial
    by_min = max(study.trials, key=lambda t: t.user_attrs.get("minority_recall", -1))

    def spec(trial: optuna.trial.FrozenTrial, tag: str) -> dict:
        p = trial.params
        return {"kind": "lightgbm", "n_estimators": [trial.user_attrs["n_estimators"]], "max_leaves": [p["num_leaves"]],
                "learning_rate": round(p["learning_rate"], 5), "min_child_samples": p["min_child_samples"],
                "colsample_bytree": round(p["colsample_bytree"], 4), "balanced": p["balanced"], "tag": tag}

    models = [spec(by_f1, "tuned_f1"), spec(by_min, "tuned_minrec"),
              {"kind": "random_forest", "n_estimators": [100], "max_leaves": [None], "balanced": False, "tag": "unw"},
              {"kind": "random_forest", "n_estimators": [100], "max_leaves": [None], "balanced": True, "tag": "bal"}]
    pool_cfg = {"name": f"pool_strong_{dm.name}", "dataset": args.data, "seeds": [0, 1, 2], "balanced": True,
                "out_dir": "results/pools_v2", "models": models}
    out = resolve_path(f"configs/experiments/pool_strong_{dm.name}.yaml")
    yaml.safe_dump(pool_cfg, open(out, "w"), sort_keys=False)
    save_json({"best_macro_f1": {"value": by_f1.value, **by_f1.params, **by_f1.user_attrs},
               "best_minority_recall": {"macro_f1": by_min.value, **by_min.params, **by_min.user_attrs}},
              run_dir / "best.json")
    log.info(f"best macro-F1 {by_f1.value:.4f}; best minority recall {by_min.user_attrs['minority_recall']:.4f}; wrote {out}")


if __name__ == "__main__":
    main()
