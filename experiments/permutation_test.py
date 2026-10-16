"""Does a 1D-CNN exploit any spatial structure in flow features?

Retrain the CNN and a control MLP on random column orders. The MLP's spread is pure training noise
(an MLP has no notion of column order). If the CNN's spread is similar, the convolution is not using
neighbouring columns, which supports leaving CNNs out of our search space.

    python experiments/permutation_test.py --config configs/experiments/permutation_test.yaml
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from eids.data.datamodule import DataModule, load_datamodule
from eids.eval.evaluator import evaluate
from eids.models.baselines.cnn1d_diab import CNN1D
from eids.models.mlp import MLP
from eids.training.classifier import train_classifier
from eids.utils.config import load_config
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger
from eids.utils.seed import resolve_device, set_seed

log = get_logger()


def permuted(dm: DataModule, order: np.ndarray) -> DataModule:
    return dataclasses.replace(
        dm, X={s: x[:, order] for s, x in dm.X.items()}, feature_names=[dm.feature_names[i] for i in order]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/permutation_test.yaml")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()

    exp = load_config(args.config)
    device = resolve_device(exp.get("device", "auto"))
    run_dir = make_run_dir(exp.name)
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")
    train_cfg = load_config(exp.train)
    rows = []

    for data_path in args.datasets or exp.datasets:
        dm = load_datamodule(load_config(data_path))
        for seed in exp.seeds:
            rng = np.random.default_rng(seed)
            orders = [np.arange(dm.n_features)] + [rng.permutation(dm.n_features) for _ in range(exp.n_permutations)]
            for p, order in enumerate(orders):
                pdm = permuted(dm, order)
                models = {
                    "cnn1d": CNN1D(dm.n_features, dm.n_classes, list(exp.cnn.channels), exp.cnn.kernel_size,
                                   exp.cnn.hidden),
                    "mlp_control": MLP(dm.n_features, dm.n_classes, hidden=[exp.cnn.hidden, exp.cnn.hidden]),
                }
                for kind, model in models.items():
                    set_seed(seed)
                    train_classifier(model, pdm, train_cfg, device, seed, name=f"{kind}_perm{p}")
                    m = evaluate(model, pdm, "test", device)
                    rows.append({"dataset": dm.name, "seed": seed, "permutation": p, "model": kind,
                                 "macro_f1": m["macro_f1"], "minority_recall": m["minority_recall"]})

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "results.csv", index=False)
    summary = df.groupby(["dataset", "model"])[["macro_f1", "minority_recall"]].agg(["mean", "std", "min", "max"])
    summary.to_csv(run_dir / "summary.csv")
    log.info("Permutation test (test split):\n" + summary.to_string())


if __name__ == "__main__":
    main()
