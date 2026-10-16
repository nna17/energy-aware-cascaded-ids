"""A1: train the candidate tier models once per seed and store their per-flow predictions.

    python experiments/build_pool.py --config configs/experiments/pool.yaml
"""
from __future__ import annotations

import argparse

from eids.cascade.pool import build_pool, model_grid
from eids.data.datamodule import load_datamodule
from eids.utils.config import load_config, resolve_path
from eids.utils.logging import get_logger
from eids.utils.seed import set_seed

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/pool.yaml")
    args = parser.parse_args()
    exp = load_config(args.config)
    data_cfg = load_config(exp.dataset)
    dm = load_datamodule(data_cfg)
    specs = model_grid(exp.models, dm.n_features)
    log.info(f"pool of {len(specs)} models on {dm.name}")
    for seed in exp.seeds:
        set_seed(seed)
        out = build_pool(specs, dm, data_cfg, seed, resolve_path(exp.out_dir) / dm.name / f"seed{seed}",
                         balanced=exp.get("balanced", True))
        log.info(f"saved {out}")


if __name__ == "__main__":
    main()
