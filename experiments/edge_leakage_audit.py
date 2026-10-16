"""Edge-IIoTset leakage audit: how much of the usual performance comes from per-capture counters?

After the standard drop list (hosts, ports, timestamps, payloads) and deduplication, train the same
LightGBM on (a) only TCP sequence/acknowledgement numbers, checksums and stream indices, (b) only the
remaining behavioural features, (c) both. Also count how many records stay distinct without the counters.

    python experiments/edge_leakage_audit.py
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from eids.data.adapters.base import FINE_LABEL
from eids.data.adapters.edge_iiotset import EdgeIIoTsetAdapter
from eids.data.splits import stratified_split
from eids.utils.config import load_config
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()
COUNTERS = ["tcp.ack", "tcp.ack_raw", "tcp.seq", "tcp.checksum", "icmp.checksum", "icmp.seq_le",
            "udp.stream", "mbtcp.trans_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/edge_iiotset.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    run_dir = make_run_dir("edge_leakage_audit")
    get_logger(log_file=run_dir / "log.txt")

    adapter = EdgeIIoTsetAdapter(load_config(args.config))
    raw = adapter.load_raw()
    n_raw = len(raw)
    df = raw.drop(columns=[c for c in adapter.leaky_columns() if c in raw.columns]).drop_duplicates()
    behavioural = [c for c in df.columns if c not in COUNTERS + [FINE_LABEL] and pd.api.types.is_numeric_dtype(df[c])]
    stats = {
        "rows_raw": n_raw,
        "rows_after_standard_drop_and_dedup": len(df),
        "distinct_rows_without_counters": len(df[behavioural + [FINE_LABEL]].drop_duplicates()),
        "n_counter_columns": len(COUNTERS), "n_behavioural_columns": len(behavioural),
    }
    stats["share_duplicates_without_counters"] = 1 - stats["distinct_rows_without_counters"] / stats["rows_after_standard_drop_and_dedup"]
    log.info(stats)

    y = df[FINE_LABEL].astype("category").cat.codes.to_numpy()
    rows = []
    for seed in args.seeds:
        split = stratified_split(y, {"train": 0.7, "val": 0.1, "test": 0.2}, seed)
        for name, cols in (("counters_only", COUNTERS), ("behavioural_only", behavioural),
                           ("both_standard_protocol", COUNTERS + behavioural)):
            X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(np.float32)
            model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, min_child_weight=1.0, class_weight="balanced",
                                       random_state=seed, n_jobs=-1, verbose=-1).fit(X[split["train"]], y[split["train"]])
            pred, true = model.predict(X[split["test"]]), y[split["test"]]
            rows.append({"seed": seed, "features": name, "n_columns": len(cols),
                         "accuracy": accuracy_score(true, pred), "macro_f1": f1_score(true, pred, average="macro")})
            log.info(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "results.csv", index=False)
    save_json(stats, run_dir / "stats.json")
    log.info("\n" + res.groupby("features")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
