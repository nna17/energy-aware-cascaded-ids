import numpy as np
import pandas as pd

from eids.data.cleaning import clean
from eids.data.splits import collect_chunks, device_split, stratified_split, temporal_split


def test_clean_drops_leaky_nonfinite_duplicates_and_conflicts():
    df = pd.DataFrame({
        "a": [1.0, 1.0, 2.0, np.inf, 3.0, 3.0],
        "ip": ["x", "y", "z", "w", "v", "u"],
        "label_fine": ["A", "A", "B", "A", "A", "B"],
    })
    out, report = clean(df, "label_fine", leaky_columns=["ip"])
    assert "ip" not in out.columns
    assert report.nonfinite_rows == 1
    assert report.exact_duplicates == 1      # rows 0 and 1 are identical once `ip` is gone
    assert report.conflicting_rows == 2      # a=3.0 labelled A and B
    assert out["a"].tolist() == [1.0, 2.0]


def test_stratified_split_keeps_rare_classes_everywhere():
    labels = np.array([0] * 1000 + [1] * 3)
    splits = stratified_split(labels, {"train": 0.7, "val": 0.1, "test": 0.2}, seed=0)
    for idx in splits.values():
        assert (labels[idx] == 1).sum() >= 1
    all_idx = np.concatenate(list(splits.values()))
    assert len(all_idx) == len(labels) and len(np.unique(all_idx)) == len(labels)


def test_collect_chunks_dedupes_and_caps_only_large_classes():
    chunks = [
        pd.DataFrame({"x": np.arange(30), "y": ["a"] * 27 + ["b"] * 3}),
        pd.DataFrame({"x": np.r_[np.arange(100, 130), [100]], "y": ["a"] * 27 + ["b"] * 3 + ["a"]}),
    ]
    out, report = collect_chunks(chunks, "y", max_rows=10, seed=0)
    assert out["y"].value_counts().to_dict() == {"a": 10, "b": 6}
    assert report.rows_read == 61 and report.duplicates_within_chunks == 1
    # the cap samples from every chunk, not only the first
    kept_a = out.loc[out["y"] == "a", "x"]
    assert (kept_a < 100).any() and (kept_a >= 100).any()


def test_collect_chunks_never_caps_uncapped_classes():
    chunk = pd.DataFrame({"x": np.arange(60), "y": ["a"] * 30 + ["BENIGN"] * 30})
    out, _ = collect_chunks([chunk], "y", max_rows=10, seed=0, uncapped=["BENIGN"])
    assert out["y"].value_counts().to_dict() == {"BENIGN": 30, "a": 10}


def test_ciciot2023_labels_in_both_spellings():
    from eids.data.adapters.ciciot2023 import CICIoT2023Adapter
    from eids.utils.config import Config

    adapter = CICIoT2023Adapter(Config())
    assert adapter.category_of("BENIGN") == "Benign"
    assert adapter.category_of("DDOS-ICMP_FLOOD") == "DDoS"
    assert adapter.category_of("DDoS-ICMP_Flood") == "DDoS"
    assert adapter.category_of("DOS-UDP_FLOOD") == "DoS"
    assert adapter.category_of("RECON-PINGSWEEP") == "Recon"
    assert adapter.category_of("UPLOADING_ATTACK") == "Web"
    assert adapter.category_of("MITM-ARPSPOOFING") == "Spoofing"


def test_pipeline_outputs(synthetic_dm):
    dm = synthetic_dm
    assert dm.X["train"].dtype == np.float32
    assert "flow_id" not in dm.feature_names
    assert dm.labels.benign_index is not None
    assert len(dm.labels.minority) >= 1
    for split in ("train", "val", "test"):
        assert set(np.unique(dm.y[split])) == set(range(dm.n_classes))


def test_clean_ignores_meta_columns_for_duplicates():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0], "meta_device": ["x", "y", "x"], "label_fine": ["b", "b", "b"]})
    out, report = clean(df, "label_fine", [])
    assert report.exact_duplicates == 1 and len(out) == 2


def test_temporal_split_puts_earliest_records_in_train():
    labels = np.array(["a"] * 10 + ["b"] * 10)
    times = np.r_[np.arange(10)[::-1], np.arange(10)].astype(float)
    split = temporal_split(labels, times, {"train": 0.7, "val": 0.1, "test": 0.2})
    for cls in ("a", "b"):
        t = {k: times[v][labels[v] == cls] for k, v in split.items()}
        assert t["train"].max() < t["val"].min() <= t["val"].max() < t["test"].min()


def test_device_split_holds_out_whole_devices_and_unseen_classes():
    rng = np.random.default_rng(0)
    devices = np.repeat([f"d{i}" for i in range(10)], 20)
    labels = rng.choice(["a", "b"], size=200)
    labels[devices == "d9"] = "only_on_d9"
    split, held_out = device_split(labels, devices, {"train": 0.7, "val": 0.1, "test": 0.2}, seed=1)
    assert set(devices[split["test"]]) <= set(held_out)
    assert not set(devices[split["train"]]) & set(held_out)
    assert "only_on_d9" not in set(labels[split["test"]])
