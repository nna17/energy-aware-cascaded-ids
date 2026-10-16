"""Bounded-memory loading with a per-class cap, and stratified splits that keep rare classes in every split."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

_KEY = "__sample_key"


@dataclass
class CollectReport:
    rows_read: int = 0
    duplicates_within_chunks: int = 0
    rows_kept: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def collect_chunks(chunks: Iterable[pd.DataFrame], label_col: str, max_rows: int | None,
                   seed: int, uncapped: Iterable[str] = ()) -> tuple[pd.DataFrame, CollectReport]:
    """Concatenate chunks, dropping duplicates inside each chunk.

    With `max_rows`, keep a uniform random sample of at most `max_rows` rows per class across all
    chunks (each row gets a random key; the smallest keys per class survive), so memory stays bounded.
    Only large classes shrink; rare classes and classes in `uncapped` (e.g. benign) are kept whole.
    """
    uncapped = set(uncapped)
    rng = np.random.default_rng(seed)
    report = CollectReport()
    kept: pd.DataFrame | None = None
    parts: list[pd.DataFrame] = []

    for chunk in chunks:
        report.rows_read += len(chunk)
        deduped = chunk.drop_duplicates()
        report.duplicates_within_chunks += len(chunk) - len(deduped)
        if not max_rows:
            parts.append(deduped)
            continue
        keep_all = deduped[label_col].isin(uncapped)
        parts.append(deduped[keep_all])
        capped = deduped[~keep_all].assign(**{_KEY: rng.random(int((~keep_all).sum()))})
        kept = capped if kept is None else pd.concat([kept, capped], ignore_index=True)
        kept = kept.sort_values(_KEY).groupby(label_col, sort=False).head(max_rows)

    if max_rows:
        parts.append(kept.drop(columns=_KEY))
    df = pd.concat(parts, ignore_index=True)
    report.rows_kept = len(df)
    return df, report


def temporal_split(labels: np.ndarray, times: np.ndarray, fractions: dict[str, float]) -> dict[str, np.ndarray]:
    """Per class, the earliest records go to the first split and the latest to the last, so that records of
    one attack session that are close in time do not end up on both sides of the train/test boundary."""
    names = list(fractions)
    weights = np.array([fractions[n] for n in names], dtype=float)
    weights /= weights.sum()
    out: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        idx = idx[np.argsort(times[idx], kind="stable")]
        n = len(idx)
        sizes = np.floor(weights * n).astype(int)
        if n >= len(names):
            sizes = np.maximum(sizes, 1)
        sizes[0] = n - sizes[1:].sum()
        for name, part in zip(names, np.split(idx, np.cumsum(sizes)[:-1])):
            out[name].append(part)
    return {n: np.sort(np.concatenate(parts)) for n, parts in out.items()}


def device_split(labels: np.ndarray, devices: np.ndarray, fractions: dict[str, float], seed: int,
                 test_device_share: float = 0.2) -> tuple[dict[str, np.ndarray], list[str]]:
    """Test = every record of a random share of devices never seen in training; train/val = a stratified split of
    the remaining devices' records. Test records of classes that no training device has are dropped, because no
    detector can name a class it never saw. Returns the split and the held-out devices."""
    rng = np.random.default_rng(seed)
    names = sorted(np.unique(devices))
    held_out = sorted(rng.choice(names, size=max(1, round(test_device_share * len(names))), replace=False).tolist())
    is_test = np.isin(devices, held_out)
    rest = np.flatnonzero(~is_test)
    inner = {k: v for k, v in fractions.items() if k != "test"}
    parts = stratified_split(labels[rest], inner, seed)
    split = {k: rest[v] for k, v in parts.items()}
    seen = np.unique(labels[split["train"]])
    test = np.flatnonzero(is_test)
    split["test"] = test[np.isin(labels[test], seen)]
    return split, held_out


def stratified_split(labels: np.ndarray, fractions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Split indices per class. Classes with >= 3 samples get at least one sample in every split."""
    names = list(fractions)
    weights = np.array([fractions[n] for n in names], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {n: [] for n in names}

    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n = len(idx)
        sizes = np.floor(weights * n).astype(int)
        if n >= len(names):
            sizes = np.maximum(sizes, 1)
        sizes[0] = n - sizes[1:].sum()  # remainder goes to the first split (train)
        bounds = np.cumsum(sizes)[:-1]
        for name, part in zip(names, np.split(idx, bounds)):
            out[name].append(part)

    return {n: np.sort(np.concatenate(parts)) for n, parts in out.items()}
