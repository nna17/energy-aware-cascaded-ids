"""Remove leaky columns, non-finite rows and duplicates. Every removal is counted in the report."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from eids.data.adapters.base import META_PREFIX


@dataclass
class CleaningReport:
    rows_in: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    nonfinite_rows: int = 0
    exact_duplicates: int = 0
    conflicting_rows: int = 0  # identical features, different labels
    rows_out: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def clean(
    df: pd.DataFrame,
    label_col: str,
    leaky_columns: list[str],
    drop_conflicting_duplicates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(df))

    present = [c for c in leaky_columns if c in df.columns]
    df = df.drop(columns=present)
    report.dropped_columns = present

    # meta_* columns (device, time) describe where a record came from; they are never features and do not make
    # two otherwise identical records distinct.
    feature_cols = [c for c in df.columns if c != label_col and not c.startswith(META_PREFIX)]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    values = df[numeric].to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(values).all(axis=1)
    report.nonfinite_rows = int((~finite).sum())
    df = df.loc[finite]

    categorical = [c for c in feature_cols if c not in numeric]
    if categorical:
        df[categorical] = df[categorical].astype("string").fillna("missing")

    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + [label_col])
    report.exact_duplicates = before - len(df)

    if drop_conflicting_duplicates:
        conflict = df.duplicated(subset=feature_cols, keep=False)
        report.conflicting_rows = int(conflict.sum())
        df = df.loc[~conflict]

    report.rows_out = len(df)
    return df.reset_index(drop=True), report
