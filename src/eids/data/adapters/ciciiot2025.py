from __future__ import annotations

import re

import pandas as pd

from eids.data.adapters.base import BENIGN_CATEGORY, FINE_LABEL, META_DEVICE, META_TIME, DatasetAdapter
from eids.utils.logging import get_logger

log = get_logger()

# Identifiers of the capture rather than behaviour: device identity, time, and raw address/port/protocol
# lists. Their *_count columns (how many distinct values in the window) are behavioural and are kept.
_ID_COLUMNS = {"device_name", "device_mac", "timestamp", "timestamp_start", "timestamp_end", "log_data-types"}
_LIST_COLUMN = re.compile(r"^network_(ips|macs|ports|protocols)_(all|dst|src)$")
_LABEL_COLUMNS = {"label_full", "label1", "label2", "label3", "label4"}
_PORT_SUFFIX = re.compile(r"-port-\d+$")


class CICIIoT2025Adapter(DatasetAdapter):
    """CIC IIoT 2025 (DataSense) processed files: per-device time-window samples (1-10 s).

    Fine label = attack scenario (label4) with the target port removed, so `dos_syn-flood-port-80` and
    `dos_syn-flood-port-1883` are one class; category = the part before the first underscore.
    """

    name = "ciciiot2025"
    benign_label = "benign"

    def _keep(self, column: str) -> bool:
        return column not in _ID_COLUMNS and not _LIST_COLUMN.match(column) and \
            (column not in _LABEL_COLUMNS or column == "label4")

    def _read(self, kind: str) -> pd.DataFrame:
        window = self.cfg.get("window_seconds", 1)
        path = self.cfg.get_path("root") / f"{kind}_samples_{window}sec.csv"
        log.info(f"[{self.name}] reading {path.name}")
        keep_meta = self.cfg.get("keep_meta", False)  # device and time, for device- or time-separated splits
        df = pd.read_csv(path, usecols=lambda c: self._keep(c) or (keep_meta and c in ("device_name", "timestamp")),
                         low_memory=False)
        if keep_meta:
            df[META_DEVICE] = df.pop("device_name").astype(str)
            start = df.pop("timestamp").astype(str).str.split("_", n=1).str[0]  # "<start>_<end>" of the window
            df[META_TIME] = pd.to_datetime(start, utc=True, format="ISO8601").astype("int64") / 1e9
        label = df.pop("label4").astype(str).str.strip().str.lower()
        df[FINE_LABEL] = self.benign_label if kind == "benign" else label.str.replace(_PORT_SUFFIX, "", regex=True)
        return df

    def load_raw(self) -> pd.DataFrame:
        return pd.concat([self._read("attack"), self._read("benign")], ignore_index=True)

    def leaky_columns(self) -> list[str]:
        return []  # identifiers are never read (see _keep)

    def category_of(self, fine_label: str) -> str:
        if fine_label == self.benign_label:
            return BENIGN_CATEGORY
        category = fine_label.split("_", 1)[0]
        return {"ddos": "DDoS", "dos": "DoS", "mitm": "MITM", "bruteforce": "BruteForce"}.get(category, category.title())
