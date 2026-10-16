from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from eids.data.adapters.base import BENIGN_CATEGORY, FINE_LABEL, DatasetAdapter
from eids.utils.logging import get_logger

log = get_logger()

# Labels are normalised to upper case: the MERGED_CSV release writes `DDOS-ICMP_FLOOD` / `BENIGN`,
# the per-attack CSV release writes `DDoS-ICMP_Flood` / `BenignTraffic`.
_WEB = {"BROWSERHIJACKING", "BACKDOOR_MALWARE", "XSS", "UPLOADING_ATTACK", "SQLINJECTION", "COMMANDINJECTION"}
_SPOOFING = {"DNS_SPOOFING", "MITM-ARPSPOOFING"}
_PREFIXES = {"DDOS-": "DDoS", "DOS-": "DoS", "MIRAI-": "Mirai", "RECON-": "Recon"}


class CICIoT2023Adapter(DatasetAdapter):
    """CIC-IoT2023 CSVs (MERGED_CSV `Merged*.csv` or `part-*.csv`), read file by file as float32."""

    name = "ciciot2023"
    benign_label = "BENIGN"

    def _files(self) -> list[Path]:
        root = self.cfg.get_path("root")
        files = sorted(root.rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files under {root}")
        max_files = self.cfg.get("max_files")
        return files[:max_files] if max_files else files

    def _read(self, path: Path) -> pd.DataFrame:
        header = pd.read_csv(path, nrows=0).columns
        label_col = next(c for c in header if c.strip().lower() == "label")
        df = pd.read_csv(path, dtype={c: np.float32 for c in header if c != label_col})
        labels = df.pop(label_col).astype(str).str.strip().str.upper().replace({"BENIGNTRAFFIC": "BENIGN"})
        df[FINE_LABEL] = labels
        return df

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        files = self._files()
        for i, path in enumerate(files, 1):
            log.info(f"[{self.name}] reading {path.name} ({i}/{len(files)})")
            yield self._read(path)

    def load_raw(self) -> pd.DataFrame:
        return pd.concat(self.iter_chunks(), ignore_index=True)

    def leaky_columns(self) -> list[str]:
        # The CSV releases have no IPs/ports/timestamps; audit-driven additions go in the data config.
        return []

    def category_of(self, fine_label: str) -> str:
        label = fine_label.upper()
        if label == self.benign_label:
            return BENIGN_CATEGORY
        if label in _WEB:
            return "Web"
        if label in _SPOOFING:
            return "Spoofing"
        if label == "VULNERABILITYSCAN":
            return "Recon"
        if label == "DICTIONARYBRUTEFORCE":
            return "BruteForce"
        for prefix, category in _PREFIXES.items():
            if label.startswith(prefix):
                return category
        raise ValueError(f"[{self.name}] unknown label {fine_label!r}; add it to the category map")
